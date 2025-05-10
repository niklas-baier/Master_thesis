import torch
from einops import rearrange
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transcribe import transcribe_results
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, \
    TrainerControl, WhisperForConditionalGeneration, EarlyStoppingCallback, Trainer
from tqdm.auto import tqdm # For progress bars
import gc # Garbage collector
from torch.utils.data import DataLoader, Subset
from train import DataCollatorSpeechSeq2SeqWithPadding
import wandb
from visualizations import exponential_decay, fit_loss_function, calculate_mean_wer
# (Keep as is - seems correct)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# --- 2. Domain Discriminator ---
class DomainDiscriminator(nn.Module):
    """Simple MLP Discriminator"""
    def __init__(self, input_dim, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # Output a single logit for Binary Cross Entropy with Logits
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # Ensure input is flattened if necessary (e.g., after pooling)
        # Assuming input x is already [batch_size, input_dim]
        return self.layer(x)

# --- 3. Whisper Domain Adapter Wrapper (Simplified - focus on components) ---
# We won't use a complex wrapper for forward pass logic,
# instead handling it in the training loop for clarity.
# We'll pass the individual components to the training function.

# --- 4. Model Setup ---
def setup_models(whisper_model_name="distil-whisper/distil-large-v3", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")

    # Load Whisper model
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    # We might need the processor later for data prep or generation
    # processor = WhisperProcessor.from_pretrained(whisper_model_name)

    # Determine the dimensionality of Whisper encoder's output
    encoder_output_dim = whisper_model.config.d_model

    # Create the domain discriminator
    discriminator = DomainDiscriminator(input_dim=encoder_output_dim).to(device)

    # Create the Gradient Reversal Layer
    grl = GradientReversalLayer(lambda_=1.0) # Lambda can be adjusted/scheduled

    # No separate task_head needed if using Whisper's built-in decoder/projection

    return whisper_model, discriminator, grl, device # Removed task_head
def warmup(whisper_model, discriminator, grl,collator,train_datasets,eval_dataset, device, lr=1e-5, weight_decay=0.01,lambda_domain_loss=0.1, BATCH_SIZE=1):
    warmup_losses = []
    mean_of_hidden_states = []
    warmup_epochs = 100
    seed = 42
    torch.manual_seed(42)
    orderings = torch.randint(0, 2, (100000,1))
    opposite = torch.where(orderings < 1 ,1,0)
    labels = torch.cat([orderings, opposite], dim=1).to(device)
    counter = 0
    warmup_datasets_clean = train_datasets[0].shuffle(seed=seed).select(range(train_datasets[0].shape[0]))
    warmup_datasets_dirty = train_datasets[1].shuffle(seed=seed).select(range(train_datasets[1].shape[0]))
    warmup_dataloader_A= DataLoader(warmup_datasets_clean, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2, drop_last=True )
    warmup_dataloader_B= DataLoader(warmup_datasets_dirty, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2, drop_last=True )
    ACCUMULATION_STEPS = 4
    best_dict_path =  'early_stopping_simple_disc.pth'
    #optimizer_disc = optim.SGD(discriminator.parameters(), lr=1e-5) # Use potentially different lr for disc, SGD experiments showed that momemntum term is needed
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=5e-5, weight_decay=weight_decay)
    criterion_domain = nn.BCEWithLogitsLoss() # positive samples get increased by factor 6
    classification_accuracy_counter = 0
    validation_accuracy = classify_results(whisper_model= whisper_model, discriminator=discriminator, grl=grl, collator=collator, test_dataset=eval_dataset, device=device, lr=lr, weight_decay=weight_decay, BATCH_SIZE=BATCH_SIZE)
    len_dataloader = min(len(warmup_dataloader_A), len(warmup_dataloader_B))
    for epoch in range(warmup_epochs):
        # Use zip to iterate through both dataloaders simultaneously
        #warmup_dataloader_A= DataLoader(warmup_datasets_clean, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2, drop_last=True )
        #warmup_dataloader_B= DataLoader(warmup_datasets_dirty, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2, drop_last=True )
        data_iterator = iter(zip(warmup_dataloader_A, warmup_dataloader_B))
        pbar = tqdm(range(len_dataloader), desc=f"Epoch {epoch+1} Training")
        if classification_accuracy_counter > 5:
            break

        for step in pbar:
            try:
                batch_A, batch_B = next(data_iterator)
                if labels[counter,0]== 0:
                   batch_A['input_features'] = torch.cat((batch_A['input_features'], batch_B['input_features']), dim=0)
                   combined_labels = torch.tensor([[0.0], [1.0]]).to(device)
                else:
                    batch_A['input_features'] = torch.cat((batch_B['input_features'], batch_A['input_features']), dim=0)
                    combined_labels = torch.tensor([[1.0], [0.0]]).to(device)
                with torch.no_grad():
                    hidden_states = whisper_model.model.encoder(batch_A['input_features'].to(device)).last_hidden_state
                mean_state = torch.mean(hidden_states,dim=1)
                mean_of_hidden_states.append(mean_state)
                domain_classification = discriminator(mean_state)
                warmup_classification_loss = criterion_domain(domain_classification, combined_labels.float())
                warmup_classification_loss.backward()
                warmup_losses.append(warmup_classification_loss.item())
                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len_dataloader:
                    optimizer_disc.step()
                    optimizer_disc.zero_grad()  # Reset gradients
                counter = counter + 1
            except:
                print("weird exception occured")
        data = [[x,y] for x,y in zip(list(range(len(warmup_losses))), warmup_losses)]
        table = wandb.Table(data=data, columns = ['steps', "loss"])
        wandb.log({"warmup_losses": wandb.plot.line(table, "steps", "loss", title ="warmup_loss")})
        #fit_loss_function(warmup_losses)
        new_validation_accuracy = classify_results(whisper_model= whisper_model, discriminator=discriminator, grl=grl, collator=collator, test_dataset=eval_dataset, device=device, lr=lr, weight_decay=weight_decay, BATCH_SIZE=BATCH_SIZE)
        if new_validation_accuracy  >= validation_accuracy:
            print("higher validation accuracy")
            validation_accuracy = new_validation_accuracy
            torch.save(discriminator.state_dict(), best_dict_path)
            classification_accuracy_counter = 0
        else:
            classification_accuracy_counter = classification_accuracy_counter + 1

    print(f"validation_accuracy is {validation_accuracy} warmup finished")
    discriminator.load_state_dict(torch.load(best_dict_path))
    return discriminator

def classify_results(whisper_model, discriminator, grl,collator,test_dataset, device, lr=1e-5, weight_decay=0.01, BATCH_SIZE=1):
    seed = 42
    torch.manual_seed(42)
    len_clean_labels = int(test_dataset.shape[0]//6)
    clean_labels = torch.zeros((len_clean_labels,1))
    dirty_labels = torch.ones((test_dataset.shape[0]-len_clean_labels,1))
    labels = torch.cat([clean_labels,dirty_labels],dim = 0)
    counter = 0
    test_dataloader= DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    len_dataloader = len(test_dataloader)
    classifications = []
    for epoch in range(1):

        for batch in tqdm(test_dataloader, desc= "classification of eval/ test_set"):
            with torch.no_grad():
                hidden_states = whisper_model.model.encoder(batch['input_features'].to(device)).last_hidden_state
                mean_state = torch.mean(hidden_states,dim=1)
                domain_classification = discriminator(mean_state)
                prediction = torch.where(domain_classification > 0,1 ,0)
                classifications.append(prediction)
                domain_labels_A = rearrange (labels[counter,:], "b -> b 1")
                counter= counter + 1
        def calculate_accuracy(predictions, labels):
            assert(predictions.shape == labels.shape)
            result = (predictions==labels).int()
            accuracy = torch.sum(result)/result.shape[0]
            return accuracy
        pred_t = torch.tensor(classifications)
        print(pred_t)
        print("evaluation")
        labels = torch.zeros_like(pred_t)
        accuracy = calculate_accuracy(pred_t, labels)
        wandb.log({"validation_accuracy_untrained": accuracy})
        return accuracy


# --- 5. Training Loop (Revised) ---
def train_adversarial(trainer, discriminator, grl,collator,eval_dataset,
                      train_datasets, test_dataset, device,run_details,
                      num_epochs=2, lr=1e-5, weight_decay=0.01,
                      lambda_domain_loss=0.1, BATCH_SIZE=1): # Weight for domain loss in encoder update

    # --- Optimizers ---
    # Optimizer for Whisper encoder + decoder/projection (main task + adversarial)
    # Filter parameters to only optimize the encoder if desired, or optimize all whisper params
    # Here we optimize all parameters of the whisper_model
    whisper_model = trainer.model
    #discriminator = warmup(whisper_model=whisper_model, discriminator=discriminator, grl=grl,collator=collator,train_datasets=train_datasets,eval_dataset=eval_dataset, device=device, lr=lr, weight_decay=weight_decay,lambda_domain_loss=lambda_domain_loss, BATCH_SIZE=BATCH_SIZE)
    accuracy = classify_results(whisper_model=whisper_model, discriminator=discriminator, grl=grl,collator=collator,test_dataset=test_dataset, device=device, lr=lr, weight_decay=weight_decay, BATCH_SIZE=BATCH_SIZE)
    optimizer_main = optim.AdamW(whisper_model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_disc = optim.AdamW(discriminator.parameters(), lr=lr, weight_decay=weight_decay) # Use potentially different lr for disc
    criterion_task = nn.CrossEntropyLoss(ignore_index=-100) # Use Whisper's default ignore index
    criterion_domain = nn.BCEWithLogitsLoss()
    clean_dataloader= DataLoader(train_datasets[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dirty_dataloader= DataLoader(train_datasets[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_A, dataloader_B = clean_dataloader, dirty_dataloader
    len_dataloader = min(len(dataloader_A), len(dataloader_B))
    early_stopping_discriminative_model_path = 'best_discriminative_model.pth'
    early_stopping_discriminator_path = 'best_discriminator.pth'
    print(f"Training for {num_epochs} epochs with {len_dataloader} steps per epoch.")
    early_stopping_discriminative_counter = 0
    base_wer = 100
    for epoch in range(num_epochs):
        transcription_path = transcribe_results(trainer=trainer, test_dataset=eval_dataset, run_details = run_details)
        mean_validation_wer = calculate_mean_wer(transcription_path)
        if mean_validation_wer <= base_wer:
            wandb.log({"mean_validation_wer": mean_validation_wer})
            base_wer = mean_validation_wer
            early_stopping_discriminative_counter = 0
            torch.save(whisper_model.state_dict(),early_stopping_discriminative_model_path)
            torch.save(discriminator.state_dict(), early_stopping_discriminator_path)
        else:
            early_stopping_discriminative_counter = early_stopping_discriminative_counter + 1
        if early_stopping_discriminative_counter >=5:
            whisper_model.load_state_dict(early_stopping_discriminative_model_path)
            return whisper_model

        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        whisper_model.train()
        discriminator.train()

        total_task_loss_epoch = 0.0
        total_domain_loss_epoch = 0.0
        total_discriminator_loss_epoch = 0.0
        correct_domain_predictions_epoch = 0
        total_domain_samples_epoch = 0
        len_dataloader = min (len(dataloader_A), len(dataloader_B))
        # Use zip to iterate through both dataloaders simultaneously
        data_iterator = iter(zip(dataloader_A, dataloader_B))
        pbar = tqdm(range(len_dataloader), desc=f"Epoch {epoch+1} Training")

        for step in pbar:
            try:
                batch_A, batch_B = next(data_iterator)
            except StopIteration:
                break

            # --- Prepare Data ---
            # Source domain data (Domain 0)
            inputs_A = batch_A['input_features'].to(device) # Input features
            labels_A = batch_A['labels'].to(device)         # Task labels for source
            domain_labels_A = torch.zeros(inputs_A.size(0), 1, device=device, dtype=torch.float) # Domain 0

            # Target domain data (Domain 1)
            inputs_B = batch_B['input_features'].to(device) # Input features
            # labels_B might exist but are not used for unsupervised domain adaptation task loss
            domain_labels_B = torch.ones(inputs_B.size(0), 1, device=device, dtype=torch.float) # Domain 1

            # Combine inputs for potentially faster processing if memory allows
            # Note: This requires careful handling if batch sizes differ or if task loss
            # is only computed on domain A. We'll process separately for clarity.

            # ======== Phase 1: Update Discriminator ========
            # Goal: Maximize discriminator's ability to classify domains correctly.
            # Freeze Whisper model parameters for this phase (optional but common)
            # We achieve this by using .detach() on features below.
            optimizer_disc.zero_grad()

            # Process Domain A for Discriminator
            with torch.no_grad(): # Get features without tracking gradients back to Whisper
                 encoder_outputs_A = whisper_model.model.encoder(inputs_A)
                 features_A = encoder_outputs_A.last_hidden_state
            # Apply mean pooling over sequence dimension
            pooled_features_A = features_A.mean(dim=1).detach() # Detach here!
            domain_logits_A = discriminator(pooled_features_A)
            loss_disc_A = criterion_domain(domain_logits_A, domain_labels_A)

            # Process Domain B for Discriminator
            with torch.no_grad():
                 encoder_outputs_B = whisper_model.model.encoder(inputs_B)
                 features_B = encoder_outputs_B.last_hidden_state
            pooled_features_B = features_B.mean(dim=1).detach() # Detach here!
            domain_logits_B = discriminator(pooled_features_B)
            loss_disc_B = criterion_domain(domain_logits_B, domain_labels_B)

            # Combine discriminator losses and update
            loss_disc = (loss_disc_A + loss_disc_B) / 2
            loss_disc.backward()
            optimizer_disc.step()

            # Track discriminator accuracy (optional)
            with torch.no_grad():
                 preds_A = (torch.sigmoid(domain_logits_A) > 0.5).float()
                 preds_B = (torch.sigmoid(domain_logits_B) > 0.5).float()
                 correct_domain_predictions_epoch += (preds_A == domain_labels_A).sum().item()
                 correct_domain_predictions_epoch += (preds_B == domain_labels_B).sum().item()
                 total_domain_samples_epoch += inputs_A.size(0) + inputs_B.size(0)


            # ======== Phase 2: Update Whisper Model (Encoder + Task Head) ========
            # Goal: Minimize task loss (on source domain A) and fool the discriminator.
            optimizer_main.zero_grad()

            # --- Task Loss (Domain A only) ---
            # Recompute encoder outputs for Domain A (gradients needed now)
            encoder_outputs_A = whisper_model.model.encoder(inputs_A)
            features_A = encoder_outputs_A.last_hidden_state # Shape: [batch_A, seq_len, hidden_dim]

            # Pass through Whisper Decoder and Projection Layer for task loss
            from transformers.models.whisper.modeling_whisper import shift_tokens_right
            decoder_input_ids_A = shift_tokens_right(labels_A, whisper_model.config.pad_token_id, whisper_model.config.decoder_start_token_id)

            decoder_outputs_A = whisper_model.model.decoder(
                input_ids=decoder_input_ids_A,
                encoder_hidden_states=features_A # Use the computed features
            )
            lm_logits_A = whisper_model.proj_out(decoder_outputs_A.last_hidden_state)

            # Calculate task loss (ignore padding)
            # Ensure correct slicing for loss calculation
            vocab_size = lm_logits_A.size(-1)
            shifted_logits_A = lm_logits_A[:, :-1, :].contiguous()
            shifted_labels_A = labels_A[:, 1:].contiguous()
            task_loss = criterion_task(shifted_logits_A.view(-1, vocab_size), shifted_labels_A.view(-1))


            # --- Adversarial Domain Loss (Both Domains) ---
            # Process Domain A through GRL and Discriminator
            pooled_features_A = features_A.mean(dim=1) # Pool features from Domain A
            reversed_pooled_features_A = grl(pooled_features_A)
            domain_logits_adv_A = discriminator(reversed_pooled_features_A)
            # We want the encoder to produce features that the discriminator thinks are from the *opposite* domain
            # So, for domain A (true label 0), the adversarial target is 1.
            # domain_loss_adv_A = criterion_domain(domain_logits_adv_A, torch.ones_like(domain_labels_A)) # Alternative view
            # Or simply use the standard labels and let GRL handle the gradient reversal effect:
            domain_loss_adv_A = criterion_domain(domain_logits_adv_A, domain_labels_A)


            # Process Domain B through Encoder, GRL, and Discriminator
            encoder_outputs_B = whisper_model.model.encoder(inputs_B) # Compute features for B
            features_B = encoder_outputs_B.last_hidden_state
            pooled_features_B = features_B.mean(dim=1) # Pool features from Domain B
            reversed_pooled_features_B = grl(pooled_features_B)
            domain_logits_adv_B = discriminator(reversed_pooled_features_B)
            # For domain B (true label 1), adversarial target is 0.
            # domain_loss_adv_B = criterion_domain(domain_logits_adv_B, torch.zeros_like(domain_labels_B)) # Alternative view
            # Standard labels with GRL:
            domain_loss_adv_B = criterion_domain(domain_logits_adv_B, domain_labels_B)


            # Combine domain losses for the main update
            domain_loss_adv = (domain_loss_adv_A + domain_loss_adv_B) / 2

            # --- Combine Losses for Main Optimizer ---
            # The GRL will reverse the gradient of domain_loss_adv before it reaches the encoder
            total_loss_main = task_loss + lambda_domain_loss * domain_loss_adv

            # Backward pass and step for the main model (Whisper)
            total_loss_main.backward()
            optimizer_main.step()

            # --- Logging ---
            total_task_loss_epoch += task_loss.item()
            total_domain_loss_epoch += domain_loss_adv.item() # Adversarial loss component for main update
            total_discriminator_loss_epoch += loss_disc.item() # Discriminator's own loss

            pbar.set_postfix({
                "Task Loss": f"{task_loss.item():.4f}",
                "Domain Loss (Adv)": f"{domain_loss_adv.item():.4f}",
                "Disc Loss": f"{loss_disc.item():.4f}"
            })

            # Clean up (optional, can help in memory-constrained environments)
            del inputs_A, labels_A, domain_labels_A, inputs_B, domain_labels_B
            del features_A, features_B, pooled_features_A, pooled_features_B
            del domain_logits_A, domain_logits_B, domain_logits_adv_A, domain_logits_adv_B
            del loss_disc_A, loss_disc_B, loss_disc, task_loss, domain_loss_adv_A, domain_loss_adv_B, domain_loss_adv, total_loss_main
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()


        # --- End of Epoch ---
        avg_task_loss = total_task_loss_epoch / len_dataloader
        avg_domain_loss = total_domain_loss_epoch / len_dataloader
        avg_discriminator_loss = total_discriminator_loss_epoch / len_dataloader
        domain_accuracy = correct_domain_predictions_epoch / total_domain_samples_epoch if total_domain_samples_epoch > 0 else 0

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Task Loss: {avg_task_loss:.4f}")
        print(f"  Avg Domain Loss (for Encoder Update): {avg_domain_loss:.4f}")
        print(f"  Avg Discriminator Loss: {avg_discriminator_loss:.4f}")
        print(f"  Domain Discriminator Accuracy: {domain_accuracy:.4f}")

        # Optional: Adapt lambda for GRL (e.g., schedule)
        # p = float(step + epoch * len_dataloader) / (num_epochs * len_dataloader)
        # new_lambda = 2. / (1. + np.exp(-10. * p)) - 1 # Example schedule
        # grl.set_lambda(new_lambda)
        # print(f"  Set GRL lambda to: {grl.lambda_:.4f}")


    # --- Save Model ---
    print("Training finished. Saving model...")
    # You might want to save the discriminator too, or just the adapted Whisper model
    #torch.save(whisper_model.state_dict(), f"whisper_adapted_dann_{epoch}epochs.pth")
    #torch.save(discriminator.state_dict(), f"discriminator_dann_{epoch}epochs.pth")
    # torch.save(discriminator.state_dict(), "discriminator_dann.pth")

    print("Model saved.")


# --- Example Usage (requires actual dataloaders) ---
if __name__ == "__main__":
    # 1. Setup Models
    whisper_model, discriminator, grl, device = setup_models()

    # 2. Prepare Dataloaders (Replace with your actual dataloaders)
    # Ensure your dataloaders yield dictionaries with keys like 'input_features' and 'labels'
    # Example placeholder:

    # 3. Run Training
    # processor = WhisperProcessor.from_pretrained("distil-whisper/distil-large-v3")
    # adapted_model = WhisperForConditionalGeneration.from_pretrained("./whisper_adapted_dann_1epochs.pth") # This won't work directly, need to load state dict
    adapted_model = WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3")
    # with torch.no_grad():
    #     # Assume dummy_input_features is prepared correctly (e.g., from dataloader_B)
    #     dummy_input_features = next(iter(dataloader_B))['input_features'].to(device)
    #     # Generate using the main model's generate function
    #     generated_ids = adapted_model.generate(inputs=dummy_input_features)
        # generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # print("Generated Text (Example):", generated_text)
