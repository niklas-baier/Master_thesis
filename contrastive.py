# --- End Debugging ---
import torch
import warnings
from test_Whisper import warn_with_traceback, get_tensor_gpu_memory
from transcribe import transcribe_evaluation
import wandb
import meeteval
import numpy as np
from einops import rearrange, reduce
from torch.amp import GradScaler
import torch.nn as nn
from train import DataCollatorSpeechSeq2SeqWithPadding
import torch.nn.functional as F
import torch.optim as optim
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from tqdm.auto import tqdm # For progress bars
import math # For inf check
from torch.utils.data import DataLoader, Subset
from train import get_cached_components
from visualizations import exponential_decay, fit_loss_function, calculate_mean_wer
# --- 1. InfoNCE Loss Function ---
def calculate_infonce_loss(features_A, features_B, temperature=0.1, device="cpu"):
    """
    Calculates InfoNCE loss for batches from two domains.
    Assumes features_A and features_B have the same shape (N, D)
    Implicitly treats sample i from A and sample i from B as positive pairs.
    """
    N = features_A.shape[0]
    if N <= 1:
        # Need at least 2 samples (1 from each domain ideally) for contrastive loss
        print("Warning: Batch size <= 1, skipping InfoNCE loss calculation.")
        return torch.tensor(0.0, device=device, requires_grad=True) # Return zero loss

    D = features_A.shape[1]

    # Concatenate features from both domains
    features = torch.cat([features_A, features_B], dim=0) # Shape: (2N, D)

    # Normalize features for cosine similarity stability
    features = F.normalize(features, p=2, dim=1)

    # Calculate pairwise cosine similarity
    # similarity_matrix[i, j] = similarity between sample i and sample j
    similarity_matrix = torch.matmul(features, features.T) # Shape: (2N, 2N)

    # Check for NaNs or Infs which can occur with temperature scaling
    if torch.isnan(similarity_matrix).any() or torch.isinf(similarity_matrix).any():
        print("Warning: NaN or Inf detected in similarity matrix before temperature scaling.")
         # Handle NaN/Inf, e.g., replace them or return zero loss
         # For simplicity, return zero loss here, but investigate the cause
        return torch.tensor(0.0, device=device, requires_grad=True)


    # Scale similarity by temperature
    similarity_matrix = similarity_matrix / temperature

    # --- Create labels for cross-entropy ---
    # The ground truth positive for sample i (from A, index 0 to N-1) is sample i+N (from B).
    # The ground truth positive for sample i+N (from B, index N to 2N-1) is sample i (from A).
    targets = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(device) # Shape: (2N,)

    # --- Mask out self-similarity ---
    # We don't want sample i to be compared with itself.
    self_mask = torch.eye(2*N, dtype=torch.bool, device=device)
    similarity_matrix = similarity_matrix.masked_fill_(self_mask, -1e4) # Fill diagonal with large negative number

    # Check for NaNs or Infs after masking and scaling (important!)
    # Large negative numbers are okay, but NaNs/Infs from division by ~zero temp or bad inputs are not.
    if not torch.isfinite(similarity_matrix).all():
        print("Warning: NaN or Inf detected in similarity matrix AFTER scaling/masking.")
        # Handle NaN/Inf
        # Count non-finite values
        num_non_finite = (~torch.isfinite(similarity_matrix)).sum().item()
        print(f"  Number of non-finite values: {num_non_finite} out of {similarity_matrix.numel()}")
        # Option: Clamp values or replace NaNs
        # similarity_matrix = torch.nan_to_num(similarity_matrix, nan=-1e9, posinf=1e9, neginf=-1e9)
        # For simplicity, let's return 0 loss, but this needs careful debugging if it happens often
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Calculate cross-entropy loss
    # F.cross_entropy expects logits of shape (batch_size, num_classes) and targets of shape (batch_size,)
    # Here, batch_size is 2N (samples), num_classes is 2N (potential positives/negatives)
    loss = F.cross_entropy(similarity_matrix, targets, reduction='mean')

    # Check if loss is finite
    if not math.isfinite(loss.item()):
        print("Warning: InfoNCE loss is not finite.")
        return torch.tensor(0.0, device=device, requires_grad=True) # Return zero loss

    return loss


# --- 2. Model Setup ---
def setup_model(whisper_model_name="distil-whisper/distil-large-v3", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")

    # Load Whisper model and processor
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    processor = WhisperProcessor.from_pretrained(whisper_model_name)

    return whisper_model, processor, device

# --- 3. Placeholder DataLoaders ---
# Replace this with your actual data loading logic
# Crucially, ensure batches from both dataloaders have the same size if possible
def get_dataloaders(batch_size=4, processor=None):
    # Dummy data - replace with real processed Whisper inputs
    # Input features should be pre-processed log-mel spectrograms
    # For WhisperForConditionalGeneration, we also need decoder inputs/labels
    dummy_seq_len = 3000 # Audio sequence length
    dummy_token_len = 50  # Target token sequence length

    # Domain A (Source) - Assume has task labels
    dummy_input_features_A = torch.randn(batch_size, 80, dummy_seq_len)
    # Prepare dummy labels - use processor if available for EOS, BOS etc.
    # Typically, labels are input_ids shifted, with padding ignored (-100)
    dummy_labels_A = torch.randint(100, processor.tokenizer.vocab_size, (batch_size, dummy_token_len))
    # Set padding token ID to -100 (ignored by loss function)
    dummy_labels_A[:, dummy_token_len // 2:] = -100
    # Add EOS token if processor is available (highly recommended)
    if processor:
        dummy_labels_A[:, (dummy_token_len // 2) -1] = processor.tokenizer.eos_token_id


    dataloader_A = [(dummy_input_features_A, dummy_labels_A)] * 10 # Dummy loop

    # Domain B (Target) - May or may not have labels, but need features
    dummy_input_features_B = torch.randn(batch_size, 80, dummy_seq_len)
    # If domain B also has labels (e.g., for multi-task learning or just evaluation)
    dummy_labels_B = torch.randint(100, processor.tokenizer.vocab_size, (batch_size, dummy_token_len))
    dummy_labels_B[:, dummy_token_len // 2:] = -100
    if processor:
        dummy_labels_B[:, (dummy_token_len // 2) -1] = processor.tokenizer.eos_token_id

    # We only strictly *need* input_features_B for contrastive loss,
    # but pass labels if available/needed for standard loss calculation on B
    dataloader_B = [(dummy_input_features_B, dummy_labels_B)] * 10 # Dummy loop
    print(f"Returning dummy dataloaders with batch size {batch_size}")
    print(f"Dummy input feature shape: {dummy_input_features_A.shape}")
    print(f"Dummy label shape: {dummy_labels_A.shape}")
    return dataloader_A, dataloader_B
def generate_contrastive_batch(counter, BATCH_SIZE, random_batch_generator, batch_B, batch_C, batch_D, batch_E, batch_F):
    batch_indexes = random_batch_generator[counter*BATCH_SIZE:counter*BATCH_SIZE+BATCH_SIZE]
    arange_batch = torch.arange(0,batch_indexes.shape[0])
    merged_batches = rearrange([ batch_B['input_features'], batch_C['input_features'], batch_D['input_features'], batch_E['input_features'], batch_F['input_features']], 'b o s l -> b o s l')
    selected_tensors = [merged_batches[b.long(), s.long(),:,:] for b, s in zip(batch_indexes, arange_batch)]
    batch_tensor = torch.cat([tensor for tensor in selected_tensors], dim=0)
    return batch_tensor
# --- 4. Training Loop ---
def train_infonce(whisper_model, processor,collator, train_dataset,eval_dataset, device,BATCH_SIZE,trainer,run_details,
        num_epochs=5, lr=5e-5, weight_decay=0.01,
        infonce_weight=0.1, temperature=0.1):

    # --- Optimizer ---
    #pre_sim = asses_similarity(whisper_model=whisper_model, processor=processor, collator=collator, train_dataset=train_dataset,device=device,BATCH_SIZE=BATCH_SIZE,num_epochs=1)
    whisper_model.gradient_checkpointing_enable()
    whisper_model.to("cuda")
    optimizer = optim.AdamW(whisper_model.parameters(), lr=lr, weight_decay=weight_decay)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random_batch_generator = torch.randint_like(torch.zeros(100000,1),low=0, high=5)
    counter = 0
    #torch.autograd.set_detect_anomaly(True)clean_dataloader= DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    # establish baseline
    #_, base_wer = evaluate_dataset(whisper_model=whisper_model, dataset= train_dataset[0],BATCH_SIZE=BATCH_SIZE, collator=collator, device=device)
    base_wer = 32
    print(f"base_wer is {base_wer}")
    early_stopping_contrastive_model_path= 'contrastive_early.pth'

    dataloader_A = DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    dataloader_B = DataLoader(train_dataset[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    dataloader_C = DataLoader(train_dataset[2], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    dataloader_D = DataLoader(train_dataset[3], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    dataloader_E = DataLoader(train_dataset[4], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    dataloader_F = DataLoader(train_dataset[5], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=1 )
    len_dataloader = min(len(dataloader_A), len(dataloader_B))
    if len_dataloader == 0:
        print("Error: Dataloaders are empty.")
        return

    print(f"Training for {num_epochs} epochs with {len_dataloader} steps per epoch.")
    print(f"InfoNCE Loss weight: {infonce_weight}, Temperature: {temperature}")
    early_stopping_discriminative_counter = 0
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        whisper_model.to("cuda")
        dataloader_A = DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        dataloader_B = DataLoader(train_dataset[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        dataloader_C = DataLoader(train_dataset[2], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        dataloader_D = DataLoader(train_dataset[3], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        dataloader_E = DataLoader(train_dataset[4], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        dataloader_F = DataLoader(train_dataset[5], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
        len_dataloader = min(len(dataloader_A), len(dataloader_B))

        total_standard_loss = 0.0
        total_contrastive_loss = 0.0
        _, mean_validation_wer = transcribe_evaluation(trainer=trainer, test_dataset=eval_dataset, run_details = run_details)
        if mean_validation_wer <= base_wer:
            wandb.log({"mean_validation_wer": mean_validation_wer})
            base_wer = mean_validation_wer
            early_stopping_discriminative_counter = 0
            torch.save(whisper_model.state_dict(),early_stopping_contrastive_model_path)
        else:
            wandb.log({"mean_validation_wer": mean_validation_wer})
            early_stopping_discriminative_counter = early_stopping_discriminative_counter + 1
        if early_stopping_discriminative_counter >=5:
            whisper_model.load_state_dict(early_stopping_discriminative_model_path)
            print("exiting contrastive loss")
            return whisper_model



        # Use zip to iterate through batches from both domains simultaneously
        # Requires dataloaders yield batches of the same size for simple pairing
        data_iterator = iter(zip(dataloader_A, dataloader_B,dataloader_C, dataloader_D, dataloader_E, dataloader_F))

        pbar = tqdm(range(len_dataloader), desc="Training Steps")
        for step in pbar:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            try:
                batch_A, batch_B,batch_C,batch_D, batch_E, batch_F = next(data_iterator)
                batch_lookup_dict = { 1: batch_B, 2: batch_C, 3:batch_D, 4:batch_E, 5:batch_F}
                batch_B['input_features'] = generate_contrastive_batch(counter=counter, BATCH_SIZE=batch_A['input_features'].shape[0], random_batch_generator=random_batch_generator, batch_B=batch_B, batch_C=batch_C, batch_D=batch_D, batch_E=batch_E, batch_F=batch_F)
                counter = counter + 1

            except StopIteration:
                break

            # --- Prepare Data ---
            with torch.autocast('cuda'):
                inputs_A = batch_A['input_features'].to(device)
                labels_A = batch_A['labels'].to(device)

                # Ensure padding in labels is -100
                labels_A = labels_A.masked_fill(labels_A == processor.tokenizer.pad_token_id, -100)


                inputs_B = batch_B['input_features'].to(device)
                # Labels for B might not be needed for contrastive, but run forward pass anyway
                # If you want standard loss on B too, provide labels_B here
                labels_B = batch_B['labels'].to(device)
                #labels_B = labels_B.masked_fill(labels_B == processor.tokenizer.pad_token_id, -100)

                # --- Forward Pass & Standard Loss ---
                # Process source domain A - calculate standard transcription loss

                # --- Pooling ---
                outputs_A = whisper_model(input_features=inputs_A, labels=labels_A)
                features_encoder_A_t = outputs_A.encoder_last_hidden_state
                standard_loss_A_t = outputs_A.loss # This is the seq2seq CE loss
                features_encoder_A = features_encoder_A_t
                standard_loss_A = standard_loss_A_t
                del features_encoder_A_t
                del standard_loss_A_t
                # Process target domain B - primarily to get encoder features
                # No labels provided = no standard loss calculated for B here
                # Provide labels=labels_B if you want to calculate standard loss on B too
                outputs_B = whisper_model(input_features=inputs_B,labels= labels_B) # No labels needed if only getting features


                # --- Get Encoder Features ---
                # Access the last hidden state of the encoder
                features_encoder_A = outputs_A.encoder_last_hidden_state # (batch, seq_len, hidden_dim)
                features_encoder_B = outputs_B.encoder_last_hidden_state # (batch, seq_len, hidden_dim)
                del inputs_B, labels_B
                if features_encoder_A is None or features_encoder_B is None:
                    print("Error: Could not retrieve encoder hidden states. Check model configuration.")
                    continue


                # --- Pooling ---
                # Pool encoder features (e.g., mean pooling)
                pooled_features_A = features_encoder_A.mean(dim=1) # (batch, hidden_dim)
                pooled_features_B = features_encoder_B.mean(dim=1) #
                # Pool encoder features (e.g., mean pooling)

                # --- InfoNCE Loss ---
                contrastive_loss = calculate_infonce_loss(pooled_features_A, pooled_features_B,temperature=temperature, device=device)

                # --- Combine Losses ---
                # Adjust weighting as needed
                # Here, we only use standard loss from domain A
                infonce_weight = 1
                #total_loss = standard_loss_A + infonce_weight * contrastive_loss
                # Alternative: If using standard loss on both:
                standard_loss_B = outputs_B.loss
                total_loss = (standard_loss_A + standard_loss_B)/2 #+ infonce_weight * contrastive_loss

                # --- Backpropagation & Optimization ---
            scaler.scale(total_loss).backward()
            #total_loss.backward()


            max_grad = 0.0
            has_nan_inf_grad = False
            '''
            for param in whisper_model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_nan_inf_grad = True
                    break
                current_max = param.grad.abs().max().item()
                if current_max > max_grad:
                    max_grad = current_max
            print(f"Max gradient value before clipping/step: {max_grad}")
            if has_nan_inf_grad:
                print("WARNING: NaN or Inf detected in gradients!")
            #
                    # Optional: Gradient clipping
            #torch.nn.utils.clip_grad_norm_(whisper_model.parameters(), 1.0)'''
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()

            #n_float16_params = [name for name, param in model.named_parameters() if param.dtype != torch.float16]
            #print(str(len(n_float16_params))+ 'this is the num of float32params')
            # --- Logging ---
            total_standard_loss += standard_loss_A.item()
            total_contrastive_loss += contrastive_loss.item()
            wandb.log({"epoch/avg_standard_loss": standard_loss_A.item(),"epoch/avg_contrastive_loss": contrastive_loss.item(),"epoch/avg_combined_loss": standard_loss_A.item()+ 0.1*contrastive_loss.item(),"epoch": epoch})

            #del  outputs_B, outputs_A, inputs_A, inputs_B, labels_A,labels_B, features_encoder_A, features_encoder_B,Batch_A, Batch_B, Batch_C, Batch_D,Batch_E, Batch_F
            #dataloader_A, dataloader_B, dataloader_C, dataloader_D, dataloader_E,




            pbar.set_postfix({
                "Std Loss": f"{standard_loss_A.item():.4f}",
                "NCE Loss": f"{contrastive_loss.item():.4f}",
                "Total Loss": f"{total_loss.item():.4f}"
                })
            torch.cuda.empty_cache()

        # --- End of Epoch ---
        avg_standard_loss = total_standard_loss / len_dataloader
        avg_contrastive_loss = total_contrastive_loss / len_dataloader

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Standard Loss (Domain A): {avg_standard_loss:.4f}")
        print(f"  Avg InfoNCE Loss: {avg_contrastive_loss:.4f}")
        #torch.cuda.memory._dump_snapshot("contrastive{step}.pickle")
    #results, mean_wer = evaluate_dataset(whisper_model=whisper_model, dataset= train_dataset[0],BATCH_SIZE=BATCH_SIZE, collator=collator, device=device)
   # print(f'comparison after training {mean_wer} to pretraining {base_wer}')
    #after_sim = asses_similarity(whisper_model=whisper_model, processor=processor, collator=collator, train_dataset=train_dataset,device=device,BATCH_SIZE=BATCH_SIZE,num_epochs=1)
   # print(f'comparison after training {after_sim} to pretraining {pre_sim}')
    return whisper_model

def asses_similarity(whisper_model, processor,collator, train_dataset, device,BATCH_SIZE,
        num_epochs=1):

    torch.manual_seed(40)
    torch.cuda.manual_seed(40)
    random_batch_generator = torch.randint_like(torch.zeros(100000,1),low=0, high=5)
    counter = 0
    base_results, base_wer = evaluate_dataset(whisper_model=whisper_model, dataset= train_dataset[0],BATCH_SIZE=BATCH_SIZE, collator=collator, device=device)
    print(f"base_wer is {base_wer}")
    dataloader_A = DataLoader(train_dataset[0], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_B = DataLoader(train_dataset[1], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_C = DataLoader(train_dataset[2], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_D = DataLoader(train_dataset[3], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_E = DataLoader(train_dataset[4], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    dataloader_F = DataLoader(train_dataset[5], batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    len_dataloader = min(len(dataloader_A), len(dataloader_B))
    if len_dataloader == 0:
        print("Error: Dataloaders are empty.")
        return

    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        whisper_model.to("cuda")
        cosines = []

        total_standard_loss = 0.0
        total_contrastive_loss = 0.0

        data_iterator = iter(zip(dataloader_A, dataloader_B,dataloader_C, dataloader_D, dataloader_E, dataloader_F))

        pbar = tqdm(range(len_dataloader), desc="Training Steps")
        for step in pbar:
            try:
                batch_A, batch_B,batch_C,batch_D, batch_E, batch_F = next(data_iterator)
                batch_lookup_dict = { 1: batch_B, 2: batch_C, 3:batch_D, 4:batch_E, 5:batch_F}
                batch_B['input_features'] = generate_contrastive_batch(counter=counter, BATCH_SIZE=batch_A['input_features'].shape[0], random_batch_generator=random_batch_generator, batch_B=batch_B, batch_C=batch_C, batch_D=batch_D, batch_E=batch_E, batch_F=batch_F)
                counter = counter + 1

            except StopIteration:
                break

            # --- Prepare Data ---
            with torch.autocast('cuda'):
                inputs_A = batch_A['input_features'].to(device)
                labels_A = batch_A['labels'].to(device)
                # Ensure padding in labels is -100
                labels_A = labels_A.masked_fill(labels_A == processor.tokenizer.pad_token_id, -100)


                inputs_B = batch_B['input_features'].to(device)
                # Labels for B might not be needed for contrastive, but run forward pass anyway
                # If you want standard loss on B too, provide labels_B here
                labels_B = batch_B['labels'].to(device)
                labels_B = labels_B.masked_fill(labels_B == processor.tokenizer.pad_token_id, -100)

                # --- Forward Pass & Standard Loss ---
                # Process source domain A - calculate standard transcription loss

                # --- Pooling ---
                outputs_A = whisper_model(input_features=inputs_A, labels=labels_A)
                outputs_B = whisper_model(input_features=inputs_B,labels= labels_B) # No labels needed if only getting features


                # --- Get Encoder Features ---
                # Access the last hidden state of the encoder
                features_encoder_A = outputs_A.encoder_last_hidden_state # (batch, seq_len, hidden_dim)
                features_encoder_B = outputs_B.encoder_last_hidden_state # (batch, seq_len, hidden_dim)

                if features_encoder_A is None or features_encoder_B is None:
                    print("Error: Could not retrieve encoder hidden states. Check model configuration.")
                    continue


                # --- Pooling ---
                # Pool encoder features (e.g., mean pooling)
                pooled_features_A = features_encoder_A.mean(dim=1) # (batch, hidden_dim)
                pooled_features_B = features_encoder_B.mean(dim=1) #
                cosine_sim = nn.CosineSimilarity(dim=1)
                cos = cosine_sim(pooled_features_A, pooled_features_B)
                cosines.append(cos)
            cos_tensor = torch.cat([x for x in cosines])
            mean_cosine_similarity = reduce(cos, 'b -> 1', 'mean')
            wandb.log({"average_cosinge_similarity": mean_cosine_similarity,"epoch": epoch})
            print(f"cosine similiarity is {mean_cosine_similarity}")
            torch.cuda.empty_cache()
            return mean_cosine_similarity






        print(f"Epoch {epoch+1} Summary:")





def evaluate_dataset(whisper_model, dataset, BATCH_SIZE, collator, device="cuda"):
    tokenizer,_, processor = get_cached_components()
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor,whisper_model.config.decoder_start_token_id )
    dataloader_evaluate = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collator, num_workers=2 )
    data_evaluate_iterator = iter(dataloader_evaluate)
    pbar = tqdm(range(len(dataloader_evaluate)), desc="Training Steps")
    results = []
    labels = []
    for step in pbar:
        try:
            batch = next(data_evaluate_iterator)
            predictions = whisper_model(input_features = batch['input_features'].to(device), labels = batch['labels'].to(device)).logits.detach()
            results.append(processor.tokenizer.batch_decode(torch.argmax(predictions, dim = -1), skip_special_tokens=True))
            labels.append(processor.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True))
        except Exception:
            print("")


    flattened_results = [item for sublist in results for item in sublist]
    flattened_labels = [item for sublist in labels for item in sublist]
    wer_description = [ meeteval.wer.wer.siso.siso_word_error_rate(x,y) for x, y in zip (flattened_results,flattened_labels)]
    wers = [x.error_rate for x in wer_description]
    mean_wer = np.mean(wers)


    return results, mean_wer


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL = "distil-whisper/distil-large-v3"
    #WHISPER_MODEL = "openai/whisper-large-v3" # Choose your Whisper model
    BATCH_SIZE = 2 # Keep relatively small for demonstration; ensure > 1
                   # Ensure dataloader_A and dataloader_B use the SAME batch size
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5 # Standard fine-tuning LR for Whisper can work
    WEIGHT_DECAY = 0.01
    INFONCE_WEIGHT = 0.1 # Weight for the contrastive loss term
    TEMPERATURE = 0.07 # Common temperature value for InfoNCE

    # --- Setup ---
    whisper_model, processor, device = setup_model(WHISPER_MODEL)
    dataloader_A, dataloader_B = get_dataloaders(BATCH_SIZE, processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor,trainer.model.config.decoder_start_token_id )
    test_dataloader= DataLoader(test_dataset, batch_size=2, collate_fn=collator, num_workers=2 )

    # --- Train ---
    torch.cuda.memory._dump_snapshot('snapshot.pickle')
    train_infonce(whisper_model, processor, test_dataloader, test_dataloader, "cuda",
            num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            infonce_weight=INFONCE_WEIGHT, temperature=TEMPERATURE)

    print("\nTraining finished.")
    # Add code here to save your adapted whisper_model weights if needed
    # torch.save(whisper_model.state_dict(), "infonce_adapted_whisper_model.pth")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import random
import numpy as np

def calculate_nt_xent_loss(clean_features, noisy_features, temperature=0.07, device='cuda'):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
    Treats clean[i] and noisy[i] as positive pairs, all others as negatives.
    """
    batch_size = clean_features.size(0)

    # Normalize features to unit sphere
    clean_features = F.normalize(clean_features, dim=1)
    noisy_features = F.normalize(noisy_features, dim=1)

    # Concatenate clean and noisy features
    # Shape: [2*batch_size, feature_dim]
    features = torch.cat([clean_features, noisy_features], dim=0)

    # Compute cosine similarity matrix for all pairs
    # Shape: [2*batch_size, 2*batch_size]
    similarity_matrix = torch.mm(features, features.t()) / temperature

    # Create mask for positive pairs
    # clean[i] is positive with noisy[i] and vice versa
    batch_indices = torch.arange(batch_size, device=device)
    positive_mask = torch.zeros((2 * batch_size, 2 * batch_size), device=device, dtype=torch.bool)

    # Clean samples (first batch_size) are positive with corresponding noisy samples (second batch_size)
    positive_mask[batch_indices, batch_indices + batch_size] = True
    # Noisy samples are positive with corresponding clean samples
    positive_mask[batch_indices + batch_size, batch_indices] = True

    # Create mask to exclude self-similarity (diagonal)
    self_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)

    # Mask out self-similarity from similarity matrix
    similarity_matrix = similarity_matrix.masked_fill(self_mask, float('-inf'))

    # Extract positive similarities
    positive_similarities = similarity_matrix[positive_mask].view(2 * batch_size, -1)

    # For NT-Xent, we compute log-softmax over all similarities except self
    # Then take the positive similarity values
    log_prob = F.log_softmax(similarity_matrix, dim=1)
    positive_log_prob = log_prob[positive_mask].view(2 * batch_size, -1)

    # NT-Xent loss is negative log probability of positive pairs
    loss = -positive_log_prob.mean()

    return loss

def calculate_multi_positive_nt_xent_loss(anchor_features, positive_features_list, temperature=0.07, device='cuda'):
    """
    NT-Xent loss with multiple positives (different far-field versions of same utterance).
    Each clean sample has multiple positive pairs from different microphones.
    """
    batch_size = anchor_features.size(0)
    num_positives = len(positive_features_list)

    # Normalize anchor features
    anchor_features = F.normalize(anchor_features, dim=1)

    # Normalize and concatenate all positive features
    positive_features = []
    for pos_feat in positive_features_list:
        positive_features.append(F.normalize(pos_feat, dim=1))

    # Concatenate: [anchor, pos1, pos2, ..., posN]
    # Shape: [(1 + num_positives) * batch_size, feature_dim]
    all_features = torch.cat([anchor_features] + positive_features, dim=0)

    # Compute similarity matrix
    similarity_matrix = torch.mm(all_features, all_features.t()) / temperature

    # Create positive mask
    total_samples = (1 + num_positives) * batch_size
    positive_mask = torch.zeros((total_samples, total_samples), device=device, dtype=torch.bool)

    batch_indices = torch.arange(batch_size, device=device)

    # Anchor samples (first batch_size) are positive with all corresponding positive samples
    for i in range(num_positives):
        start_idx = (i + 1) * batch_size
        end_idx = start_idx + batch_size
        positive_mask[batch_indices, start_idx + batch_indices] = True
        positive_mask[start_idx + batch_indices, batch_indices] = True

    # Create self-mask to exclude diagonal
    self_mask = torch.eye(total_samples, device=device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(self_mask, float('-inf'))

    # Compute NT-Xent loss
    log_prob = F.log_softmax(similarity_matrix, dim=1)

    # Only consider anchor and positive samples for loss (not pos-pos pairs)
    anchor_indices = torch.arange(batch_size, device=device)
    pos_indices = torch.cat([torch.arange(batch_size, device=device) + (i + 1) * batch_size
                           for i in range(num_positives)])

    relevant_indices = torch.cat([anchor_indices, pos_indices])
    relevant_positive_mask = positive_mask[relevant_indices][:, relevant_indices]
    relevant_log_prob = log_prob[relevant_indices][:, relevant_indices]

    # Extract positive log probabilities
    positive_log_prob = relevant_log_prob[relevant_positive_mask]
    loss = -positive_log_prob.mean()

    return loss

class EnhancedWhisperTrainer:
    def __init__(self, whisper_model, processor, device, temperature=0.07):
        self.whisper_model = whisper_model
        self.processor = processor
        self.device = device
        self.temperature = temperature

    def get_encoder_features(self, input_features, pool_method='attention'):
        """Extract and pool encoder features with different pooling strategies"""
        with torch.no_grad():
            encoder_outputs = self.whisper_model.model.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        if pool_method == 'mean':
            return hidden_states.mean(dim=1)
        elif pool_method == 'max':
            return hidden_states.max(dim=1)[0]
        elif pool_method == 'attention':
            # Learnable attention pooling
            attention_weights = torch.softmax(
                torch.sum(hidden_states * hidden_states.mean(dim=1, keepdim=True), dim=2),
                dim=1
            )
            return torch.sum(hidden_states * attention_weights.unsqueeze(2), dim=1)
        else:
            return hidden_states.mean(dim=1)
import random
import numpy as np
from datasets import Dataset
def train_improved_contrastive_aligned(
    whisper_model, processor, collator, train_datasets, eval_dataset, device,
    batch_size=4, num_epochs=20, lr=5e-5, weight_decay=0.01,
    contrastive_weight=0.3, noisy_asr_weight=0.5, temperature=0.07, warmup_epochs=1,
    use_curriculum=False, gradient_accumulation_steps=4, use_multi_positive=True,
    patience=3, min_delta=0.001, shuffle_strategy="synchronized"
):
    """
    Improved contrastive training with proper shuffling strategies.
    
    Args:
        shuffle_strategy (str): One of:
            - "synchronized": Shuffle all datasets with same random order (maintains alignment)
            - "epoch_reshuffle": Re-shuffle indices at start of each epoch
            - "none": No shuffling (original behavior)
    """

    # Setup
    whisper_model.train()
    whisper_model.gradient_checkpointing_enable()
    whisper_model.to(device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(whisper_model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()
    
    total_steps = len(train_datasets[0]) * num_epochs // batch_size
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps)

    # Early stopping variables
    best_val_asr_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Create validation dataloader (no shuffling needed)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=collator,
        shuffle=False, num_workers=2, pin_memory=True
    )

    # Get dataset size for index management
    dataset_size = len(train_datasets[0])
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # Curriculum learning
        if use_curriculum:
            progress = min(1.0, (epoch + 1) / warmup_epochs)
            current_contrastive_weight = contrastive_weight * progress
            current_noisy_asr_weight = noisy_asr_weight * progress
        else:
            current_contrastive_weight = contrastive_weight
            current_noisy_asr_weight = noisy_asr_weight

        # Generate shuffled indices for this epoch
        if shuffle_strategy == "synchronized":
            # Same shuffle order for all datasets to maintain alignment
            epoch_indices = list(range(dataset_size))
            random.shuffle(epoch_indices)
            
            # Create subset datasets with shuffled indices
            shuffled_datasets = []
            for dataset in train_datasets:
                shuffled_subset = Subset(dataset, epoch_indices)
                shuffled_datasets.append(shuffled_subset)
                
        elif shuffle_strategy == "epoch_reshuffle":
            # Different shuffle for each dataset - breaks strict alignment but creates more diverse pairs
            shuffled_datasets = []
            for dataset in train_datasets:
                epoch_indices = list(range(dataset_size))
                random.shuffle(epoch_indices)
                shuffled_subset = Subset(dataset, epoch_indices)
                shuffled_datasets.append(shuffled_subset)
                
        else:  # shuffle_strategy == "none"
            shuffled_datasets = train_datasets

        # Create dataloaders for this epoch
        dataloaders = []
        for dataset in shuffled_datasets:
            dataloader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=collator,
                shuffle=False,  # Already shuffled at dataset level
                num_workers=2, pin_memory=True
            )
            dataloaders.append(dataloader)

        # Training phase
        whisper_model.train()
        total_loss = 0.0
        total_clean_asr_loss = 0.0
        total_noisy_asr_loss = 0.0
        total_contrastive_loss = 0.0

        min_len = min(len(dl) for dl in dataloaders)
        data_iterator = iter(zip(*dataloaders))
        pbar = tqdm(range(min_len), desc=f"Epoch {epoch+1} Training")

        for step in pbar:
            try:
                batches = next(data_iterator)
                clean_batch = batches[0]
                noisy_batches = batches[1:]

                # Prepare clean data
                clean_inputs = clean_batch['input_features'].to(device)
                clean_labels = clean_batch['labels'].to(device)
                clean_labels = clean_labels.masked_fill(
                    clean_labels == processor.tokenizer.pad_token_id, -100
                )

                with autocast():
                    # Forward pass on clean data
                    clean_outputs = whisper_model(
                        input_features=clean_inputs, labels=clean_labels
                    )
                    clean_asr_loss = clean_outputs.loss
                    clean_features = clean_outputs.encoder_last_hidden_state.mean(dim=1)

                    # Process noisy samples
                    noisy_features_list = []
                    noisy_asr_loss = 0.0
                    num_noisy_samples = 0

                    for noisy_batch in noisy_batches:
                        noisy_inputs = noisy_batch['input_features'].to(device)
                        noisy_labels = noisy_batch['labels'].to(device)
                        noisy_labels = noisy_labels.masked_fill(
                            noisy_labels == processor.tokenizer.pad_token_id, -100
                        )

                        noisy_outputs = whisper_model(
                            input_features=noisy_inputs, labels=noisy_labels
                        )
                        noisy_asr_loss += noisy_outputs.loss
                        noisy_features = noisy_outputs.encoder_last_hidden_state.mean(dim=1)
                        noisy_features_list.append(noisy_features)
                        num_noisy_samples += 1

                    # Average noisy ASR loss
                    if num_noisy_samples > 0:
                        noisy_asr_loss = noisy_asr_loss / num_noisy_samples

                    # Compute contrastive loss
                    if use_multi_positive and len(noisy_features_list) > 1:
                        contrastive_loss = calculate_multi_positive_nt_xent_loss(
                            clean_features, noisy_features_list, temperature, device
                        )
                    elif len(noisy_features_list) > 0:
                        contrastive_loss = 0.0
                        for noisy_features in noisy_features_list:
                            contrastive_loss += calculate_nt_xent_loss(
                                clean_features, noisy_features, temperature, device
                            )
                        contrastive_loss = contrastive_loss / len(noisy_features_list)
                    else:
                        contrastive_loss = 0.0

                    # Combined loss
                    combined_loss = (
                        clean_asr_loss + 
                        current_noisy_asr_weight * noisy_asr_loss + 
                        current_contrastive_weight * contrastive_loss
                    )

                # Backward pass
                scaled_loss = combined_loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == min_len - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(whisper_model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()

                # Logging
                total_loss += combined_loss.item()
                total_clean_asr_loss += clean_asr_loss.item()
                total_noisy_asr_loss += noisy_asr_loss.item() if isinstance(noisy_asr_loss, torch.Tensor) else noisy_asr_loss
                total_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss

                pbar.set_postfix({
                    "Clean": f"{clean_asr_loss.item():.4f}",
                    "Noisy": f"{noisy_asr_loss.item() if isinstance(noisy_asr_loss, torch.Tensor) else noisy_asr_loss:.4f}",
                    "Contrast": f"{contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss:.4f}",
                    "Total": f"{combined_loss.item():.4f}"
                })

                if step % 10 == 0:
                    wandb.log({
                        "train/clean_asr_loss": clean_asr_loss.item(),
                        "train/noisy_asr_loss": noisy_asr_loss.item() if isinstance(noisy_asr_loss, torch.Tensor) else noisy_asr_loss,
                        "train/contrastive_loss": contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss,
                        "train/combined_loss": combined_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "step": step + epoch * min_len,
                        "epoch": epoch
                    })

                torch.cuda.empty_cache()

            except StopIteration:
                break

        # Validation phase
        whisper_model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for val_batch in tqdm(eval_dataloader, desc="Validation"):
                val_inputs = val_batch['input_features'].to(device)
                val_labels = val_batch['labels'].to(device)
                val_labels = val_labels.masked_fill(
                    val_labels == processor.tokenizer.pad_token_id, -100
                )

                with autocast():
                    val_outputs = whisper_model(
                        input_features=val_inputs, labels=val_labels
                    )
                    val_loss += val_outputs.loss.item()
                    val_steps += 1

        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')

        # Early stopping
        improvement = best_val_asr_loss - avg_val_loss
        if improvement > min_delta:
            best_val_asr_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in whisper_model.state_dict().items()}
            print(f"✓ New best validation loss: {best_val_asr_loss:.4f}")
        else:
            patience_counter += 1
            print(f"✗ No improvement. Patience: {patience_counter}/{patience}")

        # Epoch summary
        avg_steps = max(min_len, 1)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Clean ASR Loss: {total_clean_asr_loss/avg_steps:.4f}")
        print(f"  Noisy ASR Loss: {total_noisy_asr_loss/avg_steps:.4f}")
        print(f"  Contrastive Loss: {total_contrastive_loss/avg_steps:.4f}")
        print(f"  Combined Loss: {total_loss/avg_steps:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        wandb.log({
            "val/loss": avg_val_loss,
            "val/best_loss": best_val_asr_loss,
            "epoch": epoch
        })

        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch+1} epochs!")
            break

    # Load best model
    if best_model_state is not None:
        best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
        whisper_model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_asr_loss:.4f}")

    return whisper_model


# Alternative approach: Custom Dataset with built-in shuffling
class AlignedContrastiveDataset(Dataset):
    """
    Custom dataset that maintains alignment while allowing flexible shuffling strategies.
    """
    def __init__(self, datasets, shuffle_strategy="synchronized"):
        self.datasets = datasets
        self.shuffle_strategy = shuffle_strategy
        self.length = len(datasets[0])
        
        # Verify all datasets have same length
        assert all(len(d) == self.length for d in datasets), "All datasets must have same length"
        
        # Initialize indices
        self.reset_epoch()
    
    def reset_epoch(self):
        """Call this at the start of each epoch to reshuffle indices."""
        if self.shuffle_strategy == "synchronized":
            # Same shuffle for all datasets
            self.indices = list(range(self.length))
            random.shuffle(self.indices)
        elif self.shuffle_strategy == "independent":
            # Different shuffle for each dataset
            self.indices = [list(range(self.length)) for _ in self.datasets]
            for idx_list in self.indices:
                random.shuffle(idx_list)
        else:  # "none"
            self.indices = list(range(self.length))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.shuffle_strategy == "synchronized" or self.shuffle_strategy == "none":
            actual_idx = self.indices[idx]
            return [dataset[actual_idx] for dataset in self.datasets]
        else:  # independent
            return [dataset[indices[idx]] for dataset, indices in zip(self.datasets, self.indices)]


# Usage example with the custom dataset approach:
def create_aligned_dataloader_with_shuffling(datasets, batch_size, collator, shuffle_strategy="synchronized"):
    """
    Create a dataloader that handles aligned datasets with proper shuffling.
    """
    aligned_dataset = AlignedContrastiveDataset(datasets, shuffle_strategy)
    
    def collate_fn(batch):
        # batch is a list of [sample_from_dataset_0, sample_from_dataset_1, ...]
        # We need to collate each dataset separately
        num_datasets = len(batch[0])
        collated_batches = []
        
        for dataset_idx in range(num_datasets):
            dataset_samples = [item[dataset_idx] for item in batch]
            collated_batch = collator(dataset_samples)
            collated_batches.append(collated_batch)
        
        return collated_batches
    
    dataloader = DataLoader(
        aligned_dataset, 
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # Shuffling handled by dataset
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader
def adaptive_temperature_schedule(epoch, max_epochs, initial_temp=0.1, final_temp=0.05):
    """Adaptive temperature scheduling"""
    progress = epoch / max_epochs
    return initial_temp * (final_temp / initial_temp) ** progress
def compute_feature_similarity(clean_features, noisy_features):
    """Compute cosine similarity between clean and noisy features"""
    clean_norm = F.normalize(clean_features, dim=-1)
    noisy_norm = F.normalize(noisy_features, dim=-1)
    similarity = torch.sum(clean_norm * noisy_norm, dim=-1)
    return similarity.mean().item()
