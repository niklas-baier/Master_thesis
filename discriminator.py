import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm.auto import tqdm # For progress bars

# --- 1. Gradient Reversal Layer (GRL) ---
# Implementation based on https://github.com/fungtion/DANN/blob/master/models/functions.py
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store lambda_ for backward pass
        ctx.lambda_ = lambda_
        # Identity function during forward pass
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient and scale it by lambda_
        # The negative sign reverses the gradient
        output = grad_output.neg() * ctx.lambda_
        # Need to return gradients for all inputs of forward: x, lambda_
        # We don't need gradient w.r.t. lambda_, so return None for it
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
    def __init__(self, input_dim, hidden_dim=512):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Output a single logit for Binary Cross Entropy with Logits
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.layer(x)
class WhisperDomainAdapter(nn.Module):
    def __init__(self, whisper_model, discriminator, grl, device="cuda"):
        super(WhisperDomainAdapter,self).__init__()
        self.whisper_model = whisper_model
        self.discriminator = discriminator
        self.grl = grl
        self.device = device

    def forward(self, input_features, microphone_domain, labels=None, task_compute_loss=True):
        encoder_outputs = self.whisper_model.model.encoder(input_features)
        hidden_state = encoder_outputs.last_hidden_state
        reversed_features = self.grl(hidden_state)
        domain_output = self.discriminator(reversed_features)
        domain_mean = domain_output.mean(dim=1)
        breakpoint()
        domain_loss_fn = nn.CrossEntropyLoss()
        domain_mean_loss = domain_loss_fn(domain_mean, microphone_domain)
        if task_compute_loss and labels is not None:
            from transformers.models.whisper.modeling_whisper import shift_tokens_right
            decoder_input_ids = shift_tokens_right(labels, self.whisper_model.config.pad_token_id, self.whisper_model.config.decoder_start_token_id)
            decoder_output = self.whisper_model.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=hidden_state)
            lm_logits = self.whisper_model.proj_out(decoder_output.last_hidden_state)
            shifted_logits = lm_logits[:,:-1,:].contiguous()
            shifted_labels = labels[:,1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            task_loss =  loss_fct(shifted_logits.view(-1, lm_logits.size(-1)), shifted_labels.view(-1))
            return {"task_loss": task_loss, "domain_loss": domain_mean_loss, "encoder_hidden_states": hidden_state}
        return {"domain_loss": domain_mean_loss, "encoder_hidden_states": hidden_state}
    def generate(self, input_features, **kwargs):
        encoder_outputs = self.whisper_model.model.encoder(input_features)
        hidden_states = encoder_outputs.last_hidden_state
        return self.whisper_model.generate(encoder_outputs=encoder_outputs, **kwargs)





# --- 3. Model Setup ---
def setup_models(whisper_model_name="distil-whisper/distil-large-v3", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Using device: {device}")

    # Load Whisper model and processor
    # Using WhisperForConditionalGeneration, but we'll focus on the encoder
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    # We don't need the processor for the core logic here, but you'd need it for data prep
    # processor = WhisperProcessor.from_pretrained(whisper_model_name)

    # Determine the dimensionality of Whisper encoder's output
    encoder_output_dim = whisper_model.config.d_model

    # Create the domain discriminator
    discriminator = DomainDiscriminator(input_dim=encoder_output_dim).to(device)

    # Create the Gradient Reversal Layer
    grl = GradientReversalLayer(lambda_=1.0) # Lambda can be adjusted/scheduled

    # (Optional) Task Head: Example - Simple classifier on pooled features
    # Replace with your actual task head (e.g., transcription decoder is already in whisper_model)
    task_head = nn.Linear(encoder_output_dim, 2).to(device) # Example: 10 classes

    return whisper_model, discriminator, grl, task_head, device

# --- 4. Placeholder DataLoaders ---
# Replace this with your actual data loading logic
def get_dataloaders(batch_size=16):
    # Dummy data - replace with real processed Whisper inputs
    # Input features should be pre-processed log-mel spectrograms
    dummy_input_features = torch.randn(batch_size, 128, 3000) # (batch, n_mels, seq_len)
    dummy_source_labels = torch.randint(0, 10, (batch_size,)) # Example task labels

    # Domain A (Source) - Assume has task labels
    dataloader_A = [(dummy_input_features, dummy_source_labels)] * 10 # Dummy loop
    # Domain B (Target) - Assume does not need task labels for adaptation part
    dataloader_B = [(dummy_input_features,)] * 10 # Dummy loop

    print(f"Returning dummy dataloaders with batch size {batch_size}")
    print(f"Dummy input feature shape: {dummy_input_features.shape}")
    return dataloader_A, dataloader_B

# --- 5. Training Loop ---
def train_adversarial(whisper_model, discriminator, grl, task_head,
                      dataloader_A, dataloader_B, device,
                      num_epochs=5, lr=1e-5, weight_decay=0.01,
                      lambda_domain_loss=0.1): # Weight for domain loss

    # --- Optimizers ---
    # Parameters for the main model (feature extractor + optional task head)
    whisper_adapter = WhisperDomainAdapter(whisper_model=whisper_model, discriminator=discriminator, device=device, grl = grl)
    params_main = list(whisper_adapter.whisper_model.parameters())
    optimizer_main = optim.AdamW(params_main, lr=lr, weight_decay=weight_decay)

    # Parameters for the discriminator
    optimizer_disc = optim.AdamW(whisper_adapter.discriminator.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Loss Functions ---
    # Task Loss (Example: CrossEntropy for classification on source domain)
    criterion_task = nn.CrossEntropyLoss()
    # Domain Loss (Binary Cross Entropy for discriminating between domains A and B)
    criterion_domain = nn.CrossEntropyLoss() # Handles logits directly
    len_dataloader = min(len(dataloader_A), len(dataloader_B))
    print(f"Training for {num_epochs} epochs with {len_dataloader} steps per epoch.")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        whisper_model.train()
        discriminator.train()
        task_head.train()
        whisper_adapter.train()

        total_task_loss = 0.0
        total_domain_loss = 0.0
        correct_domain_predictions = 0
        total_domain_samples = 0

        # Use zip_longest if dataloaders have different lengths, or just min length
        data_iterator = iter(zip(dataloader_A, dataloader_B))

        pbar = tqdm(range(len_dataloader), desc="Training Steps")
        for step in pbar:
            try:
                batch_A, batch_B = next(data_iterator)
            except StopIteration:
                # Should not happen if using min length
                break

            # --- Prepare Data ---
            # Source domain data (Domain 0)
            breakpoint()
            inputs_A = batch_A['input_features'].to(device) # Input features
            labels_task_A = batch_A['labels'].to(device) # Task labels for source
            domain_labels_A = torch.zeros(inputs_A.size(0), 1, device=device, dtype=torch.float) # Domain 0

            # Target domain data (Domain 1)
            inputs_B = batch_B['input_features'].to(device) # Input features (batch size might differ)
             # Task labels for target might not exist or be used in unsupervised DA
            domain_labels_B = torch.ones(inputs_B.size(0), 1, device=device, dtype=torch.float) # Domain 1

            # --- Forward Pass: Feature Extraction ---
            # Process source data through Whisper encoder
            # We only need encoder output, `output_hidden_states=True` might be needed
            # depending on how you access intermediate layers if not the last one.
            # Using `encoder_last_hidden_state`. No decoder_input_ids needed for just encoder.
            results_A = whisper_adapter(input_features=inputs_A, labels=labels_task_A, microphone_domain=domain_labels_A)
            #results_B = whisper_adapter(input_features=inputs_B, labels=labels_task_A,domain_labels=domain_labels_B)
            total_loss = results_A['task_loss'] + lambda_domain_loss * results_A['domain_loss']
            total_loss.backward()
            optimizer_main.step()
            optimizer_disc.zero_grad()
            #domain_preds = whisper_model./

            # --- Backpropagation ---
            # Update main model (Whisper + Task Head)
            optimizer_main.zero_grad()
            # Update discriminator separately
            optimizer_disc.zero_grad()

            total_loss.backward() # Gradients calculated for all parts

            # Step main optimizer (Whisper + Task Head)
            # GRL has reversed the gradient from domain_loss for Whisper params
            optimizer_main.step()

            # Step discriminator optimizer
            # The gradient flowing *into* the discriminator from domain_loss is normal (not reversed)
            # We detach features_reversed to ensure only discriminator is updated based on domain loss
            # This is one way to handle it; sometimes people do separate forward passes
            # Let's recalculate domain loss without GRL for discriminator update
            domain_logits_disc = discriminator(features_combined.detach()) # Use detached features
            domain_loss_disc = criterion_domain(domain_logits_disc, domain_labels_combined)
            # We need gradients for discriminator only now
            optimizer_disc.zero_grad()
            domain_loss_disc.backward()
            optimizer_disc.step()


            with torch.no_grad():
                outputs_A = whisper_model.model.encoder(inputs_A) # Get raw encoder outputs
            features_A = outputs_A.last_hidden_state # (batch, seq_len, hidden_dim)

            # Process target data through Whisper encoder
            with torch.no_grad():
                 outputs_B = whisper_model.model.encoder(inputs_B)
            features_B = outputs_B.last_hidden_state # (batch, seq_len, hidden_dim)

            # --- Pooling ---
            # Pool the features to get a fixed-size vector per sample
            # Mean pooling is common
            pooled_features_A = features_A.mean(dim=1) # (batch, hidden_dim)
            pooled_features_B = features_B.mean(dim=1) # (batch, hidden_dim)

            # --- Task Prediction (Optional, only on source) ---
            task_output = task_head(pooled_features_A)
            task_loss = criterion_task(task_output, labels_task_A)

            # --- Domain Classification ---
            # Combine features from both domains
            features_combined = torch.cat((pooled_features_A, pooled_features_B), dim=0)
            domain_labels_combined = torch.cat((domain_labels_A, domain_labels_B), dim=0)

            # Apply GRL before discriminator
            # Note: GRL only affects gradients flowing back *towards whisper_model*
            features_reversed = grl(features_combined)

            # Get domain predictions from discriminator
            domain_logits = discriminator(features_reversed)
            domain_loss = criterion_domain(domain_logits, domain_labels_combined)

            # --- Calculate Total Loss for Main Model ---
            # Whisper is trained to MAXIMIZE domain loss (via GRL)
            # and MINIMIZE task loss.
            # lambda_domain_loss weights the importance of fooling the discriminator
            total_loss = task_loss + lambda_domain_loss * domain_loss

            # --- Backpropagation ---
            # Update main model (Whisper + Task Head)
            optimizer_main.zero_grad()
            # Update discriminator separately
            optimizer_disc.zero_grad()

            total_loss.backward() # Gradients calculated for all parts

            # Step main optimizer (Whisper + Task Head)
            # GRL has reversed the gradient from domain_loss for Whisper params
            optimizer_main.step()

            # Step discriminator optimizer
            # The gradient flowing *into* the discriminator from domain_loss is normal (not reversed)
            # We detach features_reversed to ensure only discriminator is updated based on domain loss
            # This is one way to handle it; sometimes people do separate forward passes
            # Let's recalculate domain loss without GRL for discriminator update
            domain_logits_disc = discriminator(features_combined.detach()) # Use detached features
            domain_loss_disc = criterion_domain(domain_logits_disc, domain_labels_combined)
            # We need gradients for discriminator only now
            optimizer_disc.zero_grad()
            domain_loss_disc.backward()
            optimizer_disc.step()


            # --- Logging & Metrics ---
            total_task_loss += task_loss.item()
            total_domain_loss += domain_loss_disc.item() # Log the discriminator's actual loss

            # Calculate domain accuracy (optional)
            domain_preds = (torch.sigmoid(domain_logits_disc) > 0.5).float()
            correct_domain_predictions += (domain_preds == domain_labels_combined).sum().item()
            total_domain_samples += domain_labels_combined.size(0)

            pbar.set_postfix({
                "Task Loss": f"{task_loss.item():.4f}",
                "Domain Loss": f"{domain_loss_disc.item():.4f}"
            })

        # --- End of Epoch ---
        avg_task_loss = total_task_loss / len_dataloader
        avg_domain_loss = total_domain_loss / len_dataloader
        domain_accuracy = (correct_domain_predictions / total_domain_samples) * 100 if total_domain_samples > 0 else 0.0

        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Task Loss: {avg_task_loss:.4f}")
        print(f"  Avg Domain Loss (Discriminator): {avg_domain_loss:.4f}")
        print(f"  Domain Accuracy: {domain_accuracy:.2f}%")

        # --- Adjust GRL lambda (Optional Schedule) ---
        # Example: Gradually increase lambda (from Ganin et al.)
        # p = float(epoch) / num_epochs
        # lambda_ = 2. / (1. + np.exp(-10. * p)) - 1
        # grl.set_lambda(lambda_)
    torch.save(whisper_model.state_dict(), "whisper_model_die.pth")    # print(f"  Set GRL lambda to: {grl.lambda_:.4f}")


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    WHISPER_MODEL = "distil-whisper/distil-large-v3" # Choose your Whisper model
    BATCH_SIZE = 8 # Adjust based on GPU memory
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    LAMBDA_DOMAIN = 0.1 # How much to weight the adversarial loss

    # --- Setup ---
    whisper_model, discriminator, grl, task_head, device = setup_models(WHISPER_MODEL)
    dataloader_A, dataloader_B = get_dataloaders(BATCH_SIZE)

    # --- Train ---
    train_adversarial(whisper_model, discriminator, grl, task_head,
                      dataloader_A, dataloader_B, device,
                      num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY,
                      lambda_domain_loss=LAMBDA_DOMAIN)

    print("\nTraining finished.")
    # Add code here to save your adapted whisper_model weights if needed
    # torch.save(whisper_model.state_dict(), "adapted_whisper_model.pth")
