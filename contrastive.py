# --- End Debugging ---
import torch
from torch.amp import GradScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm.auto import tqdm # For progress bars
import math # For inf check

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

# --- 4. Training Loop ---
def train_infonce(whisper_model, processor, dataloader_A, dataloader_B, device,
        num_epochs=5, lr=5e-5, weight_decay=0.01,
        infonce_weight=0.1, temperature=0.1):

    # --- Optimizer ---
    optimizer = optim.AdamW(whisper_model.parameters(), lr=lr, weight_decay=weight_decay)
    #torch.autograd.set_detect_anomaly(True)
    len_dataloader = min(len(dataloader_A), len(dataloader_B))
    if len_dataloader == 0:
        print("Error: Dataloaders are empty.")
        return

    print(f"Training for {num_epochs} epochs with {len_dataloader} steps per epoch.")
    print(f"InfoNCE Loss weight: {infonce_weight}, Temperature: {temperature}")
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        whisper_model.train()
        whisper_model.to("cuda")

        total_standard_loss = 0.0
        total_contrastive_loss = 0.0

        # Use zip to iterate through batches from both domains simultaneously
        # Requires dataloaders yield batches of the same size for simple pairing
        data_iterator = iter(zip(dataloader_A, dataloader_B))

        pbar = tqdm(range(len_dataloader), desc="Training Steps")
        for step in pbar:
            try:
                batch_A, batch_B = next(data_iterator)
            except StopIteration:
                break

            # --- Prepare Data ---
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
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
                outputs_A = whisper_model(input_features=inputs_A, labels=labels_A)
                standard_loss_A = outputs_A.loss # This is the seq2seq CE loss

                # Process target domain B - primarily to get encoder features
                # No labels provided = no standard loss calculated for B here
                # Provide labels=labels_B if you want to calculate standard loss on B too
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
                pooled_features_B = features_encoder_B.mean(dim=1) # (batch, hidden_dim)

                # --- InfoNCE Loss ---
                contrastive_loss = calculate_infonce_loss(
                        pooled_features_A, pooled_features_B,
                        temperature=temperature, device=device
                        )

                # --- Combine Losses ---
                # Adjust weighting as needed
                # Here, we only use standard loss from domain A
                total_loss = standard_loss_A + infonce_weight * contrastive_loss
                # Alternative: If using standard loss on both:
                # standard_loss_B = whisper_model(input_features=inputs_B, labels=labels_B).loss
                # total_loss = (standard_loss_A + standard_loss_B)/2 + infonce_weight * contrastive_loss

                # --- Backpropagation & Optimization ---
            scaler.scale(total_loss).backward()
            #total_loss.backward()
            max_grad = 0.0
            has_nan_inf_grad = False
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
            #torch.nn.utils.clip_grad_norm_(whisper_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()

            # --- Logging ---
            total_standard_loss += standard_loss_A.item()
            total_contrastive_loss += contrastive_loss.item()
            #del outputs_B, outputs_A, inputs_A, inputs_B, labels_A,labels_B, features_encoder_A, features_encoder_B



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
        torch.cuda.memory._dump_snapshot("contrastive{step}.pickle")


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
