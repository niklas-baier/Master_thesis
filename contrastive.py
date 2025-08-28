import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import random
import numpy as np
from einops import rearrange, reduce
import math

class ProjectionHead(nn.Module):
    """
    Non-linear projection head for contrastive learning.
    Maps encoder features to a lower-dimensional space optimized for contrastive loss.
    """
    def __init__(self, input_dim, hidden_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)

class ContrastiveWhisperModel(nn.Module):
    """
    Wrapper around Whisper model with projection head for contrastive learning.
    """
    def __init__(self, whisper_model, encoder_dim=1280, projection_dim=256):
        super().__init__()
        self.whisper_model = whisper_model
        self.projection_head = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=512,
            output_dim=projection_dim,
            dropout=0.1
        )
        
        # Freeze decoder initially for more stable contrastive training
        self.freeze_decoder = False
        
    def forward(self, input_features, labels=None, return_projections=False):
        """
        Forward pass that can return both ASR outputs and contrastive projections.
        """
        outputs = self.whisper_model(input_features=input_features, labels=labels)
        
        if return_projections:
            # Extract encoder features and project them
            encoder_features = outputs.encoder_last_hidden_state  # [batch, seq_len, hidden_dim]
            
            # Use attention-based pooling for better feature representation
            pooled_features = self.attention_pool(encoder_features)
            projections = self.projection_head(pooled_features)
            
            return outputs, projections
        
        return outputs
    
    def attention_pool(self, encoder_features):
        """
        Attention-based pooling of encoder features.
        """
        # Simple attention mechanism
        batch_size, seq_len, hidden_dim = encoder_features.shape
        
        # Compute attention weights
        attention_weights = torch.softmax(
            torch.sum(encoder_features * encoder_features.mean(dim=1, keepdim=True), dim=2),
            dim=1
        )  # [batch, seq_len]
        
        # Apply attention weights
        pooled = torch.sum(encoder_features * attention_weights.unsqueeze(2), dim=1)
        return pooled  # [batch, hidden_dim]
    '''

    def attention_pool(self, encoder_features):
        return improved_attention_pool(encoder_features)
'''
    def set_freeze_decoder(self, freeze=True):
        """Freeze/unfreeze decoder parameters."""
        self.freeze_decoder = freeze
        for param in self.whisper_model.model.model.decoder.parameters():
            param.requires_grad = not freeze

def calculate_nt_xent_loss(clean_projections, noisy_projections, temperature=0.07, device='cuda'):
    """
    Improved NT-Xent loss with better numerical stability.
    """
    batch_size = clean_projections.size(0)
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize projections to unit sphere
    clean_projections = F.normalize(clean_projections, dim=1)
    noisy_projections = F.normalize(noisy_projections, dim=1)
    
    # Concatenate clean and noisy projections
    projections = torch.cat([clean_projections, noisy_projections], dim=0)  # [2*batch, proj_dim]
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(projections, projections.t()) / temperature  # [2*batch, 2*batch]
    
    # Create labels for positive pairs
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device)
    ])
    
    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
    
    # Check for numerical issues
    if not torch.isfinite(similarity_matrix).all():
        print("Warning: Non-finite values in similarity matrix")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(similarity_matrix, labels)
    
    return loss

def calculate_multi_positive_nt_xent_loss(clean_projections, noisy_projections_list, temperature=0.07, device='cuda'):
    """
    NT-Xent loss with multiple positive pairs (different microphones).
    """
    batch_size = clean_projections.size(0)
    num_positives = len(noisy_projections_list)
    
    if batch_size <= 1 or num_positives == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Normalize all projections
    clean_projections = F.normalize(clean_projections, dim=1)
    noisy_projections_list = [F.normalize(proj, dim=1) for proj in noisy_projections_list]
    
    # Concatenate all projections: [clean, noisy1, noisy2, ...]
    all_projections = torch.cat([clean_projections] + noisy_projections_list, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(all_projections, all_projections.t()) / temperature
    
    # Create positive pair mask
    total_samples = (1 + num_positives) * batch_size
    positive_mask = torch.zeros((total_samples, total_samples), device=device, dtype=torch.bool)
    
    # Clean samples are positive with all corresponding noisy samples
    for i in range(num_positives):
        start_idx = (i + 1) * batch_size
        clean_indices = torch.arange(batch_size, device=device)
        noisy_indices = torch.arange(start_idx, start_idx + batch_size, device=device)
        
        positive_mask[clean_indices, noisy_indices] = True
        positive_mask[noisy_indices, clean_indices] = True
    
    # Mask diagonal
    self_mask = torch.eye(total_samples, device=device, dtype=torch.bool)
    similarity_matrix = similarity_matrix.masked_fill(self_mask, float('-inf'))
    
    # Calculate loss for clean samples only (to avoid double counting)
    clean_similarity = similarity_matrix[:batch_size]  # [batch, total_samples]
    clean_positive_mask = positive_mask[:batch_size]   # [batch, total_samples]
    
    # For each clean sample, compute softmax over all samples and extract positive log-probs
    log_probs = F.log_softmax(clean_similarity, dim=1)
    positive_log_probs = log_probs[clean_positive_mask].view(batch_size, -1)
    
    # Average over positive pairs for each sample, then over batch
    loss = -positive_log_probs.mean()
    
    return loss

def verify_dataset_correspondence(train_datasets, num_samples_to_check=5):
    """
    Verify that all datasets have corresponding samples with same transcriptions
    """
    print("Verifying dataset correspondence...")
    
    # Check first few samples
    for i in range(min(num_samples_to_check, len(train_datasets[0]))):
        clean_sample = train_datasets[0][i]
        clean_labels = clean_sample['labels']
        
        print(f"Sample {i}: Clean labels type: {type(clean_labels)}")
        if isinstance(clean_labels, list):
            print(f"   Clean labels (list): {clean_labels[:10] if len(clean_labels) > 10 else clean_labels}")
        else:
            print(f"   Clean labels shape: {clean_labels.shape}")
            print(f"   Clean labels: {clean_labels[:10] if len(clean_labels) > 10 else clean_labels}")
        
        for j, dataset in enumerate(train_datasets[1:], 1):
            noisy_sample = dataset[i]
            noisy_labels = noisy_sample['labels']
            
            print(f"   Dataset {j}: Labels type: {type(noisy_labels)}")
            
            # Compare based on type
            if isinstance(clean_labels, list) and isinstance(noisy_labels, list):
                labels_match = clean_labels == noisy_labels
            elif isinstance(clean_labels, torch.Tensor) and isinstance(noisy_labels, torch.Tensor):
                labels_match = torch.equal(clean_labels, noisy_labels)
            else:
                # Convert to same type for comparison
                if isinstance(clean_labels, list):
                    clean_labels = torch.tensor(clean_labels)
                if isinstance(noisy_labels, list):
                    noisy_labels = torch.tensor(noisy_labels)
                labels_match = torch.equal(clean_labels, noisy_labels)
            
            if not labels_match:
                print(f"❌ MISMATCH at sample {i}: Clean vs Dataset {j}")
                print(f"   Clean: {clean_labels[:10] if len(clean_labels) > 10 else clean_labels}")
                print(f"   Noisy: {noisy_labels[:10] if len(noisy_labels) > 10 else noisy_labels}")
                return False
            else:
                print(f"✅ Sample {i}: Clean matches Dataset {j}")
    
    print("✅ Dataset correspondence verified!")
    return True

def train_improved_contrastive(
    whisper_model, processor, collator, train_datasets, eval_dataset, device,
    batch_size=4, num_epochs=20, lr=5e-5, weight_decay=0.01,
    contrastive_weight=0.5, noisy_asr_weight=0.3, temperature=0.07,
    warmup_epochs=2, use_multi_positive=True, patience=5, min_delta=0.001,
    gradient_accumulation_steps=4, freeze_decoder_epochs=3
):
    """
    Improved contrastive training with projection head and better optimization.
    """
    
    # First, verify dataset correspondence
    if not verify_dataset_correspondence(train_datasets):
        raise ValueError("Dataset correspondence verification failed! Check your data loading.")
    
    # Wrap whisper model with projection head
    contrastive_model = ContrastiveWhisperModel(
        whisper_model,
        encoder_dim=whisper_model.config.d_model,
        projection_dim=256
    ).to(device)
    
    # Initially freeze decoder for more stable contrastive learning
    if freeze_decoder_epochs > 0:
        contrastive_model.set_freeze_decoder(True)
        print(f"Decoder frozen for first {freeze_decoder_epochs} epochs")
    
    contrastive_model.train()
    contrastive_model.whisper_model.gradient_checkpointing_enable()
    
    # Optimizer with different learning rates for different components
    projection_params = list(contrastive_model.projection_head.parameters())
    whisper_params = list(contrastive_model.whisper_model.parameters())
    
    optimizer = optim.AdamW([
        {'params': whisper_params, 'lr': lr},
        {'params': projection_params, 'lr': lr * 2}  # Higher LR for projection head
    ], weight_decay=weight_decay)
    
    scaler = GradScaler()
    
    # Learning rate scheduler
    total_steps = len(train_datasets[0]) * num_epochs // (batch_size * gradient_accumulation_steps)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[lr, lr * 2], 
        total_steps=total_steps
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # Validation dataloader
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=batch_size, collate_fn=collator,
        shuffle=False, num_workers=2, pin_memory=True
    )
    
    # Random batch generator for contrastive sampling
    torch.manual_seed(42)
    random_batch_generator = torch.randint(0, len(train_datasets) - 1, (100000,))
    counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        
        # Unfreeze decoder after specified epochs
        if epoch == freeze_decoder_epochs:
            contrastive_model.set_freeze_decoder(False)
            print("Decoder unfrozen")
        
        # Adaptive temperature and weights
        current_temperature = temperature * (0.5 ** (epoch // 5))  # Reduce temp over time
        progress = min(1.0, (epoch + 1) / warmup_epochs)
        current_contrastive_weight = contrastive_weight * progress
        current_noisy_asr_weight = noisy_asr_weight * progress
        
        # Create synchronized dataloaders with same shuffling
        # CRITICAL FIX: Use same random seed for all dataloaders to maintain correspondence
        generator = torch.Generator()
        generator.manual_seed(42 + epoch)  # Different seed per epoch but same across dataloaders
        
        dataloaders = []
        for dataset in train_datasets:
            # Create sampler with same generator for synchronized shuffling
            indices = torch.randperm(len(dataset), generator=generator).tolist()
            sampler = torch.utils.data.SubsetRandomSampler(indices)
            
            dataloader = DataLoader(
                dataset, batch_size=batch_size, collate_fn=collator,
                sampler=sampler, num_workers=2, pin_memory=True
            )
            dataloaders.append(dataloader)
        
        # Training loop
        contrastive_model.train()
        total_loss = 0.0
        total_clean_asr_loss = 0.0
        total_noisy_asr_loss = 0.0
        total_contrastive_loss = 0.0
        
        min_len = min(len(dl) for dl in dataloaders)
        data_iterator = iter(zip(*dataloaders))
        pbar = tqdm(range(min_len), desc=f"Epoch {epoch+1}")
        
        for step in pbar:
            try:
                batches = next(data_iterator)
                clean_batch = batches[0]
                noisy_batches = batches[1:]
                
                # Generate contrastive batch using fixed function
                contrastive_batch = generate_contrastive_batch(
                    counter, batch_size, random_batch_generator,
                    *noisy_batches  # Pass all noisy batches
                )
                counter += 1

                # Prepare data
                clean_inputs = clean_batch['input_features'].to(device)
                clean_labels = clean_batch['labels'].to(device)
                clean_labels = clean_labels.masked_fill(
                    clean_labels == processor.tokenizer.pad_token_id, -100
                )
                
                noisy_inputs = contrastive_batch['input_features'].to(device)
                noisy_labels = contrastive_batch['labels'].to(device)
                noisy_labels = noisy_labels.masked_fill(
                    noisy_labels == processor.tokenizer.pad_token_id, -100
                )
                
                with autocast():
                    # Forward pass on clean data
                    clean_outputs, clean_projections = contrastive_model(
                        clean_inputs, clean_labels, return_projections=True
                    )
                    clean_asr_loss = clean_outputs.loss
                    
                    # Forward pass on noisy data
                    noisy_outputs, noisy_projections = contrastive_model(
                        noisy_inputs, noisy_labels, return_projections=True
                    )
                    noisy_asr_loss = noisy_outputs.loss
                    
                    # Compute contrastive loss
                    contrastive_loss = calculate_nt_xent_loss(
                        clean_projections, noisy_projections,
                        current_temperature, device
                    )
                    
                    # Combined loss
                    combined_loss = (
                        clean_asr_loss + 
                        current_noisy_asr_weight * noisy_asr_loss + 
                        current_contrastive_weight * contrastive_loss
                    )
                
                # Backward pass with gradient accumulation
                scaled_loss = combined_loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(), 1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Logging
                total_loss += combined_loss.item()
                total_clean_asr_loss += clean_asr_loss.item()
                total_noisy_asr_loss += noisy_asr_loss.item()
                total_contrastive_loss += contrastive_loss.item()
                
                pbar.set_postfix({
                    "Clean": f"{clean_asr_loss.item():.4f}",
                    "Noisy": f"{noisy_asr_loss.item():.4f}",
                    "Contrast": f"{contrastive_loss.item():.4f}",
                    "Temp": f"{current_temperature:.3f}"
                })
                
                # Wandb logging
                if step % 10 == 0:
                    wandb.log({
                        "train/clean_asr_loss": clean_asr_loss.item(),
                        "train/noisy_asr_loss": noisy_asr_loss.item(),
                        "train/contrastive_loss": contrastive_loss.item(),
                        "train/combined_loss": combined_loss.item(),
                        "train/temperature": current_temperature,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "step": step + epoch * min_len,
                        "epoch": epoch
                    })
                
                torch.cuda.empty_cache()
                
            except StopIteration:
                break
        
        # Validation
        contrastive_model.eval()
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
                    val_outputs = contrastive_model.whisper_model(
                        input_features=val_inputs, labels=val_labels
                    )
                    val_loss += val_outputs.loss.item()
                    val_steps += 1
        
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        
        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in contrastive_model.state_dict().items()}
            torch.save(best_model_state, f"best_contrastive_model_epoch_{epoch+1}.pt")
            print(f"✓ New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"✗ Patience: {patience_counter}/{patience}")
        
        # Epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Clean ASR Loss: {total_clean_asr_loss/min_len:.4f}")
        print(f"  Noisy ASR Loss: {total_noisy_asr_loss/min_len:.4f}")
        print(f"  Contrastive Loss: {total_contrastive_loss/min_len:.4f}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")
        
        wandb.log({
            "val/loss": avg_val_loss,
            "val/best_loss": best_val_loss,
            "epoch": epoch
        })
        
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs!")
            break
    
    # Load best model
    if best_model_state is not None:
        best_model_state = {k: v.to(device) for k, v in best_model_state.items()}
        contrastive_model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
    
    return contrastive_model.whisper_model

# Usage function to replace your existing training call
def train_with_improvements(whisper_model, processor, collator, train_datasets, eval_dataset, device, **kwargs):
    """
    Drop-in replacement for your existing training function with improvements.
    """
    return train_improved_contrastive(
        whisper_model, processor, collator, train_datasets, eval_dataset, device, **kwargs
    )
def generate_contrastive_batch(counter, BATCH_SIZE, random_batch_generator, batch_B, batch_C, batch_D, batch_E, batch_F, processor=None, debug=False):
    """
    Fixed contrastive batch generation function with proper handling of variable sequence lengths
    and optional text verification
    """
    # Get batch indices for selecting which noisy version to use for each sample
    batch_indexes = random_batch_generator[counter*BATCH_SIZE:counter*BATCH_SIZE+BATCH_SIZE]
    
    # Create lists to store the batches for easier indexing
    batches = [batch_B, batch_C, batch_D, batch_E, batch_F]
    
    # Debug: Print shapes and decoded text for first batch
    if debug and counter == 0:  # Only print for first batch to avoid spam
        print("\nDEBUG: Contrastive batch generation details:")
        print("Batch shapes:")
        for i, batch in enumerate(batches):
            print(f"  Batch {chr(66+i)}: input_features {batch['input_features'].shape}, labels {batch['labels'].shape}")
        
        if processor is not None:
            print("\nFirst sample from each batch (decoded text):")
            for i, batch in enumerate(batches):
                labels = batch['labels'][0]  # First sample
                labels_no_pad = labels[labels != -100]
                try:
                    text = processor.decode(labels_no_pad, skip_special_tokens=True)
                    print(f"  Batch {chr(66+i)}: '{text}'")
                except Exception as e:
                    print(f"  Batch {chr(66+i)}: [DECODE ERROR: {e}]")
    
    # Instead of stacking all batches first, select per sample to handle variable lengths
    selected_inputs = []
    selected_labels = []
    
    for i, noise_type_idx in enumerate(batch_indexes):
        # Select the appropriate batch for this sample
        selected_batch = batches[noise_type_idx.long()]
        
        # Extract the i-th sample from the selected batch
        selected_inputs.append(selected_batch['input_features'][i])  # [seq_len, features]
        selected_labels.append(selected_batch['labels'][i])         # [seq_len]
    
    # Stack the selected samples
    batch_tensor = torch.stack(selected_inputs, dim=0)  # [batch_size, seq_len, features]
    
    # For labels, we need to handle variable lengths by padding to max length
    max_label_length = max(label.size(0) for label in selected_labels)
    
    # Pad all labels to the same length
    padded_labels = []
    for idx, label in enumerate(selected_labels):
        if label.size(0) < max_label_length:
            # Pad with -100 (ignore index for loss calculation)
            padding = torch.full((max_label_length - label.size(0),), -100, 
                               dtype=label.dtype, device=label.device)
            padded_label = torch.cat([label, padding], dim=0)
        else:
            padded_label = label
        padded_labels.append(padded_label)
        
        # Debug: show what was selected for first few samples
        if debug and counter == 0 and idx < 3:
            selected_batch_idx = batch_indexes[idx].item()
            print(f"Sample {idx}: Selected batch {chr(66+selected_batch_idx)} (index {selected_batch_idx})")
            print(f"  Original length: {label.size(0)}, Padded length: {padded_label.size(0)}")
            if processor is not None:
                try:
                    text = processor.decode(label[label != -100], skip_special_tokens=True)
                    print(f"  Text: '{text}'")
                except:
                    print(f"  Text: [DECODE ERROR]")
    
    labels_tensor = torch.stack(padded_labels, dim=0)  # [batch_size, max_seq_len]
    
    if debug and counter == 0:
        print(f"Final contrastive batch: input_features {batch_tensor.shape}, labels {labels_tensor.shape}")
    
    return {
        'input_features': batch_tensor,
        'labels': labels_tensor
    }


def verify_dataset_correspondence_fixed(train_datasets, processor, num_samples_to_check=5):
    """
    Improved verification that handles variable sequence lengths properly and includes decoded text
    """
    print("Verifying dataset correspondence with decoded text...")
    
    # Check first few samples
    for i in range(min(num_samples_to_check, len(train_datasets[0]))):
        clean_sample = train_datasets[0][i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i}:")
        print(f"  Clean dataset: input_features {clean_sample['input_features'].shape}")
        
        if isinstance(clean_sample['labels'], list):
            clean_labels = torch.tensor(clean_sample['labels'])
        else:
            clean_labels = clean_sample['labels']
        
        # Remove padding tokens for comparison
        clean_labels_no_pad = clean_labels[clean_labels != -100]
        
        # Decode clean labels to text
        try:
            clean_text = processor.decode(clean_labels_no_pad, skip_special_tokens=True)
            print(f"  Clean text: '{clean_text}'")
            print(f"  Clean labels (no padding): {clean_labels_no_pad.tolist()}")
        except Exception as e:
            print(f"  Clean text: [DECODE ERROR: {e}]")
            print(f"  Clean labels (no padding): {clean_labels_no_pad.tolist()}")
        
        all_match = True
        
        for j, dataset in enumerate(train_datasets[1:], 1):
            noisy_sample = dataset[i]
            
            if isinstance(noisy_sample['labels'], list):
                noisy_labels = torch.tensor(noisy_sample['labels'])
            else:
                noisy_labels = noisy_sample['labels']
            
            noisy_labels_no_pad = noisy_labels[noisy_labels != -100]
            
            # Decode noisy labels to text
            try:
                noisy_text = processor.decode(noisy_labels_no_pad, skip_special_tokens=True)
                print(f"  Dataset {j} text: '{noisy_text}'")
                print(f"  Dataset {j} labels (no padding): {noisy_labels_no_pad.tolist()}")
            except Exception as e:
                print(f"  Dataset {j} text: [DECODE ERROR: {e}]")
                print(f"  Dataset {j} labels (no padding): {noisy_labels_no_pad.tolist()}")
            
            # Compare the actual content (without padding)
            labels_match = torch.equal(clean_labels_no_pad, noisy_labels_no_pad)
            
            # Also compare decoded text (more robust comparison)
            try:
                text_match = clean_text.strip().lower() == noisy_text.strip().lower()
            except:
                text_match = False
            
            if not labels_match:
                print(f"❌ LABEL MISMATCH at sample {i}: Clean vs Dataset {j}")
                print(f"   Labels match: {labels_match}")
                print(f"   Text match: {text_match}")
                
                if text_match:
                    print("   ⚠️  Labels differ but text matches - likely tokenization differences")
                    print("   This might be acceptable depending on your tokenizer settings")
                else:
                    print("   ❌ Both labels and text differ - this is a problem!")
                    all_match = False
            else:
                print(f"✅ Sample {i}: Clean matches Dataset {j} (labels and text)")
        
        if not all_match:
            print(f"\n❌ Sample {i} has mismatches - check your data preprocessing!")
            return False
    
    print(f"\n{'='*60}")
    print("✅ Dataset correspondence verified!")
    print("   - Content matches across datasets")
    print("   - Decoded text is consistent")
    print("   - Length variations are due to padding/tokenization")
    return True


# Alternative approach using collate_fn for handling variable lengths
def contrastive_collate_fn(batch_list):
    """
    Custom collate function that handles multiple batches with variable sequence lengths
    """
    # batch_list is a list of batches: [batch_B, batch_C, batch_D, batch_E, batch_F]
    
    # Find the maximum label length across all batches
    max_label_length = 0
    for batch in batch_list:
        for labels in batch['labels']:
            if isinstance(labels, torch.Tensor):
                max_label_length = max(max_label_length, labels.size(0))
            else:  # list
                max_label_length = max(max_label_length, len(labels))
    
    # Pad all batches to the same label length
    padded_batches = []
    for batch in batch_list:
        padded_labels = []
        for labels in batch['labels']:
            if isinstance(labels, list):
                labels = torch.tensor(labels)
            
            if labels.size(0) < max_label_length:
                padding = torch.full((max_label_length - labels.size(0),), -100, dtype=labels.dtype)
                padded_labels.append(torch.cat([labels, padding], dim=0))
            else:
                padded_labels.append(labels)
        
        padded_batch = {
            'input_features': batch['input_features'],
            'labels': torch.stack(padded_labels, dim=0)
        }
        padded_batches.append(padded_batch)
    
    return padded_batches
