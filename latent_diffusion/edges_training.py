import torch
from diffusion_train import get_parser
from model import TimeEmbedding, RectifiedFlowUNet256, RectifiedFlow, RectifiedFlowUNetWhisper, OptimizedRectifiedFlowUNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import os
import glob
import wandb
from pathlib import Path
import re
from identity import create_identity_dataloader, identity_collate_fn
from debugging import load_best_ema_model
from latent_visualization import visualize_whisper_batch, generate_validation_samples_ema, generate_samples_ema 

# Audio Dataset Classes (from first file)
class AudioPairDataset(Dataset):
    def __init__(self, root_dir, shuffle_pairs=False):
        """
        Dataset for pairing clean audio (persons) with noisy audio (mic1-5).
        Each epoch contains ALL possible (person, mic) combinations exactly once.
        
        Args:
            root_dir (str): Root directory containing persons, mic1, mic2, ..., mic5 folders
            shuffle_pairs (bool): Whether to shuffle the order of pairs each epoch
        """
        self.root_dir = Path(root_dir)
        self.shuffle_pairs = shuffle_pairs
        
        # Define directories
        self.persons_dir = self.root_dir / "persons"
        self.mic_dirs = [self.root_dir /  f"mic{i}" for i in range(1, 6)]
        
        # Validate directories exist
        self._validate_directories()
        
        # Get all person files and extract indices
        self.person_files = self._get_person_files()
        self.indices = self._extract_indices()
        
        # Create all valid (person_idx, mic_num) pairs
        self.valid_pairs = self._create_all_valid_pairs()
        
        # Shuffle pairs for first epoch
        self.current_epoch_pairs = self.valid_pairs.copy()
        if self.shuffle_pairs:
            random.shuffle(self.current_epoch_pairs)
        
        print(f"Found {len(self.indices)} person files")
        print(f"Created {len(self.valid_pairs)} total (person, mic) pairs")
        print(f"Each epoch will iterate over all {len(self.valid_pairs)} pairs exactly once")
    
    def _validate_directories(self):
        """Check if all required directories exist"""
        if not self.persons_dir.exists():
            raise FileNotFoundError(f"Persons directory not found: {self.persons_dir}")
        
        for mic_dir in self.mic_dirs:
            if not mic_dir.exists():
                raise FileNotFoundError(f"Mic directory not found: {mic_dir}")
    
    def _get_person_files(self):
        """Get all .pth files from persons directory"""
        person_files = list(self.persons_dir.glob("*P.pth"))
        if not person_files:
            raise FileNotFoundError("No person files (*P.pth) found in persons directory")
        return sorted(person_files)
    
    def _extract_indices(self):
        """Extract numerical indices from person filenames"""
        indices = []
        pattern = r'(\d+)P\.pth$'
        
        for file_path in self.person_files:
            match = re.search(pattern, file_path.name)
            if match:
                indices.append(int(match.group(1)))
            else:
                print(f"Warning: Skipping file with unexpected format: {file_path.name}")
        
        return sorted(indices)
    def _create_all_valid_pairs(self):
        valid_pairs = [] 
        for person_idx in self.indices:
            # Only check mic1 since we only have one mic directory
            mic_file = self.mic_dirs[0] / f"{person_idx}M1.pth"
            if mic_file.exists():
                valid_pairs.append((person_idx, 1))  # Always mic 1
                    
        if not valid_pairs:
            raise FileNotFoundError("No valid (person, mic1) pairs found")
            
        print(f"Found {len(valid_pairs)} valid person-mic1 pairs")
        return valid_pairs
    def __len__(self):
        return len(self.current_epoch_pairs)
    
    def on_epoch_start(self):
        """Call this at the start of each epoch to reshuffle pair order"""
        if self.shuffle_pairs:
            self.current_epoch_pairs = self.valid_pairs.copy()
            random.shuffle(self.current_epoch_pairs)
            print(f"Reshuffled {len(self.current_epoch_pairs)} pairs for new epoch")
    
    def __getitem__(self, idx):
        """Get a specific (person, mic1) pair - SIMPLIFIED"""
        # Get the specific pair for this index
        person_idx, mic_num = self.current_epoch_pairs[idx]  # mic_num will always be 1
      
        # Load clean audio (person) - this will be the target
        person_file = self.persons_dir / f"{person_idx}P.pth"
        clean_audio = torch.load(person_file, map_location='cpu')
        
        # Load noisy audio (mic1) - this will be the source
        mic_file = self.mic_dirs[0] / f"{person_idx}M1.pth"  # Always mic1
        noisy_audio = torch.load(mic_file, map_location='cpu')
        
        return {
            'source': noisy_audio,    # Source domain (noisy mic1 audio)
            'target': clean_audio,    # Target domain (clean person audio)
            'clean': clean_audio,     # For backward compatibility
            'noisy': noisy_audio,     # For backward compatibility
            'index': person_idx,
            'mic_used': 1  # Always 1
        }
def create_audio_dataloader(root_dir, batch_size=32, shuffle=True, num_workers=4, 
                           shuffle_pairs=False, **kwargs):  # Set shuffle_pairs=False
    """
    Create a DataLoader for all possible audio pairs suitable for rectified flow training.
    """
    dataset = AudioPairDataset(
        root_dir=root_dir,
        shuffle_pairs=False  # Disable custom shuffling
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Use DataLoader's shuffle instead
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )

def collate_fn(batch):
    """
    Custom collate function for handling variable-length tensors
    Suitable for rectified flow training
    """
    source_tensors = [item['source'] for item in batch]
    target_tensors = [item['target'] for item in batch]
    clean_tensors = [item['clean'] for item in batch]
    noisy_tensors = [item['noisy'] for item in batch]
    indices = [item['index'] for item in batch]
    mics_used = [item['mic_used'] for item in batch]
    # Stack tensors (assuming they have the same shape)
    try:
        source_batch = torch.stack(source_tensors)
        target_batch = torch.stack(target_tensors)
        clean_batch = torch.stack(clean_tensors)
        noisy_batch = torch.stack(noisy_tensors)
    except RuntimeError as e:
        print(f"Error stacking tensors: {e}")
        print("Source shapes:", [t.shape for t in source_tensors])
        print("Target shapes:", [t.shape for t in target_tensors])
        raise
    
    return {
        'source': source_batch,    # For rectified flow: noisy -> clean
        'target': target_batch,    # For rectified flow: noisy -> clean
        'clean': clean_batch,      # For backward compatibility
        'noisy': noisy_batch,      # For backward compatibility
        'indices': indices,
        'mics_used': mics_used
    }


# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def train_rectified_flow_audio(model, dataloader, validation_loader, num_epochs, lr=1e-4, ema=0.99, opt_decay=1e-3, scheduler_type='onecycle', checkpoint = ""):
    """Modified training function for audio data"""
    rectified_flow = RectifiedFlow(model, device)
    
    # Improved optimizer with better defaults
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=opt_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Multiple scheduler options
    if scheduler_type == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3*lr,
            epochs=num_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4
        )
        step_per_batch = True
    elif scheduler_type == 'cosine_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=lr * 0.01
        )
        step_per_batch = False
    elif scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=lr * 0.001,
            verbose=True
        )
        step_per_batch = False
    else:  # 'cosine'
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        step_per_batch = False
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # EMA for model parameters
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.99))

    model.train()
    losses = []
    best_loss = float('inf')
    if checkpoint != "":
        try:
            model = load_best_ema_model(checkpoint, device)
            model.eval()
            
            # Handle EMA model wrapper
            if hasattr(model, 'module'):
                flow_model = model.module
            else:
                flow_model = model
                
            # Create RectifiedFlow instance
            rectified_flow = RectifiedFlow(flow_model, device)
            print("✓ RectifiedFlow instance created successfully!")
            
            # Print model info
            total_params = sum(p.numel() for p in flow_model.parameters())
            print(f"✓ Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        model.train()
        # Get a sample batch from your dataloader
    for epoch in range(num_epochs):
        # Reshuffle pairs for new epoch
        dataloader.dataset.on_epoch_start()
        
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        avg_mse_loss = 0
        batch_counter = 0

        for batch_idx, batch in enumerate(pbar):
            # Extract source and target from batch dictionary
            source = batch['source'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            #target = source + 0.1*(target - source)
            mse_loss = torch.nn.functional.mse_loss(source, target)
            avg_mse_loss = avg_mse_loss + mse_loss
            batch_counter = batch_counter + 1
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if checkpoint != "":
                loss = rectified_flow.velocity_loss(target, source)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            else:

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss = rectified_flow.velocity_loss(target, source)
                        if torch.isnan(loss) or loss > 100:
                            print(f"Skipping batch {batch_idx} due to unstable loss: {loss.item()}")
                            continue
               
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = rectified_flow.velocity_loss(target, source)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
            # Update EMA
            ema_model.update_parameters(model)
            
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}',
            })
            
            # Step scheduler per batch if needed
            if step_per_batch:
                scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        wandb.log({"losses": avg_loss})
        assert(avg_mse_loss/batch_counter <=0.15)
        # Step scheduler per epoch
        if not step_per_batch:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}, LR: {current_lr:.2e}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, '8_identity8_test_identity_50mapping.pth')
        
        # Generate samples every 10 epochs using EMA model
        if (epoch + 1) % 1 == 0:
            generate_audio_samples_ema(ema_model, dataloader, epoch + 1)
            if validation_loader:
                generate_audio_validation_samples_ema(ema_model, validation_loader, epoch + 1) 
    
    return losses, [], ema_model


def generate_audio_samples_ema(ema_model, dataloader, epoch):
    """Generate and visualize audio samples using EMA model"""
    rectified_flow = RectifiedFlow(ema_model.module, device)
    ema_model.eval()
    
    # Get a batch of source audio data
    batch = next(iter(dataloader))
    source = batch['source'][:4].to(device)  # Take first 4 samples
    target = batch['target'][:4].to(device)
    
    # Generate samples with different step counts
    with torch.no_grad():
        generated_25 = rectified_flow.sample(source, num_steps=25)
        generated_100 = rectified_flow.sample(source, num_steps=100)
    
    # Use the visualization function from latent_visualization
    try:
        # Visualize original comparison
        visualize_whisper_batch(
            clean_audio=target - source,  # Difference as clean
            prediction=generated_25 - source,  # Generated difference
            save_path=f'audio_samples_epoch_{epoch}_25steps.png'
        )
        wandb.log({f"audio_samples_epoch_{epoch}_25steps": wandb.Image(f'audio_samples_epoch_{epoch}_25steps.png')})
        
        # Visualize with 100 steps
        visualize_whisper_batch(
            clean_audio=target - source,
            prediction=generated_100 - source,
            save_path=f'audio_samples_epoch_{epoch}_100steps.png'
        )
        wandb.log({f"audio_samples_epoch_{epoch}_100steps": wandb.Image(f'audio_samples_epoch_{epoch}_100steps.png')})
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        # Fallback: just save some basic info
        print(f"Generated shapes - 25 steps: {generated_25.shape}, 100 steps: {generated_100.shape}")


def generate_audio_validation_samples_ema(ema_model, dataloader, epoch):
    """Generate validation samples for audio data"""
    generate_audio_samples_ema(ema_model, dataloader, epoch)  # Same logic for now


# Main training script
def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Set your audio dataset path here
    if args.environment == 'bwcluster':
        AUDIO_BASE_DIR = "/pfs/work9/workspace/scratch/ka_uhicv-blah/audio_data"  # Update this path
        AUDIO_BASE_DIR = "/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion"  # Update this path
    else:
        AUDIO_BASE_DIR = "/home/ka/ka_stud/ka_uhicv"  # Update this path
    
    # Check if base directory exists
    if not os.path.exists(AUDIO_BASE_DIR):
        print(f"Error: Audio base directory '{AUDIO_BASE_DIR}' not found!")
        print("Please update AUDIO_BASE_DIR path in the main() function")
        return
    
    # Create audio datasets
    print("Creating audio datasets...")
    
    # Get audio tensor dimensions by loading one sample
    try:
        sample_person_file = list(Path(AUDIO_BASE_DIR).glob("persons/*P.pth"))[0]
        sample_tensor = torch.load(sample_person_file, map_location='cpu')
        print(f"Audio tensor shape: {sample_tensor.shape}")
        audio_channels = sample_tensor.shape[0] if len(sample_tensor.shape) > 1 else 1
        print(f"Detected {audio_channels} audio channels")
    except Exception as e:
        print(f"Error loading sample audio file: {e}")
        return
    
    # Training parameters
    batch_size = args.batch_size
    print(batch_size)
    num_epochs = args.num_epochs
    lr = args.lr
    scheduler_type = args.scheduler_type
    
    wandb.init(project='audio_diffusion', config={
        "lr": lr,
        "batch_size": batch_size,
        "ema": args.ema,
        "weight_decay": args.weight_decay,
        "audio_shape": str(sample_tensor.shape),
        "num_epochs": num_epochs,
        "scheduler_type": scheduler_type,
        "run_notes": args.run_notes,
        "path" : args.checkpoint
    })
    
    # Create dataloaders
    train_dataloader = create_audio_dataloader(
        root_dir=AUDIO_BASE_DIR,
        batch_size=batch_size,
        shuffle=True,  # We handle shuffling in the dataset
        num_workers=4,
        shuffle_pairs=True,
        collate_fn=collate_fn
    )

    #train_dataloader = create_identity_dataloader( root_dir=AUDIO_BASE_DIR,batch_size=batch_size,shuffle=True,num_workers=4,shuffle_files=True,collate_fn=identity_collate_fn )
    
    # For now, use train dataloader as validation (as requested)
    val_dataloader = create_audio_dataloader(
        root_dir=AUDIO_BASE_DIR,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        shuffle_pairs=False,  # Don't shuffle validation
        collate_fn=collate_fn
    )
    
    print(f"Train dataset size: {len(train_dataloader.dataset)} pairs")
    print(f"Batches per epoch: {len(train_dataloader)}")
    
    # Create model based on audio tensor dimensions
    # Assuming audio data needs special handling - using RectifiedFlowUNetWhisper
    if len(sample_tensor.shape) != 3:  # [channels, height, width] format
        in_channels = sample_tensor.shape[0] * 2  # source + target channels
        out_channels = sample_tensor.shape[0]
        model = RectifiedFlowUNetWhisper(in_channels=2, out_channels=out_channels).to(device)
    else:
        # Fallback for different tensor formats
        pass
    model = OptimizedRectifiedFlowUNet(in_channels=2, out_channels=1).to(device)
    print("using optimized Flow unet")
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model input/output channels: {model}")
    
    # Train model
    print("Starting audio rectified flow training...")
    train_losses, val_losses, ema_model = train_rectified_flow_audio(
        model, 
        train_dataloader,
        val_dataloader,
        num_epochs=num_epochs,
        lr=lr,
        ema=args.ema,
        opt_decay=args.weight_decay,
        scheduler_type=scheduler_type,
        checkpoint = args.checkpoint,
    )
    wandb.finish()
    
    # Plot training loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Audio Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('audio_training_loss.png')
    plt.show()


if __name__ == "__main__":
    main()
