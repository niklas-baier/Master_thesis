import torch
from diffusion_train import get_parser
from model import TimeEmbedding, RectifiedFlowUNet256, RectifiedFlow, RectifiedFlowUNetWhisper, OptimizedRectifiedFlowUNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
import os
import glob
import wandb
# Enable optimizations
torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster training
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

# Modified Dataset class for edges2shoes dataset with 1500x1280 resolution
class Edges2ShoesDataset(Dataset):
    def __init__(self, base_dir, split='train', image_size=(1500, 1280)):  # CHANGED: Updated default size
        """
        Args:
            base_dir: Base directory containing train/val folders
            split: 'train' or 'val'
            image_size: Target image size (height, width) - now 1500x1280
        """
        self.base_dir = base_dir
        self.split = split
        self.image_size = image_size
        
        # Path to split directory
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory '{split_dir}' not found!")
        
        # Get all AB.jpg files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            pattern = os.path.join(split_dir, f"*_AB.{ext.split('.')[-1]}")
            self.image_files.extend(glob.glob(pattern))
        
        self.image_files.sort()
        
        print(f"Found {len(self.image_files)} images in {split} split")
        print(f"Target image size: {image_size}")
        
        if len(self.image_files) == 0:
            print(f"Warning: No images found in {split_dir}")
            print("Expected format: number_AB.jpg")
        
        # Define transforms - convert to grayscale and normalize for high resolution
        # CHANGED: Updated transforms for better high-resolution handling
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.LANCZOS),  # LANCZOS for better quality
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] for grayscale
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load the combined image
        img_path = self.image_files[idx]
        combined_img = Image.open(img_path).convert('RGB')
        
        # Get image dimensions
        width, height = combined_img.size
        
        # Split the image in half
        # Left half: edges, Right half: shoes
        edge_img = combined_img.crop((0, 0, width // 2, height))
        shoe_img = combined_img.crop((width // 2, 0, width, height))
        
        # Apply transforms (includes conversion to grayscale and resize to 1500x1280)
        edge_tensor = self.transform(edge_img)
        shoe_tensor = self.transform(shoe_img)
        
        return edge_tensor, shoe_tensor

def train_rectified_flow(model, dataloader, validation_loader, num_epochs, lr=1e-4, ema=0.99, opt_decay=1e-3, scheduler_type='onecycle'):
    rectified_flow = RectifiedFlow(model, device)
    
    # Improved optimizer with better defaults
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=opt_decay,  # Higher weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Multiple scheduler options
    if scheduler_type == 'onecycle':
        # OneCycle: Often best for diffusion models
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=3*lr,  # Peak at 3x base LR
            epochs=num_epochs,
            steps_per_epoch=len(dataloader),
            pct_start=0.3,  # Reach peak at 30%
            div_factor=25,  # Initial LR = max_lr/25
            final_div_factor=1e4
        )
        step_per_batch = True
    elif scheduler_type == 'cosine_restarts':
        # Cosine with warm restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # First restart after 10 epochs
            T_mult=1,  # Keep same cycle length
            eta_min=lr * 0.01  # Min LR = 1% of base
        )
        step_per_batch = False
    elif scheduler_type == 'reduce_on_plateau':
        # Adaptive based on loss
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
    ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.999))
    
    if "hello " == "":  # Checkpoint loading logic (currently disabled)
        best_dict = torch.load(args.checkpoint)
        model.load_state_dict(best_dict['model_state_dict'])
        ema_model.load_state_dict(best_dict['ema_state_dict'])
        scheduler.load_state_dict(best_dict['scheduler_state_dict'])
        optimizer.load_state_dict(['optimizer_state_dict'])
        print("successfully loaded old training state")

    model.train()
    losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (source, target) in enumerate(pbar):
            source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = rectified_flow.velocity_loss(source, target)
                    if torch.isnan(loss) or loss > 100:
                        print(f"Skipping batch {batch_idx} due to unstable loss: {loss.item()}")
                        continue
           
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = rectified_flow.velocity_loss(source, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            # Update EMA
            ema_model.update_parameters(model)
            
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # Step scheduler per batch if needed
            if step_per_batch:
                scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        wandb.log({"losses": avg_loss})
        
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
            }, 'best_model_1500x1280.pth')  # CHANGED: Updated filename to reflect resolution
        
        # Generate samples every epoch using EMA model
        if (epoch + 1) % 1 == 0:
            generate_samples_ema(ema_model, dataloader, epoch + 1)
            if validation_loader:
                generate_validation_samples_ema(ema_model, validation_loader, epoch + 1) 
    
    return losses, 0, ema_model

def generate_validation_samples_ema(ema_model, dataloader, epoch):
    """Generate and visualize samples using EMA model for high resolution"""
    rectified_flow = RectifiedFlow(ema_model.module, device)
    ema_model.eval()
    
    # Get a batch of source images (reduce to 4 for memory efficiency at high res)
    source, target = next(iter(dataloader))
    source, target = source[:4].to(device), target[:4].to(device)  # CHANGED: Reduced from 8 to 4 samples
    
    # Generate samples with different step counts
    with torch.no_grad():
        generated_25 = rectified_flow.sample(source, num_steps=25)
        generated_50 = rectified_flow.sample(source, num_steps=100)
    
    # Denormalize for visualization
    source_vis = (source + 1) / 2
    target_vis = (target + 1) / 2
    generated_25_vis = torch.clamp((generated_25 + 1) / 2, 0, 1)
    generated_50_vis = torch.clamp((generated_50 + 1) / 2, 0, 1)
    
    # Plot comparison - CHANGED: Adjusted for high resolution display
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))  # CHANGED: Square layout for better visualization
    
    for i in range(4):
        axes[0, i].imshow(source_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Source (Edges)')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(generated_25_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title('Generated (25 steps)')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(generated_50_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title('Generated (100 steps)')
        axes[2, i].axis('off')
        
        axes[3, i].imshow(target_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[3, i].set_title('Target (Shoes)')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'valid_samples_epoch_{epoch}_1500x1280.png', dpi=150, bbox_inches='tight')
    wandb.log({f"valid_samples_epoch_{epoch}_1500x1280.png": wandb.Image(fig)})
    plt.close()  # ADDED: Close figure to free memory

def generate_samples_ema(ema_model, dataloader, epoch):
    """Generate and visualize samples using EMA model for high resolution"""
    rectified_flow = RectifiedFlow(ema_model.module, device)
    ema_model.eval()
    
    # Get a batch of source images (reduce to 4 for memory efficiency at high res)
    source, target = next(iter(dataloader))
    source, target = source[:4].to(device), target[:4].to(device)  # CHANGED: Reduced from 8 to 4 samples
    
    # Generate samples with different step counts
    with torch.no_grad():
        generated_25 = rectified_flow.sample(source, num_steps=25)
        generated_50 = rectified_flow.sample(source, num_steps=100)
    
    # Denormalize for visualization
    source_vis = (source + 1) / 2
    target_vis = (target + 1) / 2
    generated_25_vis = torch.clamp((generated_25 + 1) / 2, 0, 1)
    generated_50_vis = torch.clamp((generated_50 + 1) / 2, 0, 1)
    
    # Plot comparison - CHANGED: Adjusted for high resolution display
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))  # CHANGED: Square layout for better visualization
    
    for i in range(4):
        axes[0, i].imshow(source_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Source (Edges)')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(generated_25_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[1, i].set_title('Generated (25 steps)')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(generated_50_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[2, i].set_title('Generated (100 steps)')
        axes[2, i].axis('off')
        
        axes[3, i].imshow(target_vis[i, 0].cpu().numpy(), cmap='gray')
        axes[3, i].set_title('Target (Shoes)')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'samples_epoch_{epoch}_1500x1280.png', dpi=150, bbox_inches='tight')
    wandb.log({f"samples_epoch_{epoch}_1500x1280.png": wandb.Image(fig)})
    plt.close()  # ADDED: Close figure to free memory

def generate_samples(model, dataloader, epoch):
    """Backward compatibility function"""
    generate_samples_ema(model, dataloader, epoch)

# Main training script
def main():
    # Set your dataset path here
    parser = get_parser()
    args = parser.parse_args()
    if args.environment == 'bwcluster':
        BASE_DIR = "/pfs/work9/workspace/scratch/ka_uhicv-blah/latent_diffusion/edges2shoes-dataset/versions/1"
    else:
        BASE_DIR = "/home/nbaier/.cache/kagglehub/datasets/balraj98/edges2shoes-dataset/versions/1/"
    
    # Check if base directory exists
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory '{BASE_DIR}' not found!")
        print("Please update BASE_DIR path in the main() function")
        return
    
    # Check if train and val directories exist
    train_dir = os.path.join(BASE_DIR, 'train')
    val_dir = os.path.join(BASE_DIR, 'val')
    
    if not os.path.exists(train_dir):
        print(f"Error: Train directory '{train_dir}' not found!")
        return
    
    # Create datasets with high resolution - CHANGED: Updated to 1500x1280
    train_dataset = Edges2ShoesDataset(BASE_DIR, split='train', image_size=(1500, 1280))
    val_dataset = None
    if os.path.exists(val_dir):
        val_dataset = Edges2ShoesDataset(BASE_DIR, split='val', image_size=(1500, 1280))
        print(f"Validation dataset created with {len(val_dataset)} samples")
    else:
        print("Validation directory not found, training without validation")
    
    # Create dataloaders - CHANGED: Reduced batch size for high resolution
    batch_size = max(1, args.batch_size // 4)  # CHANGED: Reduce batch size significantly for high res
    num_epochs = args.num_epochs
    lr = args.lr
    scheduler_type = args.scheduler_type
    
    print(f"Adjusted batch size for 1500x1280 resolution: {batch_size}")
    
    wandb.init(project='diffusion', config={
        "lr": lr,
        "batch_size": batch_size,
        "ema": args.ema,
        "weight_decay": 1e-2,
        "size": "1500x1280",  # CHANGED: Updated size info
        "num_epochs": num_epochs,
        "scheduler_type": scheduler_type,
        "run_notes": args.run_notes
    })
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2,  # CHANGED: Reduced workers for memory efficiency
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )
    
    # CHANGED: Use RectifiedFlowUNetWhisper for high resolution (1500x1280)
    model = OptimizedRectifiedFlowUNet(in_channels=2, out_channels=1).to(device)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Using RectifiedFlowUNetWhisper for 1500x1280 resolution")
    
    # Train model with more conservative settings for high resolution
    print("Starting training with 1500x1280 resolution...")
    train_losses, val_losses, ema_model = train_rectified_flow(
        model, 
        train_dataloader,
        val_dataloader,
        num_epochs=args.num_epochs,
        lr=lr,
        ema=args.ema,
        opt_decay=args.weight_decay,
        scheduler_type=scheduler_type
    )
    wandb.finish()
    
    # Plot training loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Training Loss (1500x1280)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss_1500x1280.png')
    plt.show()

if __name__ == "__main__":
    main()
