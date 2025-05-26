import torch
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

# Modified Dataset class for edges2shoes dataset
class Edges2ShoesDataset(Dataset):
    def __init__(self, base_dir, split='train', image_size=(256, 256)):
        """
        Args:
            base_dir: Base directory containing train/val folders
            split: 'train' or 'val'
            image_size: Target image size (height, width)
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
        
        if len(self.image_files) == 0:
            print(f"Warning: No images found in {split_dir}")
            print("Expected format: number_AB.jpg")
        
        # Define transforms - convert to grayscale and normalize
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
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
        
        # Apply transforms (includes conversion to grayscale)
        edge_tensor = self.transform(edge_img)
        shoe_tensor = self.transform(shoe_img)
        
        return edge_tensor, shoe_tensor

# Attention Block for U-Net
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(min(8, channels), channels)  # Fix for small channel counts
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Initialize weights properly
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.xavier_uniform_(self.v.weight)
        nn.init.zeros_(self.proj.weight)  # Zero init for residual
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        
        q = self.q(h).view(B, C, H * W).transpose(1, 2)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W).transpose(1, 2)
        
        attention = torch.softmax(torch.bmm(q, k) / (C ** 0.5), dim=2)
        out = torch.bmm(attention, v).transpose(1, 2).view(B, C, H, W)
        
        return x + self.proj(out)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),  # Fix for small channel counts
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Better weight initialization
        nn.init.xavier_uniform_(self.block1[2].weight)
        nn.init.zeros_(self.block2[2].weight)  # Zero init for residual path
        if isinstance(self.residual_conv, nn.Conv2d):
            nn.init.xavier_uniform_(self.residual_conv.weight)
    
    def forward(self, x, time_embedding):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_embedding)[:, :, None, None]
        h = h + time_emb
        
        h = self.block2(h)
        
        return h + self.residual_conv(x)

# Time Embedding
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Enhanced U-Net for Rectified Flow
'''
class RectifiedFlowUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_embedding_dim=256, base_channels=128):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # Encoder - increased channel counts for larger images
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Level 1: base_channels -> base_channels
        self.down1 = nn.ModuleList([
            ResidualBlock(base_channels, base_channels, time_embedding_dim),
            ResidualBlock(base_channels, base_channels, time_embedding_dim)
        ])
        self.down_conv1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        
        # Level 2: base_channels*2 -> base_channels*2
        self.down2 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels * 2, time_embedding_dim),
            ResidualBlock(base_channels * 2, base_channels * 2, time_embedding_dim)
        ])
        self.down_conv2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        # Level 3: base_channels*4 -> base_channels*4
        self.down3 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 4, time_embedding_dim),
            ResidualBlock(base_channels * 4, base_channels * 4, time_embedding_dim)
        ])
        self.down_conv3 = nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)
        
        # Middle - with attention at the bottleneck
        self.middle = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 8, time_embedding_dim),
            AttentionBlock(base_channels * 8),
            ResidualBlock(base_channels * 8, base_channels * 8, time_embedding_dim)
        ])
        
        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(base_channels * 8, base_channels * 4, time_embedding_dim),  # 8*base = 4*base + 4*base (skip)
            ResidualBlock(base_channels * 4, base_channels * 4, time_embedding_dim)
        ])
        
        self.up_conv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(base_channels * 4, base_channels * 2, time_embedding_dim),  # 4*base = 2*base + 2*base (skip)
            ResidualBlock(base_channels * 2, base_channels * 2, time_embedding_dim)
        ])
        
        self.up_conv3 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([
            ResidualBlock(base_channels * 2, base_channels, time_embedding_dim),   # 2*base = base + base (skip)
            ResidualBlock(base_channels, base_channels, time_embedding_dim)
        ])
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        
        # Initialize output layer to zero for better stability
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Encoder
        h0 = self.conv_in(x)
        
        h1 = h0
        for block in self.down1:
            h1 = block(h1, time_emb)
        h1_pool = self.down_conv1(h1)
        
        h2 = h1_pool
        for block in self.down2:
            h2 = block(h2, time_emb)
        h2_pool = self.down_conv2(h2)
        
        h3 = h2_pool
        for block in self.down3:
            h3 = block(h3, time_emb)
        h3_pool = self.down_conv3(h3)
        
        # Middle
        h = h3_pool
        for i, block in enumerate(self.middle):
            if i == 1:  # Attention block
                h = block(h)
            else:
                h = block(h, time_emb)
        
        # Decoder
        h = self.up_conv1(h)
        h = torch.cat([h, h3], dim=1)  # Skip connection
        for block in self.up1:
            h = block(h, time_emb)
        
        h = self.up_conv2(h)
        h = torch.cat([h, h2], dim=1)  # Skip connection
        for block in self.up2:
            h = block(h, time_emb)
        
        h = self.up_conv3(h)
        h = torch.cat([h, h1], dim=1)  # Skip connection
        for block in self.up3:
            h = block(h, time_emb)
        
        return self.conv_out(h)

# Rectified Flow Model
class RectifiedFlow:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def forward_process(self, x0, x1, t):
        """Forward process: x_t = t * x1 + (1 - t) * x0"""
        t = t.view(-1, 1, 1, 1)
        return t * x1 + (1 - t) * x0
    
    def velocity_loss(self, x0, x1):
        """Compute the velocity matching loss"""
        batch_size = x0.shape[0]
        
        # Sample random times
        t = torch.rand(batch_size, device=self.device)
        
        # Get interpolated samples
        x_t = self.forward_process(x0, x1, t)
        
        # Concatenate source image with current state
        model_input = torch.cat([x0, x_t], dim=1)
        
        # Predict velocity
        v_pred = self.model(model_input, t)
        
        # True velocity is x1 - x0
        v_true = x1 - x0
        
        # L2 loss with potential numerical stability
        loss = F.mse_loss(v_pred, v_true)
        
        # Check for NaN and return a large but finite loss if so
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping this batch")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        
        return loss
    
    def sample(self, x0, num_steps=50):
        """Generate samples using Euler method"""
        self.model.eval()
        
        with torch.no_grad():
            x = x0.clone()
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.full((x0.shape[0],), i * dt, device=self.device)
                model_input = torch.cat([x0, x], dim=1)
                v = self.model(model_input, t)
                x = x + dt * v
        
        return x
'''
class RectifiedFlowUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, time_embedding_dim=128):  # in_channels=2 for source+target concat
        super().__init__()
        
        self.time_embedding = TimeEmbedding(time_embedding_dim)
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        self.down1 = nn.ModuleList([
            ResidualBlock(64, 64, time_embedding_dim),
            ResidualBlock(64, 64, time_embedding_dim)
        ])
        self.down_conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        
        self.down2 = nn.ModuleList([
            ResidualBlock(128, 128, time_embedding_dim),
            ResidualBlock(128, 128, time_embedding_dim)
        ])
        self.down_conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Middle
        self.middle = nn.ModuleList([
            ResidualBlock(256, 256, time_embedding_dim),
            AttentionBlock(256),
            ResidualBlock(256, 256, time_embedding_dim)
        ])
        
        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResidualBlock(256, 128, time_embedding_dim),  # 256 = 128 + 128 (skip connection)
            ResidualBlock(128, 128, time_embedding_dim)
        ])
        
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResidualBlock(128, 64, time_embedding_dim),   # 128 = 64 + 64 (skip connection)
            ResidualBlock(64, 64, time_embedding_dim)
        ])
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        time_emb = self.time_embedding(t)
        
        # Encoder
        h0 = self.conv_in(x)
        
        h1 = h0
        for block in self.down1:
            h1 = block(h1, time_emb)
        h1_pool = self.down_conv1(h1)
        
        h2 = h1_pool
        for block in self.down2:
            h2 = block(h2, time_emb)
        h2_pool = self.down_conv2(h2)
        
        # Middle
        h = h2_pool
        for i, block in enumerate(self.middle):
            if i == 1:  # Attention block
                h = block(h)
            else:
                h = block(h, time_emb)
        
        # Decoder
        h = self.up_conv1(h)
        h = torch.cat([h, h2], dim=1)  # Skip connection
        for block in self.up1:
            h = block(h, time_emb)
        
        h = self.up_conv2(h)
        h = torch.cat([h, h1], dim=1)  # Skip connection
        for block in self.up2:
            h = block(h, time_emb)
        
        return self.conv_out(h)

# Rectified Flow Model
class RectifiedFlow:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def forward_process(self, x0, x1, t):
        """Forward process: x_t = t * x1 + (1 - t) * x0"""
        t = t.view(-1, 1, 1, 1)
        return t * x1 + (1 - t) * x0
    
    def velocity_loss(self, x0, x1):
        """Compute the velocity matching loss"""
        batch_size = x0.shape[0]
        
        # Sample random times
        t = torch.rand(batch_size, device=self.device)
        
        # Get interpolated samples
        x_t = self.forward_process(x0, x1, t)
        
        # Concatenate source image with current state
        model_input = torch.cat([x0, x_t], dim=1)
        
        # Predict velocity
        v_pred = self.model(model_input, t)
        
        # True velocity is x1 - x0
        v_true = x1 - x0
        
        # L2 loss
        loss = F.mse_loss(v_pred, v_true)
        return loss
    
    def sample(self, x0, num_steps=50):
        """Generate samples using Euler method"""
        self.model.eval()
        
        with torch.no_grad():
            x = x0.clone()
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.full((x0.shape[0],), i * dt, device=self.device)
                model_input = torch.cat([x0, x], dim=1)
                v = self.model(model_input, t)
                x = x + dt * v
        
        return x
# Improved training function with better stability
'''def train_rectified_flow(model, train_dataloader, val_dataloader=None, num_epochs=30, lr=1e-4, scheduler_type='cosine'):
    rectified_flow = RectifiedFlow(model, device)
    
    # More conservative optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-3,  # Slightly higher weight decay
        betas=(0.9, 0.95),  # More conservative beta2
        eps=1e-8
    )
    
    # More conservative scheduler options
    if scheduler_type == 'onecycle':
        # Much more conservative OneCycle
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=4*lr,  # Only 2x peak instead of 5x
            epochs=num_epochs,
            steps_per_epoch=len(train_dataloader),
            pct_start=0.3,
            div_factor=10,  # Less aggressive ramp up
            final_div_factor=100  # More aggressive final decay
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
            factor=0.7,  # Less aggressive reduction
            patience=5,  # More patience
            min_lr=lr * 0.001,
            verbose=True
        )
        step_per_batch = False
    else:  # 'cosine' - default and most stable
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs,
            eta_min=lr * 0.01
        )
        step_per_batch = False
    
    # More conservative mixed precision settings
    
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**10,  # Lower initial scale
        growth_factor=1.1,  # Slower growth
        backoff_factor=0.8,  # More aggressive backoff
        growth_interval=200  # Less frequent growth
    ) if device.type == 'cuda' else None
    
    # EMA for model parameters with more conservative decay
    ema_model = torch.optim.swa_utils.AveragedModel(
        model, 
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.99)  # Slower EMA
    )
    
    model.train()
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_epoch_loss = 0.0
        num_valid_batches = 0
        pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (source, target) in enumerate(pbar):
            source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass with more conservative settings
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss = rectified_flow.velocity_loss(source, target)
                
                # Skip batch if loss is NaN or too large
                if torch.isnan(loss) or loss > 100:
                    print(f"Skipping batch {batch_idx} due to unstable loss: {loss.item()}")
                    continue
                
                scaler.scale(loss).backward()
                
                # More aggressive gradient clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Lower threshold
                
                if grad_norm > 10:  # Skip if gradients are too large
                    print(f"Skipping step due to large gradients: {grad_norm}")
                    scaler.update()
                    continue
                
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = rectified_flow.velocity_loss(source, target)
                
                if torch.isnan(loss) or loss > 100:
                    print(f"Skipping batch {batch_idx} due to unstable loss: {loss.item()}")
                    continue
                
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                if grad_norm > 10:
                    print(f"Skipping step due to large gradients: {grad_norm}")
                    continue
                
                optimizer.step()
            
            # Update EMA only on successful steps
            ema_model.update_parameters(model)
            
            train_epoch_loss += loss.item()
            num_valid_batches += 1
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}',
                'GradNorm': f'{grad_norm:.2f}' if 'grad_norm' in locals() else 'N/A'
            })
            
            # Step scheduler per batch if needed - AFTER optimizer.step()
            if step_per_batch:
                scheduler.step()
        
        # Calculate average loss only from valid batches
        if num_valid_batches > 0:
            avg_train_loss = train_epoch_loss / num_valid_batches
        else:
            print(f"Warning: No valid batches in epoch {epoch+1}")
            avg_train_loss = float('inf')
        
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = 0.0
        if val_dataloader is not None:
            model.eval()
            rectified_flow_val = RectifiedFlow(model, device)
            val_valid_batches = 0
            with torch.no_grad():
                for source, target in val_dataloader:
                    source, target = source.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    loss = rectified_flow_val.velocity_loss(source, target)
                    if not torch.isnan(loss) and loss < 100:
                        val_loss += loss.item()
                        val_valid_batches += 1
            
            if val_valid_batches > 0:
                val_loss = val_loss / val_valid_batches
            else:
                val_loss = float('inf')
            
            val_losses.append(val_loss)
        
        # Step scheduler per epoch - AFTER computing losses
        if not step_per_batch:
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_loss if val_dataloader else avg_train_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        if val_dataloader:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}')
        else:
            print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.2e}')
        
        # Save best model based on validation loss if available, otherwise training loss
        current_loss = val_loss if val_dataloader else avg_train_loss
        if current_loss < best_loss and not math.isinf(current_loss):
            best_loss = current_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss if val_dataloader else None,
            }, 'best_model.pth')
        
        # Generate samples every 10 epochs using EMA model
        if (epoch + 1) % 1 == 0:
            try:
                generate_samples_ema(ema_model, train_dataloader, epoch + 1)
            except Exception as e:
                print(f"Error generating samples: {e}")
    
    return train_losses, val_losses, ema_model'''
def train_rectified_flow(model, dataloader, num_epochs, lr=1e-4, scheduler_type='onecycle'):
    rectified_flow = RectifiedFlow(model, device)
    
    # Improved optimizer with better defaults
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=1e-4,  # Higher weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Multiple scheduler options
    if scheduler_type == 'onecycle':
        # OneCycle: Often best for diffusion models
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr= 5*lr,  # Peak at 5x base LR
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
     # ema_model = torch.optim.swa_utils.AveragedModel(model)
    
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
            }, 'best_model.pth')
        
        # Generate samples every 5 epochs using EMA model
        if (epoch + 1) % 1 == 0:
            generate_samples_ema(ema_model, dataloader, epoch + 1)
    
    return losses, 0,ema_model

def generate_samples_ema(ema_model, dataloader, epoch):
    """Generate and visualize samples using EMA model"""
    rectified_flow = RectifiedFlow(ema_model.module, device)
    ema_model.eval()
    
    # Get a batch of source images
    source, target = next(iter(dataloader))
    source, target = source[:8].to(device), target[:8].to(device)  # Take first 8 samples
    
    # Generate samples with different step counts
    with torch.no_grad():
        generated_25 = rectified_flow.sample(source, num_steps=25)
        generated_50 = rectified_flow.sample(source, num_steps=100)
    
    # Denormalize for visualization
    source_vis = (source + 1) / 2
    target_vis = (target + 1) / 2
    generated_25_vis = torch.clamp((generated_25 + 1) / 2, 0, 1)
    generated_50_vis = torch.clamp((generated_50 + 1) / 2, 0, 1)
    
    # Plot comparison
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    
    for i in range(8):
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
    plt.savefig(f'samples_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.show()

def generate_samples(model, dataloader, epoch):
    """Backward compatibility function"""
    generate_samples_ema(model, dataloader, epoch)

# Main training script
def main():
    # Set your dataset path here
    BASE_DIR = "/home/nbaier/.cache/kagglehub/datasets/balraj98/edges2shoes-dataset/versions/1/"
    BASE_DIR = "/pfs/work9/workspace/scratch/ka_uhicv-blah/latent_diffusion/edges2shoes-dataset/versions/1"
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
    
    # Create datasets
    train_dataset = Edges2ShoesDataset(BASE_DIR, split='train', image_size=(256, 256))
    val_dataset = None
    if os.path.exists(val_dir):
        val_dataset = Edges2ShoesDataset(BASE_DIR, split='val', image_size=(256, 256))
        print(f"Validation dataset created with {len(val_dataset)} samples")
    else:
        print("Validation directory not found, training without validation")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=64,  # Further reduced batch size for stability
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False
        )
    
    # Initialize model with better weight initialization
    model = RectifiedFlowUNet(
        in_channels=2, 
        out_channels=1, 
        time_embedding_dim=256,
       # base_channels=64  # Reduced base channels for more stability
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model with more conservative settings
    print("Starting training...")
    train_losses, val_losses, ema_model = train_rectified_flow(
        model, 
        train_dataloader,
        #val_dataloader,
        num_epochs=50,  # Increased epochs with lower LR
        lr=2e-5,  # Lower learning rate
        scheduler_type='onecycle'  # Most stable scheduler
    )
    
    # Plot training loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
if __name__ == "__main__":
    main()
