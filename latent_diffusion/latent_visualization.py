import matplotlib.pyplot as plt
import numpy as np
from model import RectifiedFlow
import torch
import wandb
def visualize_whisper_batch(clean_audio, prediction, save_path='visualization.png', 
                          vmin=-1, vmax=1, clamp_values=True):
    """
    Visualize Whisper hidden states as heatmaps.
    
    Args:
        clean_audio: tensor/array of shape [batch, channels, 1500, 1280] or [1500, 1280]
        prediction: tensor/array of shape [batch, channels, 1500, 1280] or [1500, 1280]
        save_path: string - path to save the visualization
        vmin: float - minimum value for color scale (default: 0)
        vmax: float - maximum value for color scale (default: 1)
        clamp_values: bool - whether to clamp values outside [vmin, vmax] range (default: True)
    """
    
    # Convert tensors to numpy and squeeze extra dimensions
    def process_tensor(tensor):
        if hasattr(tensor, 'cpu'):
            tensor = tensor.cpu().numpy()
        # Remove batch and channel dimensions if present
        if tensor.ndim == 4:  # [batch, channel, height, width]
            tensor = tensor.squeeze()  # Remove dimensions of size 1
        elif tensor.ndim == 3:  # [batch, height, width] or [channel, height, width]
            tensor = tensor.squeeze()
        return tensor
    
    clean_audio = process_tensor(clean_audio)
    prediction = process_tensor(prediction)
    
    # Clamp values if requested
    if clamp_values:
        clean_audio = np.clip(clean_audio, vmin, vmax)
        prediction = np.clip(prediction, vmin, vmax)
    
    print(f"Processed shapes - Clean: {clean_audio.shape}, Prediction: {prediction.shape}")
    print(f"Value ranges - Clean: [{clean_audio.min():.3f}, {clean_audio.max():.3f}], "
          f"Prediction: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    # Create figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use provided vmin/vmax for color scaling
    print(f"Using color scale: [{vmin}, {vmax}]")
    
    # Plot clean audio (target) - transpose for better visualization
    im1 = axes[0].imshow(clean_audio.T, aspect='auto', cmap='viridis', 
                         vmin=vmin, vmax=vmax, origin='lower')
    axes[0].set_title('Clean - Target Hidden state', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Hidden Dimensions')
    
    # Plot prediction
    im2 = axes[1].imshow(prediction.T, aspect='auto', cmap='viridis', 
                         vmin=vmin, vmax=vmax, origin='lower')
    axes[1].set_title('Model Prediction for difference', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Hidden Dimensions')
    
    # Adjust layout to make room for colorbar
    plt.subplots_adjust(bottom=0.2)
    
    # Add colorbar below the plots
    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Hidden State Values', fontsize=12)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved to {save_path}")

# Example usage with synthetic data
if __name__ == "__main__":
    # Generate example data with the same shape as your Whisper hidden states
    np.random.seed(42)
    
    # Create synthetic hidden states [1500, 1280]
    time_steps, hidden_dim = 1500, 1280
    
    # Clean audio - smoother patterns around 0
    clean = np.random.randn(time_steps, hidden_dim) * 0.2
    for i in range(time_steps):
        for j in range(hidden_dim):
            clean[i, j] += np.sin(i/200) * np.cos(j/300) * np.exp(-i/1000) * 0.3
    
    # Noisy audio - more random noise around 0
    noisy = clean + np.random.randn(time_steps, hidden_dim) * 0.3
    
    # Prediction - somewhere between clean and noisy
    prediction = clean + np.random.randn(time_steps, hidden_dim) * 0.15
    
    # Visualize with default parameters (0-1 range, clamped)
    visualize_whisper_batch(clean, prediction, save_path='visualization_default.png')
    
    # Visualize with custom parameters
    visualize_whisper_batch(clean, prediction, 
                          save_path='visualization_custom.png',
                          vmin=-0.5, vmax=0.8, clamp_values=True)
    
    # Visualize without clamping to see full range
    visualize_whisper_batch(clean, prediction, 
                          save_path='visualization_unclamped.png',
                          vmin=-2, vmax=2, clamp_values=False)

# For your actual dataloader usage:
# batch = next(iter(your_dataloader))
# source_audio = batch['source']  # Shape: [1, 1, 1500, 1280]
# target_audio = batch['target']  # Shape: [1, 1, 1500, 1280]
# 
# # Get model prediction
# with torch.no_grad():
#     prediction = your_model(source_audio)
# 
# # Visualize with default 0-1 range and clamping
# visualize_whisper_batch(target_audio, prediction)
#
# # Or with custom parameters for better contrast
# visualize_whisper_batch(target_audio, prediction,
#                        vmin=0, vmax=0.5, clamp_values=True, 
#                        save_path='whisper_visualization.png')

def generate_validation_samples_ema(ema_model, dataloader, epoch):
    """Generate and visualize samples using EMA model for high resolution"""
    device = "cuda"
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
    
    # Plot comparison - CHAZZNGED: Adjusted for high resolution display
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
    device = "cuda"
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


