import torch
import os
import glob
from tqdm import tqdm
import numpy as np
from model import OptimizedRectifiedFlowUNet, RectifiedFlow

def load_best_ema_model(checkpoint_path='best_model_1500x1280.pth', device='cuda'):
    """
    Load the best trained rectified flow EMA model from checkpoint.
    
    Args:
        checkpoint_path: Path to the saved model checkpoint
        device: Device to load the model on
    
    Returns:
        Loaded EMA model ready for inference
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with same architecture as training
    base_model = OptimizedRectifiedFlowUNet(in_channels=2, out_channels=1).to(device)
    
    # Load EMA model state (preferred for inference)
    if 'ema_state_dict' in checkpoint:
        # Create EMA wrapper and load EMA weights
        ema_model = torch.optim.swa_utils.AveragedModel(
            base_model, 
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay=0.8)
        )
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        print("✓ Loaded EMA model from checkpoint (recommended for inference)")
        print(f"✓ Model trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"✓ Best training loss: {checkpoint.get('loss', 'unknown'):.4f}")
        return ema_model
    else:
        # Fallback to regular model weights
        base_model.load_state_dict(checkpoint['model_state_dict'])
        print("⚠ Loaded regular model from checkpoint (EMA not available)")
        return base_model

def load_hidden_state(pth_path, device):
    """
    Load hidden state from .pth file.
    
    Args:
        pth_path: Path to the .pth file containing hidden states
        device: Device to load the tensor on
    
    Returns:
        Loaded tensor
    """
    try:
        hidden_state = torch.load(pth_path, map_location=device)
        
        # Handle different possible formats
        if isinstance(hidden_state, dict):
            # If it's a dictionary, look for common keys
            if 'hidden_state' in hidden_state:
                hidden_state = hidden_state['hidden_state']
            elif 'tensor' in hidden_state:
                hidden_state = hidden_state['tensor']
            elif 'data' in hidden_state:
                hidden_state = hidden_state['data']
            else:
                # Take the first tensor value if it's a dict
                hidden_state = next(iter(hidden_state.values()))
        
        # Ensure it's a tensor
        if not isinstance(hidden_state, torch.Tensor):
            raise ValueError(f"Expected tensor, got {type(hidden_state)}")
        
        return hidden_state
    
    except Exception as e:
        raise RuntimeError(f"Failed to load hidden state from {pth_path}: {e}")

def prepare_hidden_state_for_model(hidden_state, target_channels=1, target_size=(1500, 1280)):
    """
    Prepare hidden state tensor for the rectified flow model.
    
    Args:
        hidden_state: Input hidden state tensor
        target_channels: Expected number of input channels for the model
        target_size: Expected spatial dimensions (height, width)
    
    Returns:
        Processed tensor ready for the model
    """
    import torch.nn.functional as F
    print(f"PREP DEBUG - Input shape: {hidden_state.shape}")
    print(f"PREP DEBUG - Input type: {type(hidden_state)}")
    # Add batch dimension if needed
    if len(hidden_state.shape) == 3:  # [C, H, W]
        hidden_state = hidden_state.unsqueeze(0)  # [1, C, H, W]
    
    print(f"Original hidden state shape: {hidden_state.shape}")
    
    # Handle channel dimension
    if hidden_state.shape[1] == 4 and target_channels == 2:
        # Option 1: Take first 2 channels
        hidden_state = hidden_state[:, :2, :, :]
        print(f"Reduced 4 channels to 2 by taking first 2 channels: {hidden_state.shape}")
        
        # Alternative options (uncomment to try):
        # Option 2: Average pairs of channels
        # channel1 = hidden_state[:, :2, :, :].mean(dim=1, keepdim=True)
        # channel2 = hidden_state[:, 2:, :, :].mean(dim=1, keepdim=True)
        # hidden_state = torch.cat([channel1, channel2], dim=1)
        # print(f"Reduced 4 channels to 2 by averaging pairs: {hidden_state.shape}")
        
    elif hidden_state.shape[1] == 1 and target_channels == 2:
        # If we have 1 channel but need 2, duplicate the channel
        hidden_state = hidden_state.repeat(1, 2, 1, 1)
        print(f"Duplicated single channel to match model input channels: {hidden_state.shape}")
        
    elif hidden_state.shape[1] != target_channels:
        print(f"Warning: Input has {hidden_state.shape[1]} channels, model expects {target_channels}")
        if hidden_state.shape[1] > target_channels:
            hidden_state = hidden_state[:, :target_channels, :, :]
            print(f"Truncated to {target_channels} channels: {hidden_state.shape}")
        else:
            # Pad with zeros or duplicate last channel
            missing_channels = target_channels - hidden_state.shape[1]
            padding = hidden_state[:, -1:, :, :].repeat(1, missing_channels, 1, 1)
            hidden_state = torch.cat([hidden_state, padding], dim=1)
            print(f"Padded to {target_channels} channels: {hidden_state.shape}")
    
    # Handle spatial dimensions
    current_h, current_w = hidden_state.shape[2], hidden_state.shape[3]
    target_h, target_w = target_size
    
    if current_h != target_h or current_w != target_w:
        print(f"Resizing from {current_h}x{current_w} to {target_h}x{target_w}")
        hidden_state = F.interpolate(
            hidden_state, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
    
    print(f"Final prepared hidden state shape: {hidden_state.shape}")
    return hidden_state

def save_hidden_state(tensor, output_path):
    """
    Save tensor as .pth file.
    
    Args:
        tensor: Tensor to save
        output_path: Output file path
    """
    # Remove batch dimension for saving
    if len(tensor.shape) == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Save as .pth file
    torch.save(tensor.cpu(), output_path)

def generate_diffusion_enhanced_hidden_states(
    input_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/test_latent',
    output_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/diffusion_enhanced_test/',
    checkpoint_path= '8_identity8_test_identity_50mapping.pth',

    num_steps=100,
    device=None
):
    """
    Generate diffusion-enhanced versions of all hidden states in the input directory.
    
    Args:
        input_dir: Directory containing input .pth files with hidden states
        output_dir: Directory to save enhanced hidden states
        checkpoint_path: Path to the trained model checkpoint
        num_steps: Number of diffusion steps for generation (100 recommended for quality)
        device: Device to run inference on
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load the EMA model (recommended for inference)
    print("\n" + "="*50)
    print("LOADING MODEL")
    print("="*50)
    try:
        model = load_best_ema_model(checkpoint_path, device)
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
    
    # Find all .pth files
    print("\n" + "="*50)
    print("FINDING INPUT FILES")
    print("="*50)
    
    pth_files = []
    pattern = os.path.join(input_dir, '*.pth')
    pth_files.extend(glob.glob(pattern))
    
    # Also check subdirectories
    pattern = os.path.join(input_dir, '**', '*.pth')
    pth_files.extend(glob.glob(pattern, recursive=True))
    
    pth_files = sorted(list(set(pth_files)))  # Remove duplicates and sort
    
    if len(pth_files) == 0:
        print(f"❌ No .pth files found in {input_dir}")
        return
    
    print(f"✓ Found {len(pth_files)} .pth files to process")
    
    # Process each hidden state
    print("\n" + "="*50)
    print("PROCESSING HIDDEN STATES")
    print("="*50)
    
    successful = 0
    failed = 0
    with torch.no_grad():
        for pth_path in tqdm(pth_files, desc="Enhancing hidden states"):
            try:
                # Clear GPU cache
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Get relative path and create output path
                rel_path = os.path.relpath(pth_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Create output subdirectory if needed
                output_subdir = os.path.dirname(output_path)
                if output_subdir:
                    os.makedirs(output_subdir, exist_ok=True)
                
                # Skip if output already exists
                if os.path.exists(output_path):
                    print(f"⏭ Skipping {rel_path} - output already exists")
                    continue
                
                # Load hidden state
                hidden_state = load_hidden_state(pth_path, device)
                print(f"DEBUG - Raw loaded shape: {hidden_state.shape}") 
                # Prepare for model input
                input_tensor = prepare_hidden_state_for_model(hidden_state)
                
                # Generate enhanced version using EMA model
                # Using 100 steps for high quality (as in your generate_samples_ema function)
                enhanced_tensor = rectified_flow.sample(input_tensor, num_steps=num_steps)
                
                # Save enhanced hidden state
                save_hidden_state(enhanced_tensor, output_path)
                
                successful += 1
                
                # Clean up tensors
                del hidden_state, input_tensor, enhanced_tensor
                
            except Exception as e:
                print(f"❌ Failed to process {pth_path}: {e}")
                failed += 1
                continue
    
    print("\n" + "="*50)
    print("ENHANCEMENT COMPLETE")
    print("="*50)
    print(f"✓ Successfully processed: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Enhanced hidden states saved to: {output_dir}")
    
    if device.type == 'cuda':
        print(f"🔥 Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

def batch_enhance_hidden_states_with_memory_management(
    input_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/test_latent',
    output_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/diffusion_enhanced_test/',
    checkpoint_path='best_model_1500x1280.pth',
    num_steps=100,
    device=None,
    max_memory_gb=10.0
):
    """
    Memory-efficient version with monitoring and automatic cleanup.
    
    Args:
        input_dir: Directory containing input .pth files
        output_dir: Directory to save enhanced hidden states
        checkpoint_path: Path to the trained model checkpoint
        num_steps: Number of diffusion steps (100 for quality, 25 for speed)
        device: Device to run inference on
        max_memory_gb: Maximum GPU memory to use before forcing cleanup
    """
    
    def check_memory():
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1e9
            if memory_used > max_memory_gb:
                print(f"⚠ High memory usage: {memory_used:.1f} GB, forcing cleanup...")
                torch.cuda.empty_cache()
                return torch.cuda.memory_allocated() / 1e9
        return 0
    
    # Use the main function with memory monitoring
    print(f"Starting batch enhancement with memory limit: {max_memory_gb} GB")
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run main enhancement function
    generate_diffusion_enhanced_hidden_states(
        input_dir=input_dir,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        num_steps=num_steps,
        device=device
    )

if __name__ == "__main__":
    # Example usage with your specified paths
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_diffusion_enhanced_hidden_states(
        input_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/test_latent',
        output_dir='/pfs/work9/workspace/scratch/ka_uhicv-blah/hidden_states_latent_diffusion/diffusion_enhanced_test/',
        checkpoint_path='8_identity8_test_identity_50mapping.pth',

        num_steps=100,  # High quality generation
        device=device
    )
