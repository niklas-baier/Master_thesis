import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import RectifiedFlow

def comprehensive_debug_rectified_flow(model, test_dataloader, device, save_dir="debug_results"):
    """
    Comprehensive debugging of rectified flow model performance
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    rectified_flow = RectifiedFlow(model, device)
    
    print("=" * 60)
    print("RECTIFIED FLOW DEBUGGING")
    print("=" * 60)
    
    # Get a single batch for detailed analysis
    batch = next(iter(test_dataloader))
    source = batch['source'][:4].to(device)  # noisy audio
    target = batch['target'][:4].to(device)  # clean audio
    
    print(f"Batch shapes - Source: {source.shape}, Target: {target.shape}")
    print(f"Source range: [{source.min():.4f}, {source.max():.4f}]")
    print(f"Target range: [{target.min():.4f}, {target.max():.4f}]")
    
    # 1. Test different sampling steps
    print("\n1. TESTING DIFFERENT SAMPLING STEPS")
    print("-" * 40)
    
    step_counts = [5, 10, 25, 50, 100, 200]
    mse_results = {}
    
    baseline_mse = F.mse_loss(source, target).item()
    print(f"Baseline MSE (noisy vs target): {baseline_mse:.6f}")
    
    with torch.no_grad():
        for steps in step_counts:
            sampled = rectified_flow.sample(source, num_steps=steps)
            mse = F.mse_loss(sampled, target).item()
            mse_results[steps] = mse
            improvement = (baseline_mse - mse) / baseline_mse * 100
            print(f"Steps {steps:3d}: MSE = {mse:.6f}, Improvement = {improvement:+.2f}%")
    
    # 2. Check velocity field behavior
    print("\n2. VELOCITY FIELD ANALYSIS")
    print("-" * 40)
    
    with torch.no_grad():
        # Test velocity field at different t values
        t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        for t in t_values:
            # Create interpolated input
            t_tensor = torch.full((source.shape[0],), t, device=device)
            interpolated = t * target + (1 - t) * source
            
            # Get velocity prediction
            velocity = model(interpolated)
            true_velocity = target - source  # True velocity for rectified flow
            
            velocity_mse = F.mse_loss(velocity, true_velocity).item()
            velocity_norm = velocity.norm().item()
            true_velocity_norm = true_velocity.norm().item()
            
            print(f"t={t:.2f}: Velocity MSE={velocity_mse:.6f}, "
                  f"Pred norm={velocity_norm:.4f}, True norm={true_velocity_norm:.4f}")
    
    # 3. Test ODE integration manually
    print("\n3. MANUAL ODE INTEGRATION TEST")
    print("-" * 40)
    
    def manual_euler_step(x, t, dt, model):
        """Manual Euler step for debugging"""
        with torch.no_grad():
            velocity = model(x)
            return x + dt * velocity
    
    # Manual integration with different step sizes
    manual_steps = [10, 50, 100]
    for num_steps in manual_steps:
        dt = 1.0 / num_steps
        x = source.clone()
        
        for i in range(num_steps):
            x = manual_euler_step(x, i * dt, dt, model)
        
        manual_mse = F.mse_loss(x, target).item()
        improvement = (baseline_mse - manual_mse) / baseline_mse * 100
        print(f"Manual {num_steps:3d} steps: MSE = {manual_mse:.6f}, Improvement = {improvement:+.2f}%")
    
    # 4. Check for training issues
    print("\n4. TRAINING DIAGNOSTICS")
    print("-" * 40)
    
    model.train()  # Switch to train mode to check gradients
    with torch.enable_grad():
        # Simulate one training step
        t = torch.rand(source.shape[0], device=device)
        interpolated = t.view(-1, 1, 1) * target + (1 - t.view(-1, 1, 1)) * source
        velocity_pred = model(interpolated)
        velocity_true = target - source
        
        loss = F.mse_loss(velocity_pred, velocity_true)
        loss.backward()
        
        # Check gradient norms
        total_grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += param_grad_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"Training loss: {loss.item():.6f}")
        print(f"Total gradient norm: {total_grad_norm:.6f}")
        print(f"Velocity pred range: [{velocity_pred.min():.4f}, {velocity_pred.max():.4f}]")
        print(f"Velocity true range: [{velocity_true.min():.4f}, {velocity_true.max():.4f}]")
    
    model.eval()
    
    # 5. Check data quality
    print("\n5. DATA QUALITY CHECK")
    print("-" * 40)
    
    # Check if source and target are actually different
    data_diff = (source - target).abs().mean().item()
    print(f"Average absolute difference between source and target: {data_diff:.6f}")
    
    if data_diff < 1e-6:
        print("WARNING: Source and target are nearly identical!")
    
    # Check signal-to-noise ratio
    signal_power = target.pow(2).mean().item()
    noise_power = (source - target).pow(2).mean().item()
    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    print(f"Signal-to-noise ratio: {snr_db:.2f} dB")
    
    # 6. Visualize results
    print("\n6. GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot MSE vs steps
    axes[0, 0].plot(step_counts, [mse_results[s] for s in step_counts], 'b-o', label='Rectified Flow')
    axes[0, 0].axhline(y=baseline_mse, color='r', linestyle='--', label='Baseline (Noisy)')
    axes[0, 0].set_xlabel('Sampling Steps')
    axes[0, 0].set_ylabel('MSE')
    axes[0, 0].set_title('MSE vs Sampling Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot improvement percentage
    improvements = [(baseline_mse - mse_results[s]) / baseline_mse * 100 for s in step_counts]
    axes[0, 1].plot(step_counts, improvements, 'g-o')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Sampling Steps')
    axes[0, 1].set_ylabel('Improvement (%)')
    axes[0, 1].set_title('Performance Improvement')
    axes[0, 1].grid(True)
    
    # Plot sample waveforms (assuming 1D audio)
    if len(source.shape) == 2:  # [batch, length]
        sample_idx = 0
        axes[1, 0].plot(source[sample_idx].cpu().numpy(), label='Noisy', alpha=0.7)
        axes[1, 0].plot(target[sample_idx].cpu().numpy(), label='Clean', alpha=0.7)
        
        # Get best sampling result
        best_steps = max(step_counts, key=lambda s: (baseline_mse - mse_results[s]))
        with torch.no_grad():
            best_result = rectified_flow.sample(source[sample_idx:sample_idx+1], num_steps=best_steps)
        axes[1, 0].plot(best_result[0].cpu().numpy(), label=f'Rectified Flow ({best_steps} steps)', alpha=0.7)
        axes[1, 0].set_title('Sample Waveforms')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot velocity field analysis
    axes[1, 1].text(0.1, 0.5, f'Training Loss: {loss.item():.6f}\n'
                                f'Gradient Norm: {total_grad_norm:.6f}\n'
                                f'Data Diff: {data_diff:.6f}\n'
                                f'SNR: {snr_db:.2f} dB', 
                   transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='center')
    axes[1, 1].set_title('Training Diagnostics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/rectified_flow_debug.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 7. Summary and recommendations
    print("\n7. SUMMARY AND RECOMMENDATIONS")
    print("-" * 40)
    
    best_mse = min(mse_results.values())
    best_steps = min(step_counts, key=lambda s: mse_results[s])
    
    print(f"Best MSE achieved: {best_mse:.6f} (with {best_steps} steps)")
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    if best_mse >= baseline_mse:
        print("❌ PROBLEM: Model performs worse than baseline!")
        print("\nPossible issues:")
        print("- Model hasn't converged (check training loss)")
        print("- Wrong model architecture for data")
        print("- Data preprocessing issues")
        print("- Learning rate too high/low")
        print("- ODE integration numerical issues")
    else:
        improvement = (baseline_mse - best_mse) / baseline_mse * 100
        print(f"✅ Model shows {improvement:.2f}% improvement over baseline")
        
        if improvement < 5:
            print("⚠️  Warning: Improvement is small, consider:")
            print("- Longer training")
            print("- Better model architecture")
            print("- More sophisticated ODE solver")
    
    return mse_results, baseline_mse

# Quick diagnostic function
def quick_debug(model, test_dataloader, device):
    """Quick diagnostic to identify obvious issues"""
    model.eval()
    rectified_flow = RectifiedFlow(model, device)
    
    batch = next(iter(test_dataloader))
    source = batch['source'][:1].to(device)
    target = batch['target'][:1].to(device)
    
    print("QUICK DIAGNOSTIC:")
    print(f"Source shape: {source.shape}")
    print(f"Target shape: {target.shape}")
    
    # Test model forward pass
    try:
        with torch.no_grad():
            velocity = model(source)
            print(f"✅ Model forward pass successful: {velocity.shape}")
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        return
    
    # Test sampling
    try:
        with torch.no_grad():
            sampled = rectified_flow.sample(source, num_steps=10)
            print(f"✅ Sampling successful: {sampled.shape}")
    except Exception as e:
        print(f"❌ Sampling failed: {e}")
        return
    
    # Quick MSE check
    baseline_mse = F.mse_loss(source, target).item()
    sampled_mse = F.mse_loss(sampled, target).item()
    
    print(f"Baseline MSE: {baseline_mse:.6f}")
    print(f"Sampled MSE: {sampled_mse:.6f}")
    print(f"Improvement: {((baseline_mse - sampled_mse) / baseline_mse * 100):+.2f}%")

if __name__ == "__main__":
    # Example usage
    model = load_your_model()  # Replace with your model loading
    test_dataloader = create_test_dataloader()  # Replace with your dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run comprehensive debug
    mse_results, baseline = comprehensive_debug_rectified_flow(model, test_dataloader, device)
