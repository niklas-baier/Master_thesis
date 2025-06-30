import os
import torch
import math

def load_pth_file(file_path):
    """Load a .pth file and return its contents."""
    return torch.load(file_path, map_location=torch.device('cpu'))

def compute_mse(tensor1, tensor2):
    """Compute the Mean Squared Error between two tensors."""
    return torch.nn.functional.mse_loss(tensor1, tensor2).item()

def calculate_std_dev(values, mean):
    """Calculate the standard deviation of a list of values."""
    squared_diffs = [(x - mean) ** 2 for x in values]
    variance = sum(squared_diffs) / len(values)
    std_dev = math.sqrt(variance)
    return std_dev

def main():
     #mic1_dir = 'diffusion_enhanced_training'#'mic2'
    #persons_dir = 'persons'
    mic1_dir = 'diffusion_enhanced_training'#'mic2'
    persons_dir = 'mic1'

    #mic1_dir = 'diffusion_enhanced_test'#'mic2'
    #persons_dir = 'test'

    # Get the list of files in both directories
    mic1_files = os.listdir(mic1_dir)
    persons_files = os.listdir(persons_dir)

    # Extract the numbers from the filenames
    mic1_numbers = {int(file.split('M')[0]) for file in mic1_files if file.endswith('.pth')}
    persons_numbers = {int(file.split('M')[0]) for file in persons_files if file.endswith('.pth')}

    # Find the common numbers
    common_numbers = mic1_numbers.intersection(persons_numbers)

    # Compute MSE for each pair
    mse_values = []
    sum_mse = 0
    average = torch.zeros((1,1500,1280))
    for number in common_numbers:
        mic1_file = os.path.join(mic1_dir, f"{number}M1.pth")
        persons_file = os.path.join(persons_dir, f"{number}M1.pth")

        tensor1 = load_pth_file(mic1_file)
        tensor2 = load_pth_file(persons_file)
        difference = tensor2-tensor1
        abs_dif = torch.abs(difference)
        average = average + abs_dif
        mse = compute_mse(tensor1, tensor2)
        mse_values.append(mse)
        sum_mse += mse
        print(f"Number: {number}, MSE: {mse}")

    # Calculate average MSE
    avg_mse = sum_mse / len(mse_values)
    print(f'avg mse : {avg_mse}')
    average =  average / len(mse_values)
    import numpy as np
    np.random.seed(0)  # For reproducibility
    # Flatten to 1D
    import numpy as np
    from scipy.stats import kstest, norm

    data = average.flatten().numpy()

    # Normalize the data: mean 0, std 1 (standard normal for KS test)
    data = (data - np.mean(data)) / np.std(data)

    # Perform Kolmogorov-Smirnov test against the standard normal distribution
    statistic, p_value = kstest(data, 'norm')

    print(f"K-S statistic: {statistic}")
    print(f"P-value: {p_value}")

    # Interpretation
    alpha = 0.10  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: Data is likely not Gaussian.")
    else:
        print("Fail to reject the null hypothesis: Data may be Gaussian.")    # Flatten the tensor to a 1D array


        print(f'Average MSE: {avg_mse}')

        # Calculate standard deviation of MSE
    std_dev = calculate_std_dev(mse_values, avg_mse)
    print(f'Standard Deviation of MSE: {std_dev}')
    import numpy as np
    import matplotlib.pyplot as plt

    # Assuming your tensor is a numpy array
    tensor = np.random.rand(1, 1500, 1280)  # Example tensor, replace with your actual data

    # Squeeze the tensor to get a 2D array
    tensor_2d = tensor.squeeze()

    # Normalize the tensor to the range [0, 1] if it's not already
    tensor_normalized = (tensor_2d - tensor_2d.min()) / (tensor_2d.max() - tensor_2d.min())

    # Create the heatmap
    plt.figure(figsize=(10, 7.5))  # Adjust the figure size as needed
    plt.imshow(tensor_normalized, cmap='gray', vmin=0, vmax=1)
    plt.colorbar(label='Intensity')
    plt.title('Heatmap of Tensor')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Save the figure as an image file
    plt.savefig('average_change.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

