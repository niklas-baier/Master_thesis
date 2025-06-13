import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
from pathlib import Path
import re
from latent_visualization import visualize_whisper_batch 

class AudioPairDataset(Dataset):
    def __init__(self, root_dir, shuffle_pairs=True):
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
        self.mic_dirs = [self.root_dir / f"mic{i}" for i in range(1, 6)]
        
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
        pattern = r'(\d+)P\.pth$'  # Fixed the regex pattern
        
        for file_path in self.person_files:
            match = re.search(pattern, file_path.name)
            if match:
                indices.append(int(match.group(1)))
            else:
                print(f"Warning: Skipping file with unexpected format: {file_path.name}")
        
        return sorted(indices)
    
    def _create_all_valid_pairs(self):
        """Create all valid (person_idx, mic_num) combinations"""
        valid_pairs = []
        
        for person_idx in self.indices:
            for mic_num in range(1, 6):
                mic_file = self.mic_dirs[mic_num - 1] / f"{person_idx}M{mic_num}.pth"
                if mic_file.exists():
                    valid_pairs.append((person_idx, mic_num))
                    
        if not valid_pairs:
            raise FileNotFoundError("No valid (person, mic) pairs found")
            
        print(f"Valid pairs per person (example for first few):")
        # Show distribution for first few indices
        for idx in sorted(self.indices)[:3]:
            mics = [mic_num for person_idx, mic_num in valid_pairs if person_idx == idx]
            print(f"  Person {idx}: mics {mics}")
        
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
        """
        Get a specific (person, mic) pair from the current epoch's shuffled list
        Returns data in format suitable for rectified flow training
        """
        # Get the specific pair for this index
        person_idx, mic_num = self.current_epoch_pairs[idx]
        
        # Load clean audio (person) - this will be the target
        person_file = self.persons_dir / f"{person_idx}P.pth"
        clean_audio = torch.load(person_file, map_location='cpu')
        
        # Load noisy audio (mic) - this will be the source
        mic_file = self.mic_dirs[mic_num - 1] / f"{person_idx}M{mic_num}.pth"
        noisy_audio = torch.load(mic_file, map_location='cpu')
        
        return {
            'source': noisy_audio,    # Source domain (noisy mic audio)
            'target': clean_audio,    # Target domain (clean person audio)
            'clean': clean_audio,     # For backward compatibility
            'noisy': noisy_audio,     # For backward compatibility
            'index': person_idx,
            'mic_used': mic_num
        }


def create_audio_dataloader(root_dir, batch_size=32, shuffle=False, num_workers=4, 
                           shuffle_pairs=True, **kwargs):
    """
    Create a DataLoader for all possible audio pairs suitable for rectified flow training.
    
    Args:
        root_dir (str): Root directory containing audio folders
        batch_size (int): Batch size
        shuffle (bool): DataLoader shuffle (keep False since we handle shuffling in dataset)
        num_workers (int): Number of worker processes
        shuffle_pairs (bool): Whether to shuffle pair order each epoch
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader
    """
    dataset = AudioPairDataset(
        root_dir=root_dir,
        shuffle_pairs=shuffle_pairs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Usually False since we handle shuffling in dataset
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
    # If shapes differ, you might need padding or other handling
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


# Example usage for rectified flow training
if __name__ == "__main__":
    # Example usage
    root_directory = "/home/ka/ka_stud/ka_uhicv"
    
    # Create dataloader for rectified flow training
    # This will create ALL possible (person, mic) combinations
    train_loader = create_audio_dataloader(
        root_dir=root_directory,
        batch_size=1,
        shuffle=False,  # We handle shuffling in the dataset
        num_workers=4,
        shuffle_pairs=True,  # Shuffle the order of pairs each epoch
        collate_fn=collate_fn
    )
    
    print(f"Dataset size: {len(train_loader.dataset)} pairs")
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Training loop example for rectified flow
    for epoch in range(2):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*50}")
        
        # Reshuffle pair order for new epoch
        train_loader.dataset.on_epoch_start()
        
        # Track what we see this epoch
        seen_pairs = set()
        mic_counts = {i: 0 for i in range(1, 6)}
        
        for batch_idx, batch in enumerate(train_loader):
            source_audio = batch['source']  # Noisy audio (mic recordings)
            target_audio = batch['target']  # Clean audio (person recordings)
            indices = batch['indices']      # Audio file indices
            mics_used = batch['mics_used']  # Mic numbers used
            
            # Track pairs seen
            for idx, mic in zip(indices, mics_used):
                seen_pairs.add((idx, mic))
                mic_counts[mic] += 1
            
            if batch_idx < 3:  # Show first few batches
                print(f"Batch {batch_idx}: Source shape: {source_audio.shape}, "
                      f"Target shape: {target_audio.shape}")
                print(f"Pairs in batch: {list(zip(indices, mics_used))}")
                clean_audio = batch['target']
                noisy_audio = batch['source']
                prediction = (clean_audio + noisy_audio) / 2
                visualize_whisper_batch(clean_audio=clean_audio-noisy_audio, prediction= prediction-noisy_audio, save_path='visualization.png')
                breakpoint()
            print(batch_idx)
            
            # Rectified flow training code would go here:
            # 1. Sample random time t ~ U(0,1)
            # 2. Interpolate between source and target: x_t = (1-t) * source + t * target
            # 3. Compute velocity field: v_t = target - source
            # 4. Train model to predict v_t given x_t and t
            
        
        print(f"\nAfter {batch_idx + 1} batches:")
        print(f"Unique pairs seen: {len(seen_pairs)}")
        print(f"Mic distribution: {mic_counts}")
    
    print(f"\nPerfect! Each epoch iterates through all {len(train_loader.dataset)} pairs exactly once.")
    print("The pairs are shuffled each epoch to avoid pattern bias during training.")
