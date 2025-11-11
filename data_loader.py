import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split

def load_input_data(model_name):
    """
    Mocks loading data from an NPZ file. 
    In your final setup, this should load and return the numpy arrays.
    """
    
    # Load the actual data as in your original script
    data = np.load(f'/Users/fahimeh/Projects/Data_science_prep/Recipe_Recommender_System/input_data/{model_name}_vae_input_data.npz')
    text_embeds = data['text_embeds']
    numeric_scaled = data['numeric_scaled']
    print("Data loaded successfully")
    return numeric_scaled, text_embeds
   


def get_dataloaders(text_embeds, numeric_scaled, batch_size=256, splits=(0.70, 0.15, 0.15)):
    """
    Combines numeric and text data, performs train/val/test split, and
    creates PyTorch DataLoaders. It now also returns the indices used for each split.

    Args:
        text_embeds (np.ndarray): The 768-dim text embeddings.
        numeric_scaled (np.ndarray): The 20-dim numeric features.
        batch_size (int): The size of the mini-batches for the DataLoader.
        splits (tuple): Ratios for (train, validation, test) split.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_indices, val_indices, test_indices)
    """
    if sum(splits) != 1.0:
        raise ValueError("The sum of the splits tuple must be 1.0")

    # 1. Convert numpy arrays to PyTorch Tensors
    numeric_tensor = torch.tensor(numeric_scaled, dtype=torch.float32)
    text_tensor = torch.tensor(text_embeds, dtype=torch.float32)

    # 2. Create the combined dataset
    full_dataset = TensorDataset(numeric_tensor, text_tensor)

    # 3. Calculate lengths for the split
    total_size = len(full_dataset)
    train_size = int(splits[0] * total_size)
    val_size = int(splits[1] * total_size)
    test_size = total_size - train_size - val_size # Use remaining to ensure exact match

    print(f"Total dataset size: {total_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    # 4. Perform the random split
    generator = torch.Generator().manual_seed(42) # Use a fixed seed for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )

    # 5. Extract indices from the Subsets
    # This is the crucial step to link back to the original full dataset index
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_indices, val_indices, test_indices