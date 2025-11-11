import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd # Added for save_loss

def save_checkpoint(model, optimizer, epoch, filename='vae_checkpoint', directory='output_checkpoints'):
    """
    Saves the model's state, optimizer's state, and training metadata.

    Args:
        model (nn.Module): The VAE model instance.
        optimizer (torch.optim.Optimizer): The optimizer instance.
        epoch (int): The current epoch number.
        filename (str): Base name for the checkpoint file.
        directory (str): Directory to save the checkpoint.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, f'{filename}_epoch_{epoch+1}.pt')
    
    # Save a map of the model's structure details for easy loading
    model_details = {
        'numeric_dim': model.decoder.numeric_decoder[2].out_features, 
        'text_dim': model.decoder.text_decoder[4].out_features, # Corrected index for new deeper decoder
        'latent_dim': model.encoder.latent_dim, 
    }
    
    # Create the state dictionary
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_details': model_details
    }
    
    torch.save(state, filepath)
    print(f"\nCheckpoint saved to {filepath}")


def load_checkpoint(filepath, model, optimizer=None):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        filepath (str): Full path to the checkpoint file.
        model (nn.Module): The VAE model instance to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer instance to load state into.
    """
    if not os.path.exists(filepath):
        print(f"Error: Checkpoint file not found at {filepath}")
        return None, None
        
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded successfully from epoch {epoch + 1}.")
    return model, optimizer, epoch


def save_latent_embeddings(model, data_loader, device, filename='latent_embeddings', directory='output_embeds'):
    """
    Runs inference on a dataset and saves all z, mu, and logvar vectors.

    Args:
        model (nn.Module): The trained VAE model instance.
        data_loader (DataLoader): The DataLoader (typically for the test set).
        device (str): The device to run inference on.
        filename (str): Base name for the output files.
        directory (str): Directory to save the .npy files.
    """
    model.eval()
    all_mu = []
    all_logvar = []
    all_latent_vectors = []
    
    # FIX: Use len(data_loader.dataset) for Subsets
    print(f"Starting inference to save latent embeddings on {len(data_loader.dataset)} samples...") 

    with torch.no_grad():
        for batch_numeric, batch_text in data_loader:
            batch_numeric = batch_numeric.to(device)
            batch_text = batch_text.to(device)
            
            # Only need the encoder output (z, mu, logvar)
            latent_vectors, mu, logvar = model.encoder(batch_numeric, batch_text)
            
            all_latent_vectors.append(latent_vectors.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_logvar.append(logvar.cpu().numpy())

    # Concatenate all batches into single numpy arrays
    final_latent_vectors = np.concatenate(all_latent_vectors, axis=0)
    final_mu = np.concatenate(all_mu, axis=0)
    final_logvar = np.concatenate(all_logvar, axis=0)

    # Save to disk
    os.makedirs(directory, exist_ok=True)
    latent_vectors_filepath = os.path.join(directory, f'{filename}_latent_vectors.npy')
    mu_filepath = os.path.join(directory, f'{filename}_mu.npy')
    logvar_filepath = os.path.join(directory, f'{filename}_logvar.npy')

    np.save(latent_vectors_filepath, final_latent_vectors)
    np.save(mu_filepath, final_mu)
    np.save(logvar_filepath, final_logvar)

    print(f"\nLatent vectors (z) saved: {latent_vectors_filepath} with shape {final_latent_vectors.shape}")
    print(f"Latent means (mu) saved: {mu_filepath} with shape {final_mu.shape}")
    print(f"Latent logvar saved: {logvar_filepath} with shape {final_logvar.shape}")
    
    return latent_vectors_filepath, mu_filepath, logvar_filepath
    
def save_loss(loss_history, directory, filename):
    """
    Saves the loss history (a list of dictionaries) to a CSV file.

    Args:
        loss_history (list): List of dictionaries containing loss values per epoch.
        directory (str): Directory to save the CSV file.
        filename (str): Name of the CSV file.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    
    df = pd.DataFrame(loss_history)
    df.to_csv(filepath, index=False)
    print(f"\nLoss history saved to {filepath}")

