import os
import torch
import csv
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import time
import math

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)

from model import WAE, wae_loss # Import WAE and the updated wae_loss
# from loss_functions import wae_loss # Removed: wae_loss is now in vae_model.py
from data_loader import  load_input_data, get_dataloaders
from train_utils import save_loss, save_checkpoint, save_latent_embeddings


# --- GLOBAL CONFIGURATION ---
working_dir = "/Users/fahimeh/Projects/Data_science_prep/Recipe_Recommender_System/vae_with_kl_annealing/"
TEXT_EMBEDING_MODEL_NAME = "all-MiniLM-L6-v2" 
NUM_EPOCHS = 100 
BATCH_SIZE = 512 
LEARNING_RATE = 1e-3
LATENT_DIM = 32

# KL Annealing Schedule (Start easy, then regularize)
KL_ANNEAL_START_EPOCH = 5  # Start increasing KL weight after 5 epochs
KL_ANNEAL_END_EPOCH = 20   # KL weight reaches its final value at epoch 20


# --- HYPERPARAMETER EXPERIMENT CONFIGS (UPDATED) ---
# Testing higher WEIGHT_TEXT values, as you found they improve the latent space.
EXPERIMENT_CONFIGS = [
    {'wt': 5.0, 'wn': 50.0, 'kl': 0.10}, 
    {'wt': 5.0, 'wn': 10.0, 'kl': 0.10}, 
    {'wt': 5.0, 'wn': 5.0, 'kl': 0.10}, 
    {'wt': 10.0, 'wn': 50.0, 'kl': 0.10}, 
    {'wt': 10.0, 'wn': 10.0, 'kl': 0.10}, 
    {'wt': 10.0, 'wn': 5.0, 'kl': 0.10},
    {'wt': 50.0, 'wn': 5.0, 'kl': 0.10},
    {'wt': 5.0, 'wn': 5.0, 'kl': 0.05},
    {'wt': 5.0, 'wn': 5.0, 'kl': 0.5}
]

# --- KL ANNEALING FUNCTION (UNCHANGED) ---
def get_kl_weight(epoch, start_epoch, end_epoch, final_weight):
    """
    Calculates the current KL weight based on the epoch for linear annealing.
    """
    if epoch < start_epoch:
        return 0.0
    elif epoch >= end_epoch:
        return final_weight
    else:
        # Linear ramp up
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        return progress * final_weight
        
# --- 1. Data Preparation (Outside the loop, as it's constant) ---

numeric_scaled, text_embeds = load_input_data(TEXT_EMBEDING_MODEL_NAME)
numeric_dim = numeric_scaled.shape[1]
text_dim = text_embeds.shape[1]

# Get data loaders with train/val/test split and, crucially, the split indices
train_loader, val_loader, test_loader, train_indices, val_indices, test_indices = get_dataloaders(
    text_embeds, 
    numeric_scaled, 
    batch_size=BATCH_SIZE
)

# --- Save Test Indices (UNCHANGED) ---
INDEX_DIR = os.path.join(working_dir, 'output_indices')
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_FILE = os.path.join(INDEX_DIR, f'{TEXT_EMBEDING_MODEL_NAME}_test_indices.npy')
np.save(INDEX_FILE, np.array(test_indices))
print(f"Test dataset indices saved to {INDEX_FILE}")
# --- END INDEX SAVE ---

# --- MAIN EXPERIMENT LOOP ---
for i, config in enumerate(EXPERIMENT_CONFIGS):
    WEIGHT_TEXT = config['wt']
    WEIGHT_NUMERIC = config['wn']
    FINAL_KL_WEIGHT = config['kl']
    
    # Create a unique filename prefix for this run
    FILENAME_PREFIX = f'_wt{WEIGHT_TEXT:.1f}_wn{WEIGHT_NUMERIC:.0f}_kl{FINAL_KL_WEIGHT:.2f}'
    
    print("\n" + "="*80)
    print(f"--- STARTING EXPERIMENT {i+1}/{len(EXPERIMENT_CONFIGS)}: {FILENAME_PREFIX} ---")
    print(f"Weights: TEXT={WEIGHT_TEXT}, NUMERIC={WEIGHT_NUMERIC}, KL={FINAL_KL_WEIGHT}")
    print("="*80)

    # --- 2. Model and Optimizer Setup (MUST BE INSIDE LOOP) ---

    device='cpu'
    # Initialize a NEW model and optimizer for each experiment
    model = WAE(numeric_dim, text_dim, LATENT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training and Validation Loop ---

    log_history = []
    total_train_samples = len(train_loader.dataset)
    total_val_samples = len(val_loader.dataset)

    print(f"Starting training on {device}...")

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # === TRAINING STEP ===
        model.train()
        running_train_loss = 0.0
        running_kl_loss = 0.0
        running_num_loss = 0.0
        running_text_loss = 0.0
        
        # Calculate dynamic KL weight for this epoch
        kl_weight = get_kl_weight(
            epoch, KL_ANNEAL_START_EPOCH, KL_ANNEAL_END_EPOCH, FINAL_KL_WEIGHT
        )
        
        for batch_numeric, batch_text in train_loader:
            batch_numeric = batch_numeric.to(device)
            batch_text = batch_text.to(device)
            
            optimizer.zero_grad()
            
            numeric_recon, text_recon, mu, logvar = model(batch_numeric, batch_text)
            
            loss, text_loss_item, num_loss_item, kl_loss_item = wae_loss(
                numeric_recon, batch_numeric, 
                text_recon, batch_text, 
                mu, logvar, 
                text_weight=WEIGHT_TEXT,
                numeric_weight=WEIGHT_NUMERIC,
                kl_weight=kl_weight 
            )
            
            loss.backward()
            optimizer.step()
            
            batch_size = batch_numeric.size(0)
            running_train_loss += loss.item() * batch_size
            running_kl_loss += kl_loss_item.item() * batch_size
            running_num_loss += num_loss_item.item() * batch_size
            running_text_loss += text_loss_item.item() * batch_size

        avg_train_loss = running_train_loss / total_train_samples
        avg_train_kl = running_kl_loss / total_train_samples
        avg_train_num = running_num_loss / total_train_samples
        avg_train_text = running_text_loss / total_train_samples


        # === VALIDATION STEP ===
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_numeric, batch_text in val_loader:
                batch_numeric = batch_numeric.to(device)
                batch_text = batch_text.to(device)
                
                numeric_recon, text_recon, mu, logvar = model(batch_numeric, batch_text)
                
                loss, _, _, _ = wae_loss(
                    numeric_recon, batch_numeric, 
                    text_recon, batch_text, 
                    mu, logvar, 
                    text_weight=WEIGHT_TEXT,
                    numeric_weight=WEIGHT_NUMERIC,
                    kl_weight=kl_weight 
                )
                
                running_val_loss += loss.item() * batch_numeric.size(0)

        avg_val_loss = running_val_loss / total_val_samples
        
        end_time = time.time()

        # --- LOGGING ---
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | KL-W: {kl_weight:.4f} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Num Recon: {avg_train_num:.4f} | Text Recon: {avg_train_text:.4f} | "
              f"KL: {avg_train_kl:.4f} | Time: {end_time - start_time:.2f}s")
              
        log_history.append({
            'epoch': epoch + 1, 
            'kl_weight': kl_weight,
            'train_loss': avg_train_loss, 
            'val_loss': avg_val_loss,
            'train_recon_num': avg_train_num,
            'train_recon_text': avg_train_text,
            'train_kl_loss': avg_train_kl,
            # Add weights to the log for easy plotting later
            'WEIGHT_TEXT': WEIGHT_TEXT,
            'WEIGHT_NUMERIC': WEIGHT_NUMERIC,
            'FINAL_KL_WEIGHT': FINAL_KL_WEIGHT,
        })
        
        # Removed intermediate checkpoint saving to save only the last one.

    # --- 4. Save Results (DYNAMIC FILENAMES) ---

    # Save the final model checkpoint
    directory_cpt = os.path.join(working_dir, 'output_checkpoints')
    os.makedirs(directory_cpt, exist_ok=True)
    checkpoint_filename = TEXT_EMBEDING_MODEL_NAME + FILENAME_PREFIX + '_final_checkpoint'
    save_checkpoint(model, optimizer, NUM_EPOCHS - 1, checkpoint_filename, directory_cpt)
    print(f"\nFinal checkpoint saved to {checkpoint_filename}.pt")


    # Save training history
    directory_loss = os.path.join(working_dir, 'output_loss')
    os.makedirs(directory_loss, exist_ok=True)
    loss_filename = f'{TEXT_EMBEDING_MODEL_NAME}{FILENAME_PREFIX}_full_loss_history.csv'
    save_loss(
        log_history, 
        directory_loss, 
        loss_filename
    )
    print(f"Loss history saved to {loss_filename}")


    # --- 5. Generate Final Latent Embeddings ---
    directory_embeds = os.path.join(working_dir, 'output_embeds')
    os.makedirs(directory_embeds, exist_ok=True)
    print("\nGenerating latent embeddings for the Test Set...")
    embedding_filename = TEXT_EMBEDING_MODEL_NAME + FILENAME_PREFIX + "_test_data_latent_embeddings"
    mu_filepath, logvar_filepath, latent_vectors_filepath = save_latent_embeddings(model, test_loader, device, embedding_filename, directory_embeds)
    
    print(f"Latent embeddings saved to: {latent_vectors_filepath}")
    print(f"\n--- EXPERIMENT {i+1} COMPLETE ---")

print("\nAll experiments finished. You can now use the analysis script on the saved loss and embedding files.")
