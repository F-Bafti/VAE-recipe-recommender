import torch
import torch.nn as nn
import torch.optim as optim

# Encoder
class Encoder(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        # FIX: Store latent_dim explicitly so it can be accessed for checkpointing
        self.latent_dim = latent_dim
        
        # Numeric branch (20 -> 32)
        self.numeric_branch = nn.Sequential(
            nn.Linear(numeric_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Text branch (768 -> 32)
        self.text_branch = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )
        # Merge (32 + 32 = 64) to mu and logvar
        self.fc_mu = nn.Linear(32 + 32, latent_dim)
        self.fc_logvar = nn.Linear(32 + 32, latent_dim)
        
    def forward(self, numeric, text):
        n = self.numeric_branch(numeric)
        t = self.text_branch(text)
        h = torch.cat([n, t], dim=1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
# Decoder
class Decoder(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        # Latent to numeric
        self.numeric_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, numeric_dim)
        )
        # Latent to text
        self.text_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, text_dim)
        )
        
    def forward(self, z):
        numeric_recon = self.numeric_decoder(z)
        text_recon = self.text_decoder(z)
        return numeric_recon, text_recon
    
# Full VAE (Named WAE in your code)
class WAE(nn.Module):
    def __init__(self, numeric_dim, text_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(numeric_dim, text_dim, latent_dim)
        self.decoder = Decoder(numeric_dim, text_dim, latent_dim)
    
    def forward(self, numeric, text):
        z, mu, logvar = self.encoder(numeric, text)
        numeric_recon, text_recon = self.decoder(z)
        return numeric_recon, text_recon, mu, logvar
    

# Loss function (MODIFIED to support individual weights and return components)
def wae_loss(numeric_recon, numeric, text_recon, text, mu, logvar, text_weight=1.0, numeric_weight=1.0, kl_weight=1.0):
    """
    Calculates the VAE Loss, combining Reconstruction Loss (MSE) and KL Divergence.
    Now supports weighting for text and numeric reconstruction, and returns all components.
    """
    # 1. Reconstruction Loss (using Mean Squared Error) - calculated per sample, summed over batch
    numeric_loss_unweighted = nn.MSELoss(reduction='sum')(numeric_recon, numeric)
    text_loss_unweighted = nn.MSELoss(reduction='sum')(text_recon, text)
    
    # Apply weights
    numeric_loss = numeric_loss_unweighted * numeric_weight
    text_loss = text_loss_unweighted * text_weight
    
    recon_loss = numeric_loss + text_loss
    
    # 2. KL loss (summed over batch)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 3. Total loss is calculated and averaged over the batch size (N)
    batch_size = numeric.size(0)
    
    # Total VAE Loss: (Weighted Recon Loss + Annealed KL Loss) / Batch Size
    total_loss = (recon_loss + kl_weight * kl_loss) / batch_size
    
    # We return total loss and the three unweighted, averaged components for logging
    return total_loss, text_loss_unweighted / batch_size, numeric_loss_unweighted / batch_size, kl_loss / batch_size