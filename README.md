# 🍽️ Recipe Recommender System with Variational Autoencoder (VAE)

This project builds a **Recipe Recommender System** using a **Variational Autoencoder (VAE)** architecture to learn latent taste representations from user–recipe interactions.  
It leverages **PyTorch**, **Numpy**, and **Pandas**, and uses a large-scale dataset of user reviews to provide personalized recipe recommendations.

For a detailed explanation of the methodology, workflow, and insights, check out the accompanying blog post:  
📖 [Read the full blog post here] ([https://f-bafti.github.io/2025/11/11/Building-a-Recipe-Recommender-System-with-VAE.html)

## 🧠 Overview

The model learns user and recipe embeddings through a **VAE** trained on rating data, capturing the underlying taste preferences of users.  
It then predicts missing ratings to generate personalized recipe recommendations.

**Dataset summary:**
- **522,517 recipes**
- **271,907 users**

---

## ⚙️ Workflow

1. **Data Preprocessing**  
   - Filtered active users and frequently rated recipes  
   - Transformed review data into a user–recipe interaction matrix  
   - Split data into training, validation, and test sets  

2. **Model Architecture**  
   - Encoder compresses high-dimensional user–recipe interactions into a low-dimensional latent space  
   - Decoder reconstructs the user’s interaction vector  
   - Loss combines reconstruction error (MSE) and KL divergence  

3. **Training**  
   - Optimized using Adam  
   - KL-divergence weighting (`α`) set to 0.01  
   - Early stopping based on validation loss  

4. **Evaluation & Recommendations**  
   - Reconstructed rating matrix used to predict unseen recipes  
   - Evaluated using RMSE and recall metrics  
