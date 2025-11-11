import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
from time import time
# --- NEW IMPORTS ---
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
from scipy.spatial.distance import cdist


# --- END NEW IMPORTS ---


def find_nearest_neighbors(Z_test, test_recipes_df, query_recipe_index, k=10):
    """
    Finds the K nearest neighbors to a query point in the latent space.
    
    Args:
        Z_test (np.ndarray): The 32D latent vectors for the test set.
        test_recipes_df (pd.DataFrame): DataFrame containing the recipes corresponding to Z_test.
        query_recipe_index (int): The row index of the query recipe *within Z_test/test_recipes_df*.
        k (int): The number of neighbors (including the query itself) to return.
        
    Returns:
        pd.DataFrame: A DataFrame of the top k nearest recipes (including the query).
    """
    print(f"\nSearching for {k-1} nearest neighbors to query index {query_recipe_index}...")
    
    # 1. Initialize and fit the NearestNeighbors model
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', algorithm='auto')
    nn.fit(Z_test)
    
    # 2. Extract the query point
    query_point = Z_test[query_recipe_index].reshape(1, -1)
    
    # 3. Find neighbors
    distances, indices = nn.kneighbors(query_point)
    
    # The indices array contains the row indices *within* Z_test/test_recipes_df
    neighbor_indices_in_test_set = indices.flatten()
    neighbor_distances = distances.flatten()
    
    # 4. Filter the DataFrame and combine with distances
    recommended_recipes = test_recipes_df.iloc[neighbor_indices_in_test_set].copy()
    recommended_recipes['Distance'] = neighbor_distances
    
    # 5. Sort (should already be sorted by kneighbors)
    recommended_recipes = recommended_recipes.sort_values(by='Distance', ascending=True)
    
    # Identify the query row explicitly
    recommended_recipes['is_query'] = (recommended_recipes.index == test_recipes_df.iloc[query_recipe_index].name)
    
    return recommended_recipes


# --- NEW FUNCTION 1: CLUSTERING ---
def perform_clustering(Z_32D, n_clusters=8, random_seed=42):
    """
    Applies K-Means clustering to the 32D latent vectors.
    
    Args:
        Z_32D (np.ndarray): The 32D latent vectors (Z_sample).
        n_clusters (int): The number of clusters (k) for K-Means.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: The cluster labels (integers) for each sample.
    """
    print(f"Applying K-Means clustering (k={n_clusters}) on {Z_32D.shape[0]} samples...")
    # Setting n_init=10 explicitly to avoid future deprecation warnings
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(Z_32D)
    print("Clustering complete.")
    return cluster_labels


# --- NEW FUNCTION 2: DIMENSIONALITY REDUCTION ---
def perform_umap_reduction(Z_32D, n_components=2, random_seed=42):
    """
    Applies UMAP to reduce the dimensionality of the latent vectors.
    
    Args:
        Z_32D (np.ndarray): The 32D latent vectors (Z_sample).
        n_components (int): The target dimensionality (usually 2).
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        np.ndarray: The 2D (or n_components) UMAP coordinates.
    """
    print(f"Applying UMAP reduction ({Z_32D.shape[1]}D -> {n_components}D) on {Z_32D.shape[0]} samples...")
    t0 = time()
    umap_reducer = UMAP(
        n_components=n_components, 
        random_state=random_seed, 
        n_neighbors=15, # Balance between local and global structure
        min_dist=0.1
    )
    Z_2D = umap_reducer.fit_transform(Z_32D)
    t1 = time()
    print(f"UMAP completed in {(t1-t0):.2f} seconds.")
    return Z_2D


def perform_pca_reduction(Z, n_components=2, random_seed=42):
    """
    Applies PCA to reduce the dimensionality of the latent vectors and returns explained variance.
    
    Args:
        Z (np.ndarray): The high-dimensional latent vectors (e.g., 32D).
        n_components (int): The target dimensionality (usually 2).
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        Z_reduced (np.ndarray): The reduced n_components-dimensional coordinates.
        explained_variance (np.ndarray): Explained variance ratio of each component.
    """
    print(f"Applying PCA reduction ({Z.shape[1]}D -> {n_components}D) on {Z.shape[0]} samples...")
    t0 = time()
    pca_reducer = PCA(n_components=n_components, random_state=random_seed)
    Z_reduced = pca_reducer.fit_transform(Z)
    explained_variance = pca_reducer.explained_variance_ratio_
    t1 = time()
    print(f"PCA completed in {(t1-t0):.2f} seconds.")
    print(f"Explained variance ratio: {explained_variance}")
    return Z_reduced, explained_variance


def plot_losses(filepath, title="Training and Validation Loss Over Epochs"):
    """
    Reads a CSV file containing loss data and plots the training and validation curves.

    Args:
        filepath (str): Path to the CSV file containing loss history.
        title (str): Title for the plot.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Loss file not found at {filepath}")
        return

    plt.figure(figsize=(10, 6))
    
    # Plotting Total Loss
    plt.plot(df['epoch'], df['train_loss'], label='Total Training Loss (Weighted)', color='darkblue')
    plt.plot(df['epoch'], df['val_loss'], label='Total Validation Loss (Weighted)', color='red')
    
    # Optional: Plotting Unweighted Numeric and KL for insight
    if 'train_recon_num' in df.columns:
        plt.plot(df['epoch'], df['train_recon_num'], label='Unweighted Numeric Recon Loss', color='green', linestyle='--')
    if 'train_kl_loss' in df.columns:
        plt.plot(df['epoch'], df['train_kl_loss'], label='KL Loss (Annealed)', color='orange', linestyle=':')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


import numpy as np

def check_collapse_diagnostics(mu_filepath, logvar_filepath):
    """
    Loads mu and logvar tensors and returns statistics to check for VAE posterior collapse.
    Returns a dictionary of values for easy DataFrame creation.
    """
    try:
        mu = np.load(mu_filepath)
        logvar = np.load(logvar_filepath)
    except FileNotFoundError as e:
        print(f"Error loading diagnostic data: {e}")
        return None

    result = {}
    result['num_samples'] = mu.shape[0]
    result['latent_dim'] = mu.shape[1]

    # Mu statistics
    mu_overall_mean = np.mean(mu)
    mu_overall_std = np.std(mu)
    result['mu_mean'] = mu_overall_mean
    result['mu_std'] = mu_overall_std

    # Mu interpretation
    if mu_overall_std < 0.1:
        result['mu_check'] = "Posterior collapse"
    elif mu_overall_std < 0.5:
        result['mu_check'] = "Caution: compact latent space"
    else:
        result['mu_check'] = "Healthy"

    # Logvar statistics
    logvar_overall_std = np.std(logvar)
    avg_variance = np.mean(np.exp(logvar))
    result['logvar_std'] = logvar_overall_std
    result['logvar_avg_variance'] = avg_variance

    # Logvar interpretation
    if logvar_overall_std < 0.05:
        result['logvar_check'] = "Caution: low variance"
    else:
        result['logvar_check'] = "Healthy"

    return result


def visualize_latent_embeddings(ax, Z_2D, cluster_labels, n_clusters, title=""):
    """
    Plots pre-calculated 2D coordinates (e.g., UMAP) on the given matplotlib axis.
    """
    if Z_2D is None or cluster_labels is None:
        print("Error: Missing data for plotting.")
        return

    scatter = ax.scatter(
        Z_2D[:, 0],
        Z_2D[:, 1],
        c=cluster_labels,
        cmap='Spectral',
        s=5,
        alpha=0.6
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('UMAP 1', fontsize=8)
    ax.set_ylabel('UMAP 2', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    return scatter  # so we can add colorbars later if we want


def plot_training_metrics(df, min_epoch=0):
    """
    Plots training metrics (loss and component losses) across epochs for different weights.
    Expects df to have the following columns:
    ['epoch', 'val_loss', 'train_loss', 'train_recon_num', 'train_recon_text',
     'train_kl_loss', 'WEIGHT_NUMERIC', 'WEIGHT_TEXT', 'FINAL_KL_WEIGHT']
    """
    sns.set(style="whitegrid", font_scale=1.2)
    unique_weights = df[['WEIGHT_NUMERIC', 'WEIGHT_TEXT', 'FINAL_KL_WEIGHT']].drop_duplicates()
    palette = sns.color_palette("husl", len(unique_weights))
    
    df = df[df['epoch'] >= min_epoch]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(unique_weights.iterrows()):
        subset = df[
            (df['WEIGHT_NUMERIC'] == row['WEIGHT_NUMERIC']) &
            (df['WEIGHT_TEXT'] == row['WEIGHT_TEXT']) &
            (df['FINAL_KL_WEIGHT'] == row['FINAL_KL_WEIGHT'])
        ]
        label = f"wn={row['WEIGHT_NUMERIC']}, wt={row['WEIGHT_TEXT']}, kl={row['FINAL_KL_WEIGHT']}"
        color = palette[i]

        # val & train loss
        axes[0].plot(subset['epoch'], subset['val_loss'], label=f"val {label}", color=color, linestyle='--', linewidth=2)
        axes[0].plot(subset['epoch'], subset['train_loss'], label=f"train {label}", color=color, linewidth=2)

        # train_recon_num
        axes[1].plot(subset['epoch'], subset['train_recon_num'], label=label, color=color, linewidth=2)

        # train_recon_text
        axes[2].plot(subset['epoch'], subset['train_recon_text'], label=label, color=color, linewidth=2)

        # train_kl_loss
        axes[3].plot(subset['epoch'], subset['train_kl_loss'], label=label, color=color, linewidth=2)

    axes[0].set_title("Validation vs Training Loss")
    axes[1].set_title("Train Reconstruction (Numeric)")
    axes[2].set_title("Train Reconstruction (Text)")
    axes[3].set_title("Train KL Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


def analyze_cluster_quality(run_name: str, Z_32D: np.ndarray, cluster_labels: np.ndarray, test_recipes: pd.DataFrame):
    """
    Analyzes clustering quality for a VAE model:
      1. Computes mean and std for each cluster
      2. Finds top 5 closest points to centroid in latent space
      3. Returns structured data for later plotting or inspection
    """
    # print("\n" + "="*80)
    # print(f"--- ANALYZING RUN: {run_name} (Clusters: {len(np.unique(cluster_labels))}) ---")
    # print("="*80)

    ANALYSIS_COLUMNS = [
        'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
        'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 
        'ProteinContent'
    ]

    # 1. PREPARE DATA
    numeric_profile_df = test_recipes.select_dtypes(include=np.number).copy()

    try:
        numeric_profile_df = numeric_profile_df[ANALYSIS_COLUMNS]
    except KeyError as e:
        print(f"ERROR: Missing required column: {e}")
        return None

    numeric_profile_df['cluster'] = cluster_labels

    # 2. QUANTITATIVE ANALYSIS
    mean_profile = numeric_profile_df.groupby('cluster').mean()
    std_profile = numeric_profile_df.groupby('cluster').std()

    profile_table = pd.concat([mean_profile.round(3), std_profile.round(3)], 
                              axis=1, 
                              keys=['Mean', 'Std'])

    # print("\n## 1. Quantitative Cluster Profiles (Scaled Features)")
    # print(profile_table.to_markdown())

    # 3. NUMERIC COHESION ANALYSIS
    # print("\n## 2. Numeric Cluster Cohesion (Top 5 Closest Recipes)")
    results = {'cluster_means': mean_profile, 
               'cluster_stds': std_profile, 
               'top5_members': {}}

    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]

        if len(cluster_indices) < 5:
            # print(f"\n--- Cluster {cluster_id} too small for analysis (size={len(cluster_indices)}) ---")
            continue

        Z_cluster = Z_32D[cluster_mask]
        centroid = np.mean(Z_cluster, axis=0, keepdims=True)
        distances = cdist(Z_cluster, centroid, metric='euclidean').flatten()

        closest_idx = np.argsort(distances)[:5]
        original_idx = cluster_indices[closest_idx]

        cohesion_df = numeric_profile_df.loc[original_idx, ANALYSIS_COLUMNS].T
        cohesion_df.columns = [f'Top {i+1}' for i in range(5)]

        # print(f"\n--- Cluster {cluster_id} (Size: {len(cluster_indices)}) ---")
        # print(cohesion_df.round(3).to_markdown())
        # print("---")

        # store for later plotting or export
        results['top5_members'][cluster_id] = {
            'indices': original_idx,
            'profiles': cohesion_df
        }

    return results
   

def get_cluster_recommendations(cluster_assignments: np.ndarray, recipe_df: pd.DataFrame, liked_recipe_id: int):
    """
    Identifies the cluster of a liked recipe and returns all other recipes 
    within that same cluster as recommendations.

    Args:
        cluster_assignments (np.ndarray): The 1D NumPy array of **cluster labels** (e.g., shape `(N,)`), where N is the number 
                                          of recipes. This array must contain the integer 
                                          Cluster ID assigned to each recipe, NOT the 
                                          multidimensional embedding vectors.
        recipe_df (pd.DataFrame): The DataFrame containing the original recipe data.
                                  It MUST have a 'RecipeId' column that matches
                                  the length/index of the cluster_assignments.
        liked_recipe_id (int): The ID of the recipe the user liked.
    
    Returns:
        pd.DataFrame: A DataFrame containing the recommended recipes.
    """
    
    # 1. Add the cluster assignments to the DataFrame
    # IMPORTANT: Ensure the indices align when adding the column.
    if len(cluster_assignments.shape) != 1:
        print(f"Error: Expected 1D array of cluster labels, but received array with shape {cluster_assignments.shape}. Please pass the cluster IDs (e.g., Z_labels), not the embeddings (e.g., Z_32D).")
        return pd.DataFrame()
        
    if len(cluster_assignments) != len(recipe_df):
        print("Error: The length of cluster_assignments does not match the length of recipe_df.")
        return pd.DataFrame()

    df = recipe_df.copy()
    df['cluster'] = cluster_assignments
    
    # print("--- Recommendation Process Initiated ---")

    # 2. Find the liked recipe
    liked_recipe = df[df['RecipeId'] == liked_recipe_id]

    if liked_recipe.empty:
        print(f"Error: Recipe ID {liked_recipe_id} not found in the dataset.")
        return pd.DataFrame()
    
    # Extract the cluster ID (it will be a single value)
    target_cluster = liked_recipe['cluster'].iloc[0]
    liked_recipe_name = liked_recipe['Name'].iloc[0]
    
    # print(f"Liked Recipe: '{liked_recipe_name}' (ID: {liked_recipe_id})")
    # print(f"Assigned Cluster: {target_cluster}")
    # print("-" * 35)

    # 3. Filter for all recipes in the same cluster, excluding the liked one
    recommendations = df[
        (df['cluster'] == target_cluster) & 
        (df['RecipeId'] != liked_recipe_id)
    ]
    
    # 4. Prepare and return the output
    if recommendations.empty:
        print(f"No other recipes found in Cluster {target_cluster} to recommend.")
    else:
        # Select and format relevant columns for display
        output_cols = ['RecipeId', 'Name', 'RecipeCategory', 'RecipeIngredientParts', '', 'cluster', 'Calories', 'ProteinContent', 'CarbohydrateContent', 'RecipeInstructions']
        # If the columns don't exist, only show what does.
        final_cols = [col for col in output_cols if col in recommendations.columns]
        
        # Sort recommendations to make them easy to review
        recommendations = recommendations.sort_values(by='Name').reset_index(drop=True)
        
        # print(f"Found {len(recommendations)} recommendations in Cluster {target_cluster}.")
        
        return recommendations[final_cols]


def plot_cluster_quality(results_dict, run_name):
    """
    Plots cluster mean and std for each numeric feature side by side.
    """
    cluster_means = results_dict['cluster_means']
    cluster_stds = results_dict['cluster_stds']

    # Prepare long format for plotting
    means_long = cluster_means.reset_index().melt(id_vars='cluster', var_name='Feature', value_name='Mean')
    stds_long = cluster_stds.reset_index().melt(id_vars='cluster', var_name='Feature', value_name='Std')

    # Merge mean and std info for aligned plotting
    merged = pd.merge(means_long, stds_long, on=['cluster', 'Feature'])

    # Create a grid of subplots: Mean | Std
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    fig.suptitle(f'Cluster Feature Profiles — {run_name}', fontsize=16)

    # --- Left: Means ---
    sns.barplot(
        data=merged, x='Feature', y='Mean', hue='cluster', ax=axes[0], capsize=0.1
    )
    axes[0].set_title('Cluster Means per Feature', fontsize=14)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Mean (Scaled)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Cluster')

    # --- Right: Standard Deviations ---
    sns.barplot(
        data=merged, x='Feature', y='Std', hue='cluster', ax=axes[1], capsize=0.1
    )
    axes[1].set_title('Cluster Standard Deviations', fontsize=14)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Std Dev (Scaled)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Cluster')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()