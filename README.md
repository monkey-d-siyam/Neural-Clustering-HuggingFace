# Neural Clustering with Hugging Face Datasets

A PyTorch implementation of autoencoder-based clustering on MNIST.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train.py`
3. Cluster & evaluate: `python src/main.py`


## Features
- **Autoencoder Architecture**: Learns compact latent representations of data.
- **Clustering Integration**: Applies K-Means to latent embeddings.
- **Visualization**: t-SNE/PCA plots for cluster analysis.
- **Metrics**: Silhouette Score, Davies-Bouldin Index.

## Technologies
- PyTorch
- Hugging Face `datasets`
- Scikit-learn
- Matplotlib/Seaborn
