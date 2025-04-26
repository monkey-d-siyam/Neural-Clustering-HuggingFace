# Neural Clustering with Hugging Face Datasets

A PyTorch implementation of autoencoder-based clustering on MNIST.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train.py`
3. Cluster & evaluate: `python src/main.py`

## Results
- Training loss: ~0.01 after 50 epochs
- Silhouette Score: ~0.5