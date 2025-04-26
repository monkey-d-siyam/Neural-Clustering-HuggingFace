from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_loader import get_mnist_loaders
from model import Autoencoder
import torch
from visualize import plot_tsne


def cluster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("../models/autoencoder.pth"))  # Check path
    _, test_loader = get_mnist_loaders()

    # Initialize lists
    latent_vectors = []  # ✅ Initialize here
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device).view(-1, 784)
            latent = model.encoder(images)
            latent_vectors.append(latent.cpu().numpy())
            true_labels.append(batch["label"].numpy())

    # Concatenate after collecting all batches
    latent_vectors = np.concatenate(latent_vectors)
    true_labels = np.concatenate(true_labels)

    # Normalize embeddings
    latent_vectors = StandardScaler().fit_transform(latent_vectors)

    # Cluster
    kmeans = KMeans(n_clusters=10, init='k-means++', n_init=10).fit(latent_vectors)
    print(f"Silhouette Score: {silhouette_score(latent_vectors, kmeans.labels_):.2f}")

    # Visualize
    plot_tsne(latent_vectors, kmeans.labels_)


if __name__ == "__main__":
    cluster()  # ✅ Correctly triggers the function