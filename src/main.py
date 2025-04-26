import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_loader import get_mnist_loaders
from model import Autoencoder
import torch
from visualize import plot_tsne  # Import the function


def cluster():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load("../models/autoencoder.pth"))  # Path adjusted
    _, test_loader = get_mnist_loaders()

    latent_vectors = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device).view(-1, 784)
            latent = model.encoder(images)
            latent_vectors.append(latent.cpu().numpy())
            true_labels.append(batch["label"].numpy())

    latent_vectors = np.concatenate(latent_vectors)
    true_labels = np.concatenate(true_labels)

    # Cluster embeddings
    kmeans = KMeans(n_clusters=10).fit(latent_vectors)
    print(f"Silhouette Score: {silhouette_score(latent_vectors, kmeans.labels_):.2f}")

    # Call the visualization here (inside the function)
    plot_tsne(latent_vectors, kmeans.labels_)


if __name__ == "__main__":
    cluster()  # This triggers the code above