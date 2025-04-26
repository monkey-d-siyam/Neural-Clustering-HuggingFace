import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(latent_vectors, cluster_labels):  # ✅ Only 2 parameters now
    tsne = TSNE(n_components=2).fit_transform(latent_vectors)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=cluster_labels, cmap="tab10", alpha=0.6)
    plt.colorbar()
    plt.title("t-SNE Visualization of Clusters")
    plt.savefig("clusters.png")
    plt.show()