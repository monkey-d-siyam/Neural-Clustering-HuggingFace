import torch
import torch.nn as nn
from torch import optim
from model import Autoencoder
from data_loader import get_mnist_loaders


def train_autoencoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = get_mnist_loaders()
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Initialize cluster centers
    latent_dim = 128
    cluster_centers = torch.randn(10, latent_dim).to(device)  # 10 clusters

    for epoch in range(100):  # Train longer
        model.train()
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device).view(-1, 784)

            # Forward pass
            outputs = model(images)
            latent = model.encoder(images)

            # Reconstruction loss
            recon_loss = criterion(outputs, images)

            # Clustering loss (DEC)
            q = 1.0 / (1.0 + (latent.unsqueeze(1) - cluster_centers).pow(2).sum(2))
            q = q ** 2 / q.sum(0)  # Soft cluster assignments
            p = (q.T / q.sum(1)).T.detach()  # Target distribution
            kl_loss = (p * torch.log(p / q)).sum(1).mean()

            # Total loss
            loss = recon_loss + 0.1 * kl_loss  # Weighted combination

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/100], Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), "../models/autoencoder.pth")

if __name__ == "__main__":
    print("Starting training...")  # Debug statement
    train_autoencoder()