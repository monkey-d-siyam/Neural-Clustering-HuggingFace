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

    for epoch in range(50):
        total_loss = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            images = images.view(images.size(0), -1)  # Flatten to [batch_size, 784]
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/50], Loss: {total_loss / len(train_loader):.4f}")

    # Save model to models/ folder
    torch.save(model.state_dict(), "../models/autoencoder.pth")

def kl_divergence_loss(q):
    p = (q**2) / q.sum(0)
    p = (p.T / p.sum(1)).T
    return (p * torch.log(p / q)).sum(1).mean()

latent = model.encoder(images)
q = 1.0 / (1.0 + torch.sum((latent.unsqueeze(1) - cluster_centers)**2)
q = q**2 / q.sum(0)
loss = reconstruction_loss + 0.1 * kl_divergence_loss(q)

if __name__ == "__main__":
    train_autoencoder()