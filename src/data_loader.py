from datasets import load_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Grayscale normalization
    ])

    dataset = load_dataset("mnist")

    def transform_example(example):
        example["image"] = transform(example["image"])  # No RGB conversion
        return example

    # Apply transforms
    train_data = dataset["train"].map(transform_example)
    test_data = dataset["test"].map(transform_example)

    # Set PyTorch format
    train_data.set_format(type="torch", columns=["image", "label"])
    test_data.set_format(type="torch", columns=["image", "label"])

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader