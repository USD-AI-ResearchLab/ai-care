import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train = datasets.MNIST(root="data/", train=True, download=True, transform=transform)
    test  = datasets.MNIST(root="data/", train=False, download=True, transform=transform)
    return train, test