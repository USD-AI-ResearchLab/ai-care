import torch
from torchvision import datasets, transforms

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    train = datasets.CIFAR10(root="data/", train=True, download=True, transform=transform)
    test  = datasets.CIFAR10(root="data/", train=False, download=True, transform=transform)
    return train, test