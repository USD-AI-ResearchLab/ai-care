import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    """Small synthetic dataset for CPU-sane execution."""
    def __init__(self, n=2000, dims=32):
        self.X = torch.randn(n, dims)
        self.y = (self.X.sum(dim=1) > 0).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]