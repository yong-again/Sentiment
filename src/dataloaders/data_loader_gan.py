import torch
from torch.utils.data import Dataset, DataLoader

class TFIDFDataset(Dataset):
    def __init__(self, tfidf_data):
        self.data = tfidf_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]