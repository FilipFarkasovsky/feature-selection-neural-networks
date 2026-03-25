import torch


class TrainingSet(torch.utils.data.Dataset):
    
    def __init__(self, X, Y, device = 'cpu'):
        assert len(Y) == len(X)        
        self.X = X.clone().detach().to(device=device, dtype=torch.float32)
        self.Y = Y.clone().detach().to(device=device, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TestSet(torch.utils.data.Dataset):
    
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]