import numpy as np
import torch


class LocallyConnected1d(torch.nn.Module):
    """
    Locally connected layer - no weight sharing across features.
    Used to separately process original features and their knockoffs.
    """

    def __init__(self, n_filters, kernel_size, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((n_filters, kernel_size), 0.1))
        self.bias = (torch.nn.Parameter(torch.zeros(n_filters)) if bias else None)

    def forward(self, X):
        # X shape: (batch, features, 2)
        X = torch.einsum('bik,ik->bi', X, self.weight)
        if self.bias is not None:
            X = X + self.bias.unsqueeze(0)
        return X.unsqueeze(2)


class DeepPINK(torch.nn.Module):
    """
    DeepPINK model- compares original features vs knockoffs
    """
    def __init__(self, model, n_input):
        super().__init__()
        self.lc1 = LocallyConnected1d(n_input, 2)
        self.lc2 = LocallyConnected1d(n_input, 1)
        torch.nn.init.xavier_normal_(self.lc2.weight)
        self.model = model

    def forward(self, X):
        X = self.lc1(X)
        X = self.lc2(X)
        X = torch.squeeze(X, 2)
        X = self.model(X)
        return X

    def get_weights(self):
        """
        Compute feature importance based on:
        difference between original and knockoff contributions.
        """
        W0 = self.lc2.weight.detach().cpu().numpy()
        
        # Accumulate weights across linear layers
        W_acc = None
        W = []
        for layer in self.model.layers:
            if isinstance(layer, torch.nn.Linear):
                W.append(layer.weight.data.numpy().T)
                W_acc = W[-1] if (W_acc is None) else np.dot(W_acc, W[-1])

        w = np.squeeze(W0 * W_acc)

        if len(w.shape) == 1:
            z = self.lc1.weight.data.numpy()[:, 0] * w
            z_tilde = self.lc1.weight.data.numpy()[:, 1] * w
            return z ** 2. - z_tilde ** 2.
        else:
            z = self.lc1.weight.data.numpy()[:, 0, np.newaxis] * w
            z_tilde = self.lc1.weight.data.numpy()[:, 1, np.newaxis] * w
            return np.mean(z ** 2. - z_tilde ** 2., axis=1)