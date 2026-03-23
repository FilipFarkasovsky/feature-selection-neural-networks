import captum.attr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .cancelout import CancelOut
from .utils import TrainingSet, TestSet

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight.size()[1] == 1:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(1e-3)

class Model(torch.nn.Module):
    def __init__(self, input_size, n_classes, hidden_dims = None, dropout=0.04308691548552568):
        torch.nn.Module.__init__(self)
        n_out = 1 if (n_classes <= 2) else n_classes
        layers = []
        inplace = False
        if hidden_dims is None:
            hidden_dims = [32] * 3 
        
        prev_dim = input_size
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout, inplace=inplace))
            layers.append(torch.nn.ReLU(inplace=inplace))
            prev_dim = h_dim

        layers.append(torch.nn.Linear(prev_dim, n_out))
        self.layers = torch.nn.Sequential(*layers)
        self.apply(init_weights)
            
    def forward(self, x):
        return self.layers(x)


class ModelWithCancelOut(torch.nn.Module):

    def __init__(self, input_size, n_classes, hidden_dims = None, cancel_out_activation='sigmoid'):
        torch.nn.Module.__init__(self)
        self.cancel_out = CancelOut(input_size, activation=cancel_out_activation)
        self.model = Model(input_size, n_classes, hidden_dims = hidden_dims)

    def forward(self, x):
        x = self.cancel_out(x)
        return self.model(x)


class NNwrapper:

    def __init__(self, model, n_classes):
        self.model = model
        self.n_classes = n_classes
        self.loss_callbacks = []
        self.trained = False

    def add_loss_callback(self, func):
        self.loss_callbacks.append(func)

    def fit(
            self,
            X,
            Y,
            device='cpu',
            learning_rate=1e-3,
            epochs=200,  
            batch_size=64,  # 64
            weight_decay=1e-5  # 1e-5
    ):
        self.model.to(device)

        # Create dataloader for all samples
        train_loader = DataLoader(TrainingSet(X, Y),batch_size=batch_size, shuffle=True)
        criterion = torch.nn.BCEWithLogitsLoss() if self.n_classes <= 2 else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_hat = self.model(x)
                if self.n_classes <= 2:
                    y_hat = y_hat.squeeze()
                    y = y.float()
                loss = criterion(y_hat, y)
                for cb in self.loss_callbacks:
                    loss += cb()
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.trained = True