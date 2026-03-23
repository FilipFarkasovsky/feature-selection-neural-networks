import captum.attr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .cancelout import CancelOut
from .deeppink import DeepPINK
from .utils import TrainingSet, TestSet

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight.size()[1] == 1:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(1e-3)

class Model(torch.nn.Module):

    def __init__(self, input_size, n_classes, hidden_dims = None, dropout=0.04308691548552568, activation='mish'):
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

            if activation == 'relu':
                layers.append(torch.nn.ReLU(inplace=inplace))
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU(0.2, inplace=inplace))
            elif activation == 'prelu':
                layers.append(torch.nn.PReLU(h_dim))
            elif activation == 'tanh':
                layers.append(torch.nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(torch.nn.Sigmoid())
            elif activation == 'mish':
                layers.append(torch.nn.Mish(inplace=inplace))
            elif activation == 'selu':
                layers.append(torch.nn.SELU(inplace=inplace))
            else:
                layers.append(torch.nn.Hardswish(inplace=inplace))
            
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
        train_loader = DataLoader(
            TrainingSet(X, Y),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        criterion = torch.nn.BCEWithLogitsLoss() if self.n_classes <= 2 else torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
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

                total_loss += loss.item()

        self.model.eval()
        self.trained = True
    
    def predict_proba(self, X, device='cpu'):
        self.model.eval()
        dataset = TestSet(X)
        loader = DataLoader(dataset, batch_size=len(X), shuffle=False, sampler=None, num_workers=0)
        predictions = []
        for sample in loader:
            x = sample.to(device)
            y_pred = self.model.forward(x)
            if self.n_classes <= 2:
                y_pred = torch.sigmoid(y_pred)
            else:
                y_pred = torch.softmax(y_pred, dim=1)
            predictions += y_pred.data.squeeze().tolist()
        return np.array(predictions)

    def predict(self, X, device='cpu'):
        y_proba = self.predict_proba(X, device=device)
        if len(y_proba.shape) == 2:
            return np.argmax(y_proba, axis=1)
        else:
            return (y_proba > 0.5).astype(int)

    def feature_importance(self, X):
        X = torch.FloatTensor(X)
        X.requires_grad_()
        ig = captum.attr.Saliency(self.model)
        attr = ig.attribute(X, target=0, abs=True)
        scores = attr.detach().numpy()
        return np.mean(np.abs(scores), axis=0)

    @staticmethod
    def create(n_input, n_classes, arch='nn', hidden_dims = None):
        loss_callbacks = []
        if arch == 'nn':
            model = Model(n_input, n_classes, hidden_dims = hidden_dims)
        elif arch == 'cancelout-sigmoid':
            model = ModelWithCancelOut(n_input, n_classes, cancel_out_activation='sigmoid')
            loss_callbacks.append(lambda: model.cancel_out.weight_loss())
        elif arch == 'cancelout-softmax':
            model = ModelWithCancelOut(n_input, n_classes, cancel_out_activation='softmax')
        elif arch == 'deeppink':
            _lambda = 0.05 * np.sqrt(2.0 * np.log(n_input) / 1000)
            model = DeepPINK(Model(n_input, n_classes, hidden_dims=hidden_dims), n_input)
            for layer in model.children():
                # if isinstance(layer, torch.nn.Linear) and (layer.out_features > 1):
                if isinstance(layer, torch.nn.Linear):
                    loss_callbacks.append(lambda l=layer: _lambda * torch.sum(torch.abs(l.weight)))
        else:
            raise NotImplementedError(f'Unknown neural architecture "{arch}"')
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)
        return wrapper