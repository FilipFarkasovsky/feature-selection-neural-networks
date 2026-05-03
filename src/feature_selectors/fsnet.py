import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

from feature_selectors.base_models.base_selector import BaseSelector, ResultType

# from .base_models.nn_models.nn_wrapper import Model
# from .base_models.nn_models.fsnet import FSNet

class FSNetFeatureSelector(BaseSelector):
    """
    FSNet feature selector using a differentiable feature selection layer
    with reconstruction regularization.
    """
    result_type = ResultType.WEIGHTS
    DEFAULT_HIDDEN_DIMS = (10,10)

    def __init__(
        self,
        n_features=None,
        hidden_dims=None,
        **kwargs
    ):
        super().__init__(n_features)
        self.hidden_dims = tuple(hidden_dims) if hidden_dims is not None else self.DEFAULT_HIDDEN_DIMS
        
    def fit(self, X, y, n_informative, **kwargs):
        n_classes = len(set(y))
        n_features = X.shape[1]

        # --- Prepare FSNet model ---
        base_model = Model(2 * n_informative, n_classes)
        fsnet = FSNet(
            base_model, 
            n_features, 
            n_bins = 30,
            n_selected = 2 * n_informative, 
            n_classes = n_classes)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model.to(device)
        fsnet.to(device)

        # --- Preprocess data ---
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.long, device=device)

        # --- Train FSNet ---
        fsnet.fit(X_tensor, y_tensor)

        # --- Extract features from model ----
        self._weights = fsnet.get_feature_importances().astype(float).tolist()
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self






# -*- coding: utf-8 -*-
#
#  nn_wrapper.py
#  
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import tqdm
import captum.attr
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .base_models.nn_models.utils import TrainingSet, TestSet


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        if m.weight.size()[1] == 1:
            torch.nn.init.xavier_uniform_(m.weight)
        else:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        m.bias.data.fill_(1e-3)


class GaussianNoise(torch.nn.Module):

    def __init__(self, stddev):
        torch.nn.Module.__init__(self)
        self.stddev = stddev

    def forward(self, X):
        if self.training:
            X = X + self.stddev * torch.randn_like(X, device = 'cuda')
        return X


"""
class Model(torch.nn.Module):

    def __init__(self, input_size, n_classes, latent_size=16):
        torch.nn.Module.__init__(self)
        n_out = 1 if (n_classes <= 2) else n_classes
        self.layers = torch.nn.Sequential(
            # torch.nn.LayerNorm(input_size),
            GaussianNoise(0.1),
            torch.nn.Linear(input_size, latent_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(latent_size, latent_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(latent_size, n_out))
        self.apply(init_weights)
            
    def forward(self, x):
        return self.layers(x)
"""
class Model(torch.nn.Module):

    def __init__(self, input_size, n_classes, latent_size=58, gaussian_noise=0.7466805127272365, dropout=0.04308691548552568, n_hidden_layers=5, layer_norm=0, activation='mish'):
        torch.nn.Module.__init__(self)
        n_out = 1 if (n_classes <= 2) else n_classes
        layers = []
        if gaussian_noise > 0:
            layers.append(GaussianNoise(gaussian_noise))
        inplace = False
        for k in range(n_hidden_layers):

            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout, inplace=inplace))

            if k == 0:
                layers.append(torch.nn.Linear(input_size, latent_size))
            else:
                layers.append(torch.nn.Linear(latent_size, latent_size))

            if layer_norm:
                layers.append(torch.nn.LayerNorm(latent_size))

            if activation == 'relu':
                layers.append(torch.nn.ReLU(inplace=inplace))
            elif activation == 'leakyrelu':
                layers.append(torch.nn.LeakyReLU(0.2, inplace=inplace))
            elif activation == 'prelu':
                layers.append(torch.nn.PReLU(latent_size))
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

        layers.append(torch.nn.Linear(latent_size, n_out))

        self.layers = torch.nn.Sequential(*layers)
        self.to('cuda')
        self.apply(init_weights)
            
    def forward(self, x):
        return self.layers(x)


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
            device='cuda',
            learning_rate=0.0017601777068292975,  # 0.0005
            epochs=416,  # 1000
            batch_size=56,  # 64
            weight_decay=0.00048519293899787247,  # 1e-5
            val=0.2,
            early_stopping_patience=66,  # 5
            optimizer='adagrad',  # 'adam'
            sam_type='no-sam'  # 'no-sam'
    ):
        self.model.to(device)

        if val > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=val)
        else:
            X_train, y_train = X, Y
            X_test = np.asarray([])
            y_test = np.asarray([])

        dataset = TrainingSet(X_train, y_train)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        
        if val > 0:
            val_dataset = TrainingSet(X_test, y_test)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        else:
            val_loader = None

        self.model.train()
        if self.n_classes <= 2:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')

        if optimizer == 'adam':
            optimizer_class = torch.optim.Adam
        elif optimizer == 'sgd':
            optimizer_class = torch.optim.SGD
        elif optimizer == 'rmsprop':
            optimizer_class = torch.optim.RMSprop
        elif optimizer == 'adamw':
            optimizer_class = torch.optim.AdamW
        else:
            optimizer_class = torch.optim.Adagrad
        if sam_type == 'no-sam':
            optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif sam_type == 'sam':
            optimizer = SharpnessAwareMinimizer(self.model.parameters(), optimizer_class, lr=learning_rate, weight_decay=weight_decay, adaptive=False)
        else:
            optimizer = SharpnessAwareMinimizer(self.model.parameters(), optimizer_class, lr=learning_rate, weight_decay=weight_decay, adaptive=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)

        n_epochs_without_improvement = 0
        state_dict_history = []
        pbar = tqdm.tqdm(range(epochs))
        for e in pbar:

            # Training error
            total_error = 0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()

                def closure():
                    y_hat = self.model.forward(x)
                    if self.n_classes > 2:
                        y_hat = torch.log_softmax(y_hat, dim=1)
                    else:
                        y_hat = y_hat.reshape(len(y_hat))

                    try:
                        loss = criterion(y_hat, y)
                    except RuntimeError:
                        loss = criterion(y_hat, y.float())

                    for loss_callback in self.loss_callbacks:
                        loss = loss + loss_callback()  # Add regularisation terms
                    loss.backward()
                    return loss

                loss = closure()
                if sam_type == 'no-sam':    
                    optimizer.step()
                else:
                    optimizer.step(closure=closure)

                total_error += loss.item()
            scheduler.step(total_error)

            if val_loader is not None:

                # Validation error
                with torch.no_grad():
                    val_total_error = 0
                    for x, y in val_loader:
                        x = x.to(device)
                        y = y.to(device)
                        y_hat = self.model.forward(x)
                        if self.n_classes > 2:
                            y_hat = torch.log_softmax(y_hat, dim=1)
                        else:
                            y_hat = y_hat.reshape(len(y_hat))
                        try:
                            loss = criterion(y_hat, y)
                        except RuntimeError:
                            loss = criterion(y_hat, y.float())
                        val_total_error += loss.item()

                # Keep track of parameters
                state_dict_history.append((val_total_error, self.model.state_dict()))
                if len(state_dict_history) >= 2:
                    if state_dict_history[-1][0] >= state_dict_history[-2][0]:
                        n_epochs_without_improvement += 1
                        if n_epochs_without_improvement >= early_stopping_patience:  # 5
                            break
                    else:
                        n_epochs_without_improvement = 0

                pbar.set_description(str(total_error))

        # Restore best parameters
        if val > 0:
            i = np.argmin([error for error, state_dict in state_dict_history])
            self.model.load_state_dict(state_dict_history[i][1])

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
    def create(dataset_name, n_input, n_classes, arch='nn'):
        loss_callbacks = []
        if arch == 'nn':
            model = Model(n_input, n_classes)
        else:
            raise NotImplementedError(f'Unknown neural architecture "{arch}"')
        wrapper = NNwrapper(model, n_classes)
        for loss_callback in loss_callbacks:
            wrapper.add_loss_callback(loss_callback)
        return wrapper





# -*- coding: utf-8 -*-
#
#  fsnet.py
#  
#  Copyright 2022 Antoine Passemiers <antoine.passemiers@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.

import numpy as np
import torch


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(1e-3)


class WeightPredictor(torch.nn.Module):

    def __init__(self, n_input, n_low, n_output, lhs=True, activation='none'):
        torch.nn.Module.__init__(self)
        self.U = None
        self.n_input = n_input
        self.n_low = n_low
        self.n_output = n_output
        self.lhs = lhs
        if self.lhs:
            shape = (self.n_input, self.n_low)
        else:
            shape = (self.n_low, self.n_output)
        self.weight = torch.nn.Parameter(torch.randn(*shape, requires_grad=True, device='cuda'))
        if activation == 'tanh':
            self.activation = torch.nn.Tanh()
        else:
            self.activation = None
        torch.nn.init.xavier_uniform_(self.weight)

    def init(self, U):
        device = self.weight.device  # use the same device as the model
        self.U = U.clone().detach().to(device=device, dtype=torch.float32)
        if self.lhs:
            assert self.U.size() == (self.n_low, self.n_output)
        else:
            assert self.U.size() == (self.n_input, self.n_low)

    def forward(self):
        if self.lhs:
            X = torch.mm(self.weight, self.U)
        else:
            X = torch.mm(self.U, self.weight)
        if self.activation is not None:
            X = self.activation(X)
        return X


class Selector(torch.nn.Module):

    def __init__(self, n_input, low, k, tau_0=10, tau_e=0.01):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.k = k
        self.tau_0 = tau_0
        self.tau_e = tau_e
        self.tau = self.tau_0
        self.weight_predictor = WeightPredictor(
            self.n_input, low, self.k, lhs=False, activation='none')
        # self.gumbel = torch.distributions.gumbel.Gumbel(0, 0.3)
        self.uniform = torch.distributions.uniform.Uniform(1e-5, 1. - 1e-5)

    def init(self, U):
        self.weight_predictor.init(U)

    def forward(self, X):
        logits = self.compute_logits()
        logits = logits.to('cuda')
        # Concrete variables in the selection layer
        if self.training:
            g = self.sample_gumbel(logits.size(), logits.device)  # pass device
            noisy_logits = (logits + g) / self.tau
        else:
            noisy_logits = logits
        M_T = torch.softmax(noisy_logits, 0)  # Array of shape (n_features, n_selected)

        X = X.to('cuda')

        # Select features
        if self.training:
            X_subset = torch.mm(X, M_T)
        else:
            indices, _ = Selector.uargmax(M_T)
            X_subset = X[:, indices]
        return X_subset

    def sample_gumbel(self, shape, device):
        x = self.uniform.sample(shape).to(device)  # add device arg
        return -torch.log(-torch.log(x))

    def compute_logits(self):
        return self.weight_predictor()

    def update_temperature(self, e, n_epochs):
        self.tau = self.tau_0 * (self.tau_e / self.tau_0) ** ((e + 1) / n_epochs)

    def get_selected_features(self):
        logits = self.compute_logits()
        M_T = torch.softmax(logits, 0)
        idx, _ = Selector.uargmax(M_T)
        return idx.data.numpy()  # add .cpu()

    def get_feature_importances(self):
        logits = self.compute_logits()
        M_T = torch.softmax(logits, 0)
        return np.mean(M_T.detach().cpu().numpy(), axis=1)  # add .cpu()

    @staticmethod
    def uargmax(A):
      # A: tensor of shape (n_features, k)
        device = A.device
        print(device + "uargmax")
        # Ensure values are strictly positive
        A = A - A.min() + 1e-5

        A_work = A.clone()  # avoid modifying original tensor

        n_features, k = A_work.shape

        indices = torch.empty(k, dtype=torch.long, device=device)
        weights = torch.zeros(n_features, dtype=A.dtype, device=device)

        for col in range(k):
            # Flatten and find global max
            flat_idx = torch.argmax(A_work)
            i = flat_idx // k
            j = flat_idx % k

            indices[col] = i
            weights[i] = A_work[i, j]

            # Zero out selected row and column (like original)
            A_work[i, :] = 0
            A_work[:, j] = 0

        return indices, weights


class Encoder(torch.nn.Module):

    def __init__(self, n_input, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, self.n_output),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, X):
        # return self.layers(X)
        return X  # Identity function


class Decoder(torch.nn.Module):

    def __init__(self, n_input, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(self.n_input, self.n_output),
            torch.nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.apply(init_weights)

    def forward(self, X):
        # return self.layers(X)
        return X  # Identity function


class Reconstruction(torch.nn.Module):

    def __init__(self, n_input, n_low, n_output):
        torch.nn.Module.__init__(self)
        self.n_input = n_input
        self.n_output = n_output
        self.weight_predictor = WeightPredictor(
            self.n_input, n_low, self.n_output, lhs=True, activation='none')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_predictor.to(device)

    def init(self, U):
        self.weight_predictor.init(U)

    def forward(self, X):
        weights = self.weight_predictor()
        return torch.mm(X, weights)


class FSNet(torch.nn.Module):

    def __init__(self, model, n_input, n_bins, n_selected, n_classes):
        torch.nn.Module.__init__(self)
        self.model = model
        self.n_input = n_input
        self.n_bins = n_bins
        self.n_selected = n_selected
        self.n_classes = n_classes
        self.selector = Selector(n_input, n_bins, n_selected)
        self.encoder = Encoder(n_selected, n_selected)
        self.decoder = Decoder(n_selected, n_selected)
        self.reconstruction = Reconstruction(n_selected, n_bins, n_input)

    def fit(self, X, y, n_epochs=600, batch_size=64, _lambda=10, weight_decay=1e-6):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)
        self.selector.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reconstruction.to(device)
        self.model.to(device)
        # Initialize weight predictors
        U = FSNet.compute_u(X, n_bins=self.n_bins, device=device)
        self.selector.init(U)
        self.reconstruction.init(U.t())
        self.selector.to(device)
        dataset = TrainingSet(X, y, device)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0)
        self.model.train()
        if self.n_classes <= 2:
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = torch.nn.NLLLoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.9, patience=10, verbose=False, threshold=0.0001,
            threshold_mode='rel', cooldown=5, min_lr=1e-5, eps=1e-08)
        self.model.to(device)
        for e in range(n_epochs):

            # Lower the temperature
            self.selector.update_temperature(e, n_epochs)

            total_loss = 0
            for _X, _y in loader:
                _X = _X.to(device)
                _y = _y.to(device)
                optimizer.zero_grad()

                # Select a subset of features
                X_subset = self.selector.forward(_X)

                # Predict the target variable from the selected subset of features
                X_latent = self.encoder.forward(X_subset)
                y_hat = self.model.forward(X_latent)

                # Reconstruct the input data
                X_reconstructed = self.decoder.forward(X_latent)
                X_reconstructed = self.reconstruction.forward(X_reconstructed)

                # Compute loss function
                if self.n_classes > 2:
                    loss1 = criterion(y_hat, _y)
                else:
                    loss1 = criterion(torch.squeeze(y_hat), torch.squeeze(_y.float()))
                loss2 = _lambda * torch.mean((_X - X_reconstructed) ** 2)
                loss = loss1 + loss2

                # print(_X.size(), X_subset.size(), X_latent.size(), y_hat.size(), X_reconstructed.size())

                #print(loss1.item(), loss2.item())
                total_loss += loss.item()

                # Update parameters
                loss.backward()
                optimizer.step()
            scheduler.step(total_loss)

            # print(f'Total loss at epoch {e + 1}: {total_loss}')

        self.model.eval()

    def predict(self, X):
        X = torch.FloatTensor(X)
        X = self.selector.forward(X)
        X = self.encoder.forward(X)
        y_pred = self.model.forward(X)
        if self.n_classes <= 2:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
        return torch.squeeze(y_pred).data.numpy()

    def get_selected_features(self):
        return self.selector.get_selected_features()

    def get_feature_importances(self):
        importances = self.selector.get_feature_importances()
        return importances

    @staticmethod
    def compute_u(X, n_bins=20, device='cpu'):  # add device arg
        n_features = X.shape[1]
        U = np.zeros((n_features, n_bins), dtype=float)
        for j in range(n_features):
            hist = np.histogram(X[:, j].cpu().numpy(), n_bins)
            U[j, :] = 0.5 * hist[0][:] * (hist[1][:-1] + hist[1][1:])
        U -= U.mean()
        U /= U.std()
        return torch.FloatTensor(U).to(device)  




# Adapted from https://github.com/davda54/sam/blob/main/sam.py

from typing import Optional, Callable, List

import torch


class SharpnessAwareMinimizer(torch.optim.Optimizer):
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SharpnessAwareMinimizer, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


"""
class SharpnessAwareMinimizer(torch.optim.Optimizer):

    def __init__(
            self,
            params: List[torch.nn.Parameter],
            base_optimizer_class,
            rho: float = 0.05,
            eps: float = 1e-12,
            adaptive: bool = False,
            **kwargs
    ):
        super(SharpnessAwareMinimizer, self).__init__(params, dict(rho=rho, adaptive=adaptive, **kwargs))
        self.base_optimizer = base_optimizer_class(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.rho: float = rho
        self.adaptive: bool = adaptive
        self.eps: float = eps

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> None:
        closure = torch.enable_grad()(closure)

        with torch.no_grad():
            grad_norm = self.compute_grad_norm()
            for group in self.param_groups:
                scale = group['rho'] / (grad_norm + self.eps)
                for p in group['params']:
                    if p.grad is None:
                        continue
                    self.state[p]['old_param'] = p.data.clone()
                    e_w = (torch.square(p) if self.adaptive else 1.0) * p.grad * scale.to(p)
                    p.add_(e_w)
            self.zero_grad()

        closure()

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.data = self.state[p]['old_param']
            self.base_optimizer.step()

    def compute_grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        grad = torch.stack([
            ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(device)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ])
        return torch.norm(grad, p=2)
"""