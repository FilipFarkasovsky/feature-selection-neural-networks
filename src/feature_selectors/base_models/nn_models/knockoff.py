import numpy as np
import scipy.linalg
import torch


def make_g_spd(sigma):
    sigma = torch.FloatTensor(sigma)
    s = torch.nn.Parameter(0.0001 * torch.ones(len(sigma)))
    optimizer = torch.optim.SGD([s], lr=1e-3)
    for _ in range(100):
        optimizer.zero_grad()
        G1 = torch.cat([sigma, sigma - torch.diag(s)], dim=1)
        G2 = torch.cat([sigma - torch.diag(s), sigma], dim=1)
        G = torch.cat([G1, G2], dim=0)
        eigenvalues = torch.linalg.eigvalsh(G)
        loss = -torch.min(eigenvalues)
        loss.backward()
        print(loss.item())
        optimizer.step()
    return s.cpu().data.numpy()


def generate_gaussian_knockoffs(X, eps=1e-3):
    mu = np.mean(X, axis=0)
    p = len(mu)
    sigma = np.cov(X, rowvar=False)

    # Regularization
    lambda_ = 0.8
    sigma = lambda_ * np.diag(np.diagonal(sigma)) + (1. - lambda_) * sigma

    s = np.diagonal(sigma)

    # Ensure G is spd
    G = np.block([
        [sigma, sigma - np.diag(s)],
        [sigma - np.diag(s), sigma]
    ])
    # assert G.shape[0] == np.linalg.matrix_rank(G)

    S = np.diag(s)
    sigma_inv_d = scipy.linalg.lstsq(sigma, S)[0]
    V = 2. * S - np.dot(S, sigma_inv_d)
    L = np.linalg.cholesky(V + eps * np.eye(p))

    n, p = X.shape
    mu_tilde = X - np.dot(X - np.tile(mu, (n, 1)), sigma_inv_d)
    return mu_tilde + np.dot(np.random.normal(size=mu_tilde.shape), L.T)
