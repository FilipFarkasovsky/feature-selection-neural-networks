"""
Neural Network Pipeline — PyTorch
----------------------------------
Entry point: run_pipeline(X, Y)

Architecture rationale
----------------------
Input dim ~100-130 is moderately high for 1 000 samples, so a straight
128→16→16 bottleneck risks an information loss that hurts early training.
A staged reduction  input→128→64→16→16→out  gives the network room to
learn gradually and is still lightweight enough to avoid over-fitting.

Regularisation stack:
  • BatchNorm  — stabilises activations, acts as a mild regulariser
  • Dropout(0.3 / 0.2) — explicit regularisation on wider layers
  • Early stopping — stops training when val-loss stops improving

Weights: Kaiming Uniform (He) — the standard choice for ReLU networks.
Metric : Macro F1 on a held-out test set (stratified split).
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, accuracy_score, roc_auc_score


# ──────────────────────────────────────────────────────────────────────────────
# 1. Model
# ──────────────────────────────────────────────────────────────────────────────

class NeuralNet(nn.Module):
    """
    Staged-reduction MLP with BatchNorm + Dropout + ReLU.

    Layer widths: in → 128 → 64 → 16 → 16 → n_classes
    The first two layers handle the high-dimensional input and compress it
    before the two 16-unit layers specified by the user.
    """

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()

        self.net = nn.Sequential(
            # ── Block 1: input → 128 ──────────────────────────────────────
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # ── Block 2: 128 → 64 ────────────────────────────────────────
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # ── Block 3: 64 → 16  (first user-requested layer) ───────────
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # ── Block 4: 16 → 16  (second user-requested layer) ──────────
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # ── Output ────────────────────────────────────────────────────
            nn.Linear(16, n_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming Uniform initialisation for every Linear layer."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Early Stopping
# ──────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stops training when validation loss has not improved for `patience` epochs.
    Saves the best model weights in memory so they can be restored.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_tensor(arr: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype)


def _train_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb)
        total_loss += loss.item() * len(xb)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def _predict(model, loader, device) -> np.ndarray:
    model.eval()
    preds = []
    for xb, _ in loader:
        logits = model(xb.to(device))
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    # ── split sizes ──────────────────────────────────────────────────────────
    val_size:     float = 0.15,
    test_size:    float = 0.15,
    # ── training hyper-parameters ────────────────────────────────────────────
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs:   int   = 300,
    patience:     int   = 20,
    # ── misc ─────────────────────────────────────────────────────────────────
    random_state: int   = 42,
    verbose:      bool  = True,
) -> dict:
    """
    Full classification pipeline.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.  n_features should be in the ~100-130 range.
    Y : np.ndarray, shape (n_samples,)
        Class labels (int or str — encoded internally).

    Returns
    -------
    dict with keys:
        model        – trained NeuralNet (best weights restored)
        macro_f1     – macro-averaged F1 on the test set
        report       – full sklearn classification_report string
        history      – dict of train/val loss lists
        label_encoder– fitted LabelEncoder
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    # ── 4a. Encode labels ────────────────────────────────────────────────────
    le = LabelEncoder()
    Y_enc = le.fit_transform(Y).astype(np.int64)
    n_classes = len(le.classes_)
    input_dim = X.shape[1]

    if verbose:
        print(f"Classes     : {n_classes}  {list(le.classes_)}")
        print(f"Input dim   : {input_dim}")
        print(f"Samples     : {len(X)}")

    # ── 4b. Stratified split: train / val / test ─────────────────────────────
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    trainval_idx, test_idx = next(sss_test.split(X, Y_enc))

    X_trainval, Y_trainval = X[trainval_idx], Y_enc[trainval_idx]
    X_test,     Y_test     = X[test_idx],     Y_enc[test_idx]

    relative_val = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_val, random_state=random_state
    )
    train_idx, val_idx = next(sss_val.split(X_trainval, Y_trainval))

    X_train, Y_train = X_trainval[train_idx], Y_trainval[train_idx]
    X_val,   Y_val   = X_trainval[val_idx],   Y_trainval[val_idx]

    if verbose:
        print(
            f"Split       : train={len(X_train)}  "
            f"val={len(X_val)}  test={len(X_test)}"
        )

    # ── 4c. DataLoaders ──────────────────────────────────────────────────────
    def make_loader(Xd, Yd, shuffle):
        ds = torch.utils.data.TensorDataset(
            _to_tensor(Xd, torch.float32),
            _to_tensor(Yd, torch.int64),
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle
        )

    train_loader = make_loader(X_train, Y_train, shuffle=True)
    val_loader   = make_loader(X_val,   Y_val,   shuffle=False)
    test_loader  = make_loader(X_test,  Y_test,  shuffle=False)

    # ── 4d. Model, loss, optimiser, scheduler ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = NeuralNet(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max_epochs, eta_min=1e-6
    )
    stopper = EarlyStopping(patience=patience)

    # ── 4e. Training loop ────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}

    actual_epochs = 0
    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch

        train_loss = _train_epoch(model, train_loader, criterion, optimiser, device)
        val_loss   = _eval_epoch(model,   val_loader, criterion,             device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d}/{max_epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}"
            )

        if stopper.step(val_loss, model):
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    stopper.restore_best(model)

    # ── 4f. Evaluation on test set ───────────────────────────────────────────
    y_pred   = _predict(model, test_loader, device)
    macro_f1 = f1_score(Y_test, y_pred, average="macro")
    accuracy = accuracy_score(Y_test, y_pred)
    report   = classification_report(
        Y_test, y_pred,
        target_names=[str(c) for c in le.classes_],
    )

    if verbose:
        print("\n── Test-set results ──────────────────────────────────────────")
        print(f"Macro F1 : {macro_f1:.4f}")
        print(report)

    return {
        "model":         model,
        "macro_f1":      macro_f1,
        "accuracy":      accuracy,
        "report":        report,
        "history":       history,
        "label_encoder": le,
        "epochs_trained":actual_epochs,   

    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. Smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng    = np.random.default_rng(0)
    X_demo = rng.standard_normal((1000, 115)).astype(np.float32)
    Y_demo = rng.integers(0, 4, size=1000)

    results = run_pipeline(X_demo, Y_demo)
    print(f"\nFinal Macro F1: {results['macro_f1']:.4f}")