from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ==============================================================
# Torch MLP classifier with a scikit-like API
# - fit(X, y, epochs=..., batch_size=..., verbose)
# - partial_fit(X, y, epochs=1, batch_size=..., verbose=0)
# - predict(X) -> np.ndarray[int]
# - predict_proba(X) -> np.ndarray[float32]
# - evaluate(X, y) -> {"loss": float, "acc": float, "f1": float}
# ==============================================================


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(256, 128), num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MLPConfig:
    input_dim: int
    num_classes: int
    hidden_sizes: tuple = (256, 128)
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: Optional[int] = 42
    grad_clip: Optional[float] = 1.0              # clip global grad-norm if not None
    class_weight: Optional[object] = None         # None | "balanced" | Dict[int,float]


class MLPClassifierTorch:
    """Incremental MLP for multi-class classification (CrossEntropy).

    Compatible with envs that call `.partial_fit(X, y)` and `.predict(X)`.
    """

    def __init__(self, config: MLPConfig):
        if config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
        self.cfg = config
        self.device = _device()
        self.model = _MLP(config.input_dim, config.hidden_sizes, config.num_classes, config.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self._set_criterion(class_weight=config.class_weight, num_classes=config.num_classes)
        self._classes = np.arange(config.num_classes, dtype=np.int64)

    # ------------------------------- training API -------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 256, verbose: int = 1):
        """Full/bulk training (can be called multiple times)."""
        self.model.train()
        dl = self._make_loader(X, y, batch_size, shuffle=True)
        for e in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total = 0
            for xb, yb in dl:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.cfg.grad_clip))
                self.optimizer.step()

                total_loss += float(loss.detach().cpu().item()) * yb.size(0)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == yb).sum().item())
                total += yb.size(0)
            if verbose:
                print(f"[MLP] epoch {e+1:03d}/{epochs}  loss={total_loss/total:.4f}  acc={total_correct/total:.4f}")
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1, batch_size: int = 256, verbose: int = 0):
        """Incremental update: one or more epochs over the provided batch."""
        return self.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose)

    # ------------------------------- inference API ------------------------------
    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self._as_tensor(X)
        logits = self.model(X)
        preds = logits.argmax(dim=1)
        return preds.cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X = self._as_tensor(X)
        logits = self.model(X)
        prob = F.softmax(logits, dim=1)
        return prob.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_topk(self, X: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Return (topk_indices, topk_probs) for analysis/diagnostics."""
        self.model.eval()
        X = self._as_tensor(X)
        logits = self.model(X)
        prob = F.softmax(logits, dim=1)
        topk_prob, topk_idx = torch.topk(prob, k=min(k, prob.shape[1]), dim=1)
        return topk_idx.cpu().numpy().astype(np.int64), topk_prob.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Return loss/acc/macro-F1. No sklearn dependency."""
        self.model.eval()
        X_t = self._as_tensor(X)
        y_t = torch.from_numpy(np.asarray(y).reshape(-1)).long().to(self.device)
        logits = self.model(X_t)
        loss = self.criterion(logits, y_t)
        preds = logits.argmax(dim=1)

        acc = float((preds == y_t).float().mean().cpu().item())
        f1 = self._macro_f1(preds.cpu().numpy(), y_t.cpu().numpy(), num_classes=self.cfg.num_classes)
        return {"loss": float(loss.detach().cpu().item()), "acc": acc, "f1": f1}

    # ------------------------------- io -----------------------------------------
    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        # Optional: verify shapes; for now we assume same architecture
        self.model.load_state_dict(ckpt["state_dict"])

    # ------------------------------- helpers ------------------------------------
    def _as_tensor(self, X: np.ndarray) -> torch.Tensor:
        X = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        return X

    def _make_loader(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y = torch.from_numpy(np.asarray(y).reshape(-1)).long()
        ds = TensorDataset(X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def _set_criterion(self, class_weight: Optional[object], num_classes: int):
        """class_weight: None | 'balanced' | Dict[int, float]"""
        if class_weight is None:
            self.criterion = nn.CrossEntropyLoss()
            return
        if isinstance(class_weight, dict):
            # dict[int,float] -> tensor
            w = torch.zeros(num_classes, dtype=torch.float32)
            for k, v in class_weight.items():
                if 0 <= int(k) < num_classes:
                    w[int(k)] = float(v)
            self.criterion = nn.CrossEntropyLoss(weight=w.to(self.device))
            return
        if isinstance(class_weight, str) and class_weight.lower() == "balanced":
            # 'balanced' will be computed per batch in fit() via inverse class frequency if needed.
            # Simpler: keep static equal weights and rely on replay/gan balancing in env;
            # or set uniform weights to 1. We choose uniform here but expose hook for future dynamic weights.
            self.criterion = nn.CrossEntropyLoss()
            return
        # fallback
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def _macro_f1(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int) -> float:
        """
        Compute macro-F1 without sklearn. y_pred/y_true are int arrays.
        """
        y_pred = y_pred.astype(np.int64).ravel()
        y_true = y_true.astype(np.int64).ravel()
        f1s = []
        for c in range(num_classes):
            tp = np.sum((y_true == c) & (y_pred == c)).astype(np.int64)
            fp = np.sum((y_true != c) & (y_pred == c)).astype(np.int64)
            fn = np.sum((y_true == c) & (y_pred != c)).astype(np.int64)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        return float(np.mean(f1s)) if len(f1s) else 0.0
