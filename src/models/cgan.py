from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


# ==============================================================
# Conditional GAN (PyTorch) for tabular, min-max scaled [0,1] features
# - generate(labels_int: np.ndarray, per_label: int) -> np.ndarray
# - train(X: np.ndarray, y_int: np.ndarray, epochs: int, ...) -> None
# - save(path) / load(path)
# ==============================================================


def _get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class _MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        layers: int,
        out_activation: Optional[str] = None,
    ):
        super().__init__()
        seq = []
        d = in_dim
        for _ in range(max(0, layers - 1)):
            seq += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)
        self.out_activation = out_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.out_activation == "tanh":
            x = torch.tanh(x)
        return x


@dataclass
class CGANConfig:
    x_dim: int
    num_classes: int
    noise_dim: int = 32
    min_units: int = 128
    g_layers: int = 3
    d_layers: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    beta1: float = 0.5
    beta2: float = 0.999
    label_smooth: float = 0.9
    clamp_outputs: bool = True          # clamp generated outputs to [0,1]
    balance_sampling: bool = True       # use WeightedRandomSampler if class imbalance
    seed: Optional[int] = 42


class ConditionalGANTorch:
    """
    Fully-connected CGAN with label one-hot conditioning.

    Discriminator takes concat([x, y_onehot]).
    Generator     takes concat([z, y_onehot]).

    Notes:
    - Features are assumed scaled to [0,1]. We clamp generator outputs to [0,1]
      at inference (and can also clamp in train forward to stabilize PPO usage).
    """

    def __init__(
        self,
        x_dim: int,
        num_classes: int,
        noise_dim: int = 32,
        min_units: int = 128,
        g_layers: int = 3,
        d_layers: int = 3,
        batch_size: int = 128,
        lr: float = 1e-3,
        beta1: float = 0.5,
        beta2: float = 0.999,
        label_smooth: float = 0.9,
        clamp_outputs: bool = True,
        balance_sampling: bool = True,
        device: Optional[torch.device] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.cfg = CGANConfig(
            x_dim=int(x_dim),
            num_classes=int(num_classes),
            noise_dim=int(noise_dim),
            min_units=int(min_units),
            g_layers=int(g_layers),
            d_layers=int(d_layers),
            batch_size=int(batch_size),
            lr=float(lr),
            beta1=float(beta1),
            beta2=float(beta2),
            label_smooth=float(label_smooth),
            clamp_outputs=bool(clamp_outputs),
            balance_sampling=bool(balance_sampling),
            seed=seed,
        )
        self.device = device or _get_device()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Models
        self.G = _MLP(
            self.cfg.noise_dim + self.cfg.num_classes,
            self.cfg.x_dim,
            self.cfg.min_units,
            self.cfg.g_layers,
            out_activation=None,
        ).to(self.device)
        self.D = _MLP(
            self.cfg.x_dim + self.cfg.num_classes,
            1,
            self.cfg.min_units,
            self.cfg.d_layers,
            out_activation=None,
        ).to(self.device)

        # Optims
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.cfg.lr, betas=(self.cfg.beta1, self.cfg.beta2))

        # Loss
        self.bce = nn.BCEWithLogitsLoss()

    # ----------------------------- utils -----------------------------
    def _one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels.long(), num_classes=self.cfg.num_classes).float()

    def _sample_noise(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.cfg.noise_dim, device=self.device)

    # ----------------------------- API ------------------------------
    @torch.no_grad()
    def generate(self, labels_int: np.ndarray, per_label: int = 1) -> np.ndarray:
        """
        Generate samples conditioned on integer class labels.
        labels_int: shape [K] of class ids
        per_label:  number of samples per provided label id
        returns:    array [K*per_label, x_dim]
        """
        self.G.eval()
        labels = np.asarray(labels_int).reshape(-1)
        if labels.size == 0 or per_label <= 0:
            return np.empty((0, self.cfg.x_dim), dtype=np.float32)

        labels_rep = np.repeat(labels, per_label)
        y = torch.from_numpy(labels_rep.astype(np.int64)).to(self.device)
        y_oh = self._one_hot(y)
        z = self._sample_noise(y.shape[0])
        zg = torch.cat([z, y_oh], dim=1)
        x_fake = self.G(zg)
        if self.cfg.clamp_outputs:
            x_fake = torch.clamp(x_fake, 0.0, 1.0)
        return x_fake.detach().cpu().numpy().astype(np.float32)

    def train(
        self,
        X: np.ndarray,
        y_int: np.ndarray,
        epochs: int = 50,
        d_steps: int = 1,
        g_steps: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Fit CGAN on provided data. Assumes X is already min-max scaled to [0,1].
        """
        self.G.train()
        self.D.train()

        X = np.asarray(X, dtype=np.float32)
        y_int = np.asarray(y_int, dtype=np.int64)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y_int)

        ds = TensorDataset(X_t, y_t)

        # Optional class-balanced sampling
        if self.cfg.balance_sampling:
            # Compute inverse-frequency weights
            unique, counts = np.unique(y_int, return_counts=True)
            freq = {u: c for u, c in zip(unique, counts)}
            weights = np.array([1.0 / max(1, freq[int(lbl)]) for lbl in y_int], dtype=np.float32)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            dl = DataLoader(ds, batch_size=self.cfg.batch_size, sampler=sampler, drop_last=True)
        else:
            dl = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

        for e in range(epochs):
            loss_D_running = 0.0
            loss_G_running = 0.0
            n_batches = 0

            for xb, yb in dl:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                y_oh = self._one_hot(yb)

                # -------------------- Train D --------------------
                for _ in range(d_steps):
                    pred_real = self.D(torch.cat([xb, y_oh], dim=1))
                    target_real = torch.full_like(pred_real, self.cfg.label_smooth)
                    loss_real = self.bce(pred_real, target_real)

                    z = self._sample_noise(xb.shape[0])
                    xg = self.G(torch.cat([z, y_oh], dim=1)).detach()
                    if self.cfg.clamp_outputs:
                        xg = torch.clamp(xg, 0.0, 1.0)
                    pred_fake = self.D(torch.cat([xg, y_oh], dim=1))
                    target_fake = torch.zeros_like(pred_fake)
                    loss_fake = self.bce(pred_fake, target_fake)

                    loss_D = loss_real + loss_fake
                    self.opt_D.zero_grad(set_to_none=True)
                    loss_D.backward()
                    self.opt_D.step()
                    loss_D_running += float(loss_D.detach().cpu())
                # -------------------- Train G --------------------
                for _ in range(g_steps):
                    z = self._sample_noise(xb.shape[0])
                    xg = self.G(torch.cat([z, y_oh], dim=1))
                    if self.cfg.clamp_outputs:
                        xg = torch.clamp(xg, 0.0, 1.0)
                    pred = self.D(torch.cat([xg, y_oh], dim=1))
                    target = torch.ones_like(pred)
                    loss_G = self.bce(pred, target)

                    self.opt_G.zero_grad(set_to_none=True)
                    loss_G.backward()
                    self.opt_G.step()
                    loss_G_running += float(loss_G.detach().cpu())

                n_batches += 1

            if verbose and n_batches > 0 and (e + 1) % max(1, epochs // 10) == 0:
                avg_d = loss_D_running / max(1, n_batches * d_steps)
                avg_g = loss_G_running / max(1, n_batches * g_steps)
                print(f"[CGAN] epoch {e+1:03d}/{epochs}  D={avg_d:.4f}  G={avg_g:.4f}")

    # ----------------------------- IO -------------------------------
    def save(self, path: str, with_optim: bool = False) -> None:
        payload = {
            "G": self.G.state_dict(),
            "D": self.D.state_dict(),
            "cfg": asdict(self.cfg),
        }
        if with_optim:
            payload["opt_G"] = self.opt_G.state_dict()
            payload["opt_D"] = self.opt_D.state_dict()
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None, load_optim: bool = False) -> None:
        ckpt = torch.load(path, map_location=map_location or self.device)
        cfg = ckpt.get("cfg", None)
        if cfg is not None:
            # If config differs, you could re-init; here we assume shapes are unchanged
            pass
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.G.to(self.device)
        self.D.to(self.device)
        if load_optim:
            if "opt_G" in ckpt:
                self.opt_G.load_state_dict(ckpt["opt_G"])
            if "opt_D" in ckpt:
                self.opt_D.load_state_dict(ckpt["opt_D"])
