from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Tuple, Optional, List

import numpy as np
import pandas as pd


class ClassReplayBuffer:
    """
    Per-class replay buffer for continual learning / fine-tuning.

    - Stores feature vectors (float32) grouped by integer class id.
    - Fixed `capacity_per_class` via ring buffers (deque with maxlen).
    - NumPy I/O for smooth interop with PyTorch.
    - RL-friendly samplers (exact-size with replacement, weighted mix).

    Public API (backwards compatible with your envs):
      add(x, y)
      add_iter(pairs)
      sample_per_class(class_id, k)
      sample_mixed(total_k)
      sample_dict(per_class)

    New helpers (for hybrid env/RL):
      sample_per_class_exact(class_id, k, replace_if_needed=True)
      sample_dict_exact(per_class, replace_if_needed=True)
      sample_weighted(total_k, class_weights=None)
      seed_from_frames(df, feature_cols, max_rows=None)
      available_classes(), has_class(c), set_seed(seed)
    """

    def __init__(
        self,
        n_classes: int,
        capacity_per_class: int = 10000,
        feature_dim: Optional[int] = None,
        seed: Optional[int] = 42,
    ) -> None:
        self.n_classes = int(n_classes)
        self.capacity_per_class = int(capacity_per_class)
        self.feature_dim = None if feature_dim is None else int(feature_dim)
        self.rng = np.random.default_rng(seed)
        self.buffers: Dict[int, deque] = {
            c: deque(maxlen=self.capacity_per_class) for c in range(self.n_classes)
        }

    # ------------------------------ config ------------------------------
    def set_seed(self, seed: Optional[int]) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------ add ---------------------------------
    def add(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Add a batch of samples to their per-class buffers.

        Shapes:
            x: (N, F) -> coerced to float32
            y: (N,)   -> coerced to int
        """
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (N,F), got shape={x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must share the same first dimension")
        if self.feature_dim is None:
            self.feature_dim = int(x.shape[1])
        elif x.shape[1] != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: buffer={self.feature_dim}, x={x.shape[1]}")

        x = x.astype(np.float32, copy=False)
        for xi, yi in zip(x, y):
            c = int(yi)
            if 0 <= c < self.n_classes:
                self.buffers[c].append(xi.copy())

    def add_iter(self, pairs: Iterable[Tuple[np.ndarray, int]]) -> None:
        """Add samples from an iterator of (x, y)."""
        for xi, yi in pairs:
            xi = np.asarray(xi)
            if xi.ndim != 1:
                raise ValueError("add_iter expects xi as 1D feature vector")
            c = int(yi)
            if 0 <= c < self.n_classes:
                if self.feature_dim is None:
                    self.feature_dim = int(xi.shape[0])
                elif xi.shape[0] != self.feature_dim:
                    raise ValueError(f"feature_dim mismatch: buffer={self.feature_dim}, xi={xi.shape[0]}")
                self.buffers[c].append(xi.astype(np.float32, copy=False).copy())

    def seed_from_frames(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        max_rows: Optional[int] = None,
    ) -> int:
        """
        Convenience: add up to max_rows from a pandas frame with given feature/label columns.
        Returns number of rows added.
        """
        if max_rows is None:
            view = df
        else:
            view = df.iloc[: int(max_rows)]
        x = view[feature_cols].values
        y = view[label_col].values
        before = self.size()
        self.add(x, y)
        return self.size() - before

    # ------------------------------ sample ------------------------------
    def sample_per_class(self, class_id: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample up to k items from a single class buffer (no replacement).
        Returns (X, y); may be empty if buffer has no items.
        """
        c = int(class_id)
        buf = self.buffers.get(c, None)
        if buf is None or len(buf) == 0 or k <= 0:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        k = int(min(k, len(buf)))
        idx = self.rng.choice(len(buf), size=k, replace=False)
        X = np.stack([buf[i] for i in idx], axis=0)
        y = np.full((k,), c, dtype=int)
        return X, y

    def sample_per_class_exact(
        self, class_id: int, k: int, replace_if_needed: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample exactly k items from a class.
        If the buffer has fewer than k samples and replace_if_needed=True,
        sample with replacement to meet k.
        """
        c = int(class_id)
        buf = self.buffers.get(c, None)
        F = (self.feature_dim or 0)
        if buf is None or k <= 0:
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        n = len(buf)
        if n == 0:
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)

        if k <= n or not replace_if_needed:
            idx = self.rng.choice(n, size=min(k, n), replace=False)
        else:
            # need replacement to reach k
            idx = self.rng.choice(n, size=k, replace=True)

        X = np.stack([buf[i] for i in idx], axis=0)
        y = np.full((len(idx),), c, dtype=int)
        return X, y

    def sample_mixed(self, total_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a mixed batch across classes with (roughly) equal share per non-empty class.
        Returns (X, y); may be empty if all buffers are empty.
        """
        total_k = int(total_k)
        if total_k <= 0:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        non_empty = [c for c in range(self.n_classes) if len(self.buffers[c]) > 0]
        if not non_empty:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)

        share = max(1, total_k // len(non_empty))
        X_parts, y_parts = [], []
        collected = 0
        for c in non_empty:
            need = min(share, total_k - collected)
            if need <= 0:
                break
            Xc, yc = self.sample_per_class(c, need)
            if Xc.size:
                X_parts.append(Xc)
                y_parts.append(yc)
                collected += Xc.shape[0]

        if collected < total_k:
            # top-up randomly from whatever is available
            while collected < total_k:
                c = int(self.rng.choice(non_empty))
                Xc, yc = self.sample_per_class(c, 1)
                if Xc.size:
                    X_parts.append(Xc)
                    y_parts.append(yc)
                    collected += 1
                else:
                    # remove empty class from candidates
                    non_empty = [cc for cc in non_empty if len(self.buffers[cc]) > 0]
                    if not non_empty:
                        break

        if not X_parts:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        return X, y

    def sample_weighted(
        self,
        total_k: int,
        class_weights: Optional[Dict[int, float]] = None,
        replace_if_needed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted mixed sampling. Provide a dict of class->weight (unnormalized).
        If None, weights are inversely proportional to current buffer sizes
        (i.e., up-weight minority / unseen classes).
        """
        total_k = int(total_k)
        F = (self.feature_dim or 0)
        if total_k <= 0:
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)

        avail = [c for c in range(self.n_classes) if len(self.buffers[c]) > 0]
        if not avail:
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)

        if class_weights is None:
            # inverse frequency weighting
            sizes = np.array([len(self.buffers[c]) for c in avail], dtype=np.float32)
            with np.errstate(divide="ignore"):
                w = 1.0 / np.maximum(sizes, 1.0)
            w = w / w.sum()
        else:
            w_list, cs = [], []
            for c in avail:
                w_list.append(float(class_weights.get(c, 0.0)))
                cs.append(c)
            w = np.array(w_list, dtype=np.float32)
            if w.sum() <= 0:
                # fallback: uniform among available
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            avail = cs

        # decide how many per class (multinomial draw)
        draws = self.rng.multinomial(total_k, w)
        X_parts, y_parts = [], []
        for c, k in zip(avail, draws):
            if k <= 0:
                continue
            Xc, yc = self.sample_per_class_exact(c, int(k), replace_if_needed=replace_if_needed)
            if Xc.size:
                X_parts.append(Xc)
                y_parts.append(yc)

        if not X_parts:
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        return np.vstack(X_parts), np.concatenate(y_parts)

    def sample_dict(self, per_class: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Sample up to k per class (no replacement)."""
        X_parts, y_parts = [], []
        for c, k in per_class.items():
            Xc, yc = self.sample_per_class(int(c), int(k))
            if Xc.size:
                X_parts.append(Xc)
                y_parts.append(yc)
        if not X_parts:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        return np.vstack(X_parts), np.concatenate(y_parts)

    def sample_dict_exact(
        self,
        per_class: Dict[int, int],
        replace_if_needed: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample exactly k per class (using replacement when needed)."""
        X_parts, y_parts = [], []
        for c, k in per_class.items():
            Xc, yc = self.sample_per_class_exact(int(c), int(k), replace_if_needed=replace_if_needed)
            if Xc.size:
                X_parts.append(Xc)
                y_parts.append(yc)
        if not X_parts:
            F = (self.feature_dim or 0)
            return np.empty((0, F), dtype=np.float32), np.empty((0,), dtype=int)
        return np.vstack(X_parts), np.concatenate(y_parts)

    # ------------------------------ stats/io ---------------------------
    def size(self) -> int:
        return sum(len(self.buffers[c]) for c in range(self.n_classes))

    def size_per_class(self) -> Dict[int, int]:
        return {c: len(self.buffers[c]) for c in range(self.n_classes)}

    def fill_ratio(self) -> Dict[int, float]:
        return {c: len(self.buffers[c]) / float(self.capacity_per_class) for c in range(self.n_classes)}

    def has_class(self, c: int) -> bool:
        return len(self.buffers.get(int(c), ())) > 0

    def available_classes(self) -> List[int]:
        return [c for c in range(self.n_classes) if len(self.buffers[c]) > 0]

    def save_npz(self, path: str) -> None:
        # Flatten to arrays per class for portability
        arrays = {}
        for c in range(self.n_classes):
            if len(self.buffers[c]) == 0:
                arrays[f"c{c}"] = np.empty((0, self.feature_dim or 0), dtype=np.float32)
            else:
                arrays[f"c{c}"] = np.stack(list(self.buffers[c]), axis=0)
        np.savez_compressed(path, **arrays, n_classes=self.n_classes, feature_dim=(self.feature_dim or 0))

    def load_npz(self, path: str) -> None:
        data = np.load(path, allow_pickle=False)
        self.n_classes = int(data["n_classes"]) if "n_classes" in data else self.n_classes
        self.feature_dim = int(data["feature_dim"]) if "feature_dim" in data else self.feature_dim
        self.buffers = {c: deque(maxlen=self.capacity_per_class) for c in range(self.n_classes)}
        for c in range(self.n_classes):
            key = f"c{c}"
            if key in data:
                arr = np.asarray(data[key])
                for row in arr:
                    self.buffers[c].append(np.asarray(row, dtype=np.float32, copy=False).copy())
