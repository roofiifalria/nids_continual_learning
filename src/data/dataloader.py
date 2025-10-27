from __future__ import annotations

import json
import os
from typing import Tuple, Dict, Optional

import pandas as pd


# ==============================================================
# Dataloader that reads the artifacts written by preprocessing.py
# - preprocessed/train_df.parquet (preferred) OR preprocessed/train_df.csv
# - preprocessed/test_df.parquet  (preferred) OR preprocessed/test_df.csv
# - preprocessed/label_dict.json  (list/dict of class names)
#
# Returns:
#   train_df, test_df (numeric feature columns + string class col if present + int 'label')
#   name_to_dense: Dict[str, int]  (class name -> dense id)
#
# Notes:
# - If you used the streaming preprocessor with --class_col (e.g., Attack),
#   this loader will map that column to the integer 'label' using label_dict.json.
# - If a string class column is not provided, but 'label' int exists already,
#   it will keep 'label' and derive a best-effort mapping from label_dict.json.
# ==============================================================


def _read_frames(pre_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Read preprocessed train/test as Parquet if present, else CSV. Return (train, test, is_parquet)."""
    train_pq = os.path.join(pre_dir, "train_df.parquet")
    test_pq = os.path.join(pre_dir, "test_df.parquet")
    train_csv = os.path.join(pre_dir, "train_df.csv")
    test_csv = os.path.join(pre_dir, "test_df.csv")

    if os.path.exists(train_pq) and os.path.exists(test_pq):
        return pd.read_parquet(train_pq), pd.read_parquet(test_pq), True
    if os.path.exists(train_csv) and os.path.exists(test_csv):
        return pd.read_csv(train_csv), pd.read_csv(test_csv), False

    raise FileNotFoundError(
        f"Missing preprocessed files under {pre_dir}. "
        f"Expected train_df.(parquet|csv) and test_df.(parquet|csv)."
    )


def _load_label_names(pre_dir: str) -> Optional[list[str]]:
    """Load class name list from label_dict.json if present."""
    p = os.path.join(pre_dir, "label_dict.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as jf:
            data = json.load(jf)
        # The preprocessor stored {name: name}; we only need the keys in stable order
        return list(data.keys())
    return None


def get_data(
    data_folder: str = "./",
    *,
    label_col: str = "Label",
    class_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Load preprocessed splits and return (train_df, test_df, name_to_dense).

    Args:
      data_folder: Project root containing 'preprocessed/'.
      label_col:   Fallback binary label column name (e.g., 'Label') if class_col is not provided.
      class_col:   Preferred multi-class column (e.g., 'Attack'). If provided and present, overrides label_col.

    Behavior:
      - If class_col is present in both frames, map it -> dense 'label' using label_dict.json (if present).
      - Else if label_col is present, map it -> dense 'label'.
      - Else if an integer 'label' column already exists, keep it and derive mapping from label_dict.json when available.
    """
    pre = os.path.join(data_folder, "preprocessed")
    train, test, _is_parquet = _read_frames(pre)
    label_names = _load_label_names(pre)

    # Decide which column will serve as the string target for mapping.
    # Prefer class_col (multi-class), then label_col (e.g., "Label"), else None.
    target_col = None
    if class_col and (class_col in train.columns) and (class_col in test.columns):
        target_col = class_col
    elif (label_col in train.columns) and (label_col in test.columns):
        target_col = label_col

    # Build name->id mapping.
    if label_names:
        # Use names from label_dict.json (saved by preprocessor)
        # Keep deterministic order (as written), else fall back to sorted
        classes_ordered = label_names
    else:
        # Fall back: infer from data (union of target_col values if available; else from existing 'label' ints)
        if target_col is not None:
            classes_ordered = sorted(pd.unique(pd.concat([train[target_col].astype(str),
                                                          test[target_col].astype(str)], axis=0)))
        elif "label" in train.columns and "label" in test.columns:
            # We only have integer ids; create stringified names for a consistent mapping
            all_ids = sorted(pd.unique(pd.concat([train["label"], test["label"]], axis=0).astype(int)))
            classes_ordered = [str(i) for i in all_ids]
        else:
            raise ValueError(
                "Could not determine class names: provide class_col/label_col or ensure 'label' exists."
            )

    name_to_dense: Dict[str, int] = {name: i for i, name in enumerate(classes_ordered)}

    # Ensure integer 'label' column exists and is consistent with mapping.
    if target_col is not None:
        train["label"] = train[target_col].astype(str).map(name_to_dense).astype(int)
        test["label"] = test[target_col].astype(str).map(name_to_dense).astype(int)
    else:
        # No string target_col; keep existing 'label' if present
        if "label" not in train.columns or "label" not in test.columns:
            raise ValueError(
                "No target string column and no 'label' column found. "
                "Re-run preprocessing with --class_col (e.g., Attack) or --label_col."
            )
        # At this point, 'label' exists. We can't reconstruct names without label_dict.json,
        # but we already handled that via classes_ordered above.

        # Make sure dtype is int
        train["label"] = train["label"].astype(int)
        test["label"] = test["label"].astype(int)

    return train, test, name_to_dense


if __name__ == "__main__":
    tr, te, mapping = get_data("./", class_col="Attack")  # example: prefer multi-class 'Attack'
    print(tr.shape, te.shape)
    print("classes:", mapping)
