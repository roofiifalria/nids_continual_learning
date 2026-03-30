from __future__ import annotations
"""
Streaming, memory-safe preprocessing for large NF_UQ_NIDS_V2-style CSVs (10–20 GB).

Key properties:
- Two-pass chunked processing (never loads the whole dataset in RAM):
  1) Scan to compute per-column min/max and mean; collect label counts.
  2) Transform in chunks (numeric coercion -> impute -> min-max scale) and write out.
- Streaming stratified split without train_test_split (keeps memory tiny).
- Uses float32 to reduce memory/IO.
- Supports Parquet output via a persistent ParquetWriter (recommended) or CSV.

USAGE (multi-class recommended):
  # Multi-class: use the attack category column (e.g., Attack)
  python preprocessing.py /path/to/NF_UQ_NIDS_V2 \
      --class_col Attack --save_dir preprocessed --chunksize 200000 --parquet

CSV (slower, larger) example:
  python preprocessing.py /path/to/NF_UQ_NIDS_V2 \
      --class_col Attack --save_dir preprocessed --chunksize 200000
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------- Global Variables ---------------------------
GLOBAL_FEATURES = [
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "FLOW_DURATION_MILLISECONDS", "TCP_FLAGS",
    "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT"
]

ATTACK_MAPPING = {
    "Benign": "Benign", "DDoS": "DDoS", "ddos": "DDoS",
    "DDOS attack-HOIC": "DDoS", "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS", "DoS": "DoS", "dos": "DoS",
    "DoS attacks-Hulk": "DoS", "DoS attacks-GoldenEye": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS", "DoS attacks-Slowloris": "DoS",
    "scanning": "Scanning", "Reconnaissance": "Reconnaissance",
    "xss": "XSS", "Brute Force -XSS": "XSS", "password": "Brute Force",
    "Brute Force": "Brute Force", "SSH-Bruteforce": "Brute Force",
    "FTP-BruteForce": "Brute Force", "Brute Force -Web": "Brute Force",
    "injection": "Injection", "Bot": "Botnet", "Infilteration": "Infiltration",
    "Backdoor": "Backdoor", "backdoor": "Backdoor", "Exploits": "Exploits",
    "Fuzzers": "Fuzzers", "Generic": "Generic", "mitm": "MITM",
    "ransomware": "Ransomware", "Shellcode": "Shellcode",
    "Analysis": "Analysis", "Theft": "Theft", "Worms": "Worms"
}
# --------------------------- End Global Variables ---------------------------


# --------------------------- I/O helpers ---------------------------

def list_csvs(path: str) -> List[str]:
    """Return a sorted list of CSV files from a directory or a single CSV path."""
    if os.path.isdir(path):
        files = sorted(
            [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".csv")]
        )
        if not files:
            raise FileNotFoundError(f"No CSV files found under directory: {path}")
        return files
    if os.path.isfile(path) and path.lower().endswith(".csv"):
        return [path]
    raise FileNotFoundError(f"Path {path} is not a CSV or directory of CSVs")


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized numeric coercion; leaves NaN on failures (we handle later)."""
    return df.apply(pd.to_numeric, errors="coerce")


# --------------------------- PASS 1 (scan) ---------------------------

def pass1_scan(
    files: List[str],
    class_col: str,
    chunksize: int
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, int], Dict[str, float]]:
    """
    First pass: compute global per-column min/max/mean (ignoring NaNs) and label counts.
    Returns:
      - minmax: {col: (min, max)} for non-constant, numeric columns
      - label_counts: {label_name: count}
      - means: {col: mean} (for NaN imputation during pass 2)
    """
    stats: Dict[str, Dict[str, float | int]] = {}
    label_counts: Dict[str, int] = defaultdict(int)

    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=True):
            # --- Tambahkan kode filtering di sini ---
            chunk[class_col] = chunk[class_col].map(ATTACK_MAPPING)
            chunk = chunk.dropna(subset=[class_col])
            columns_to_keep = GLOBAL_FEATURES + [class_col]
            chunk = chunk[columns_to_keep]
            # --- Akhir tambahan kode ---
            
            if class_col not in chunk.columns:
                raise KeyError(f"'{class_col}' not found in columns for {fp}")

            # label distribution from class_col (e.g., Attack)
            y = chunk[class_col].astype(str)
            vc = y.value_counts()
            for k, c in vc.items():
                label_counts[str(k)] += int(c)

            # numeric coercion
            X = chunk.drop(columns=[class_col])
            X = safe_numeric(X)
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # per-column stats (ignore NaNs)
            col_min = X.min(axis=0, skipna=True, numeric_only=True)
            col_max = X.max(axis=0, skipna=True, numeric_only=True)
            col_sum = X.sum(axis=0, skipna=True, numeric_only=True)
            col_cnt = X.count(axis=0, numeric_only=True)  # non-NaN counts

            for col in X.columns:
                cnt = int(col_cnt.get(col, 0))
                if cnt == 0:
                    continue  # all-NaN for this chunk
                cmin = float(col_min[col])
                cmax = float(col_max[col])
                csum = float(col_sum[col])

                s = stats.get(col)
                if s is None:
                    stats[col] = {"min": cmin, "max": cmax, "sum": csum, "cnt": cnt}
                else:
                    s["min"] = min(s["min"], cmin)
                    s["max"] = max(s["max"], cmax)
                    s["sum"] += csum
                    s["cnt"] += cnt

    # finalize: drop zero-variance columns and invalid stats
    minmax: Dict[str, Tuple[float, float]] = {}
    means: Dict[str, float] = {}
    for col, s in stats.items():
        cnt = int(s["cnt"])
        if cnt <= 0:
            continue
        mn = float(s["min"])
        mx = float(s["max"])
        if not np.isfinite(mn) or not np.isfinite(mx):
            continue
        if mn == mx:
            continue  # constant column -> drop
        minmax[col] = (mn, mx)
        means[col] = float(s["sum"]) / max(1, cnt)

    return minmax, dict(label_counts), means


# --------------------------- Parquet streaming ---------------------------

def _open_parquet_writers(train_path: str, test_path: str):
    """
    Prepare a container for lazily opening ParquetWriter objects once we know the schema.
    Returns a dict used by _parquet_append and closed at the end of pass 2.
    """
    return {"train": None, "test": None, "schema": None, "train_path": train_path, "test_path": test_path}


def _parquet_append(df: pd.DataFrame, split_flag: int, writers: dict):
    """
    Append a frame portion (split_flag: 0=train, 1=test) to the appropriate Parquet writer.
    Creates the writer lazily using the first batch's schema.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    df_part = df[df["__SPLIT__"] == split_flag].drop(columns=["__SPLIT__"])
    if df_part.empty:
        return
    table = pa.Table.from_pandas(df_part, preserve_index=False)

    if writers["schema"] is None:
        writers["schema"] = table.schema

    key = "train" if split_flag == 0 else "test"
    if writers[key] is None:
        path = writers["train_path"] if key == "train" else writers["test_path"]
        writers[key] = pq.ParquetWriter(path, writers["schema"], compression="zstd")

    writers[key].write_table(table)


# --------------------------- PASS 2 (transform & write) ---------------------------

def pass2_transform_and_write(
    files: List[str],
    class_col: str,
    minmax: Dict[str, Tuple[float, float]],
    means: Dict[str, float],
    label_counts: Dict[str, int],
    save_dir: str,
    chunksize: int,
    test_size: float,
    random_state: int,
    parquet: bool,
):
    os.makedirs(save_dir, exist_ok=True)
    out_train = os.path.join(save_dir, "train_df.parquet" if parquet else "train_df.csv")
    out_test = os.path.join(save_dir, "test_df.parquet" if parquet else "test_df.csv")

    # streaming stratified split targets
    def split_targets(label_counts: Dict[str, int], test_size: float):
        tgt = {}
        for lab, n in label_counts.items():
            n_test = int(round(n * test_size))
            tgt[lab] = [n - n_test, n_test]  # [train_remaining, test_remaining]
        return tgt

    targets = split_targets(label_counts, test_size)
    rng = np.random.default_rng(random_state)

    # stable class order for mapping
    classes_sorted = sorted(label_counts.keys())
    name_to_id = {name: i for i, name in enumerate(classes_sorted)}

    # parquet writers (lazy)
    writers = _open_parquet_writers(out_train, out_test) if parquet else None

    # only columns found valid in pass 1
    keep_cols = list(minmax.keys())

    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=True):
            # --- Tambahkan kode filtering di sini ---
            if class_col not in chunk.columns:
                continue
            
            chunk[class_col] = chunk[class_col].map(ATTACK_MAPPING)
            chunk = chunk.dropna(subset=[class_col])
            columns_to_keep = GLOBAL_FEATURES + [class_col]
            chunk = chunk[columns_to_keep]
            # --- Akhir tambahan kode ---

            y = chunk[class_col].astype(str)

            present = [c for c in keep_cols if c in chunk.columns]
            if not present:
                continue

            X = chunk[present].apply(pd.to_numeric, errors="coerce")
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # impute NaNs with pass-1 means (fallback 0.0 if somehow missing)
            for c in present:
                X[c] = X[c].fillna(means.get(c, 0.0))

            # min-max scale
            for c in present:
                mn, mx = minmax[c]
                denom = (mx - mn)
                if denom == 0:
                    X[c] = 0.0
                else:
                    X[c] = (X[c] - mn) / denom
                X[c] = X[c].astype(np.float32)

            df = pd.concat([X, y.rename(class_col)], axis=1)
            df["label"] = df[class_col].map(name_to_id).astype(np.int32)

            # streaming stratified assignment (0=train, 1=test)
            flags = []
            for lab in df[class_col].values:
                tr_rem, te_rem = targets[lab]
                p_test = te_rem / max(1, (tr_rem + te_rem))
                if (rng.random() < p_test) and te_rem > 0:
                    flags.append(1); targets[lab][1] -= 1
                else:
                    flags.append(0); targets[lab][0] -= 1
            df["__SPLIT__"] = np.array(flags, dtype=np.int8)

            if parquet:
                _parquet_append(df, 0, writers)  # train
                _parquet_append(df, 1, writers)  # test
            else:
                df_train = df[df["__SPLIT__"] == 0].drop(columns=["__SPLIT__"])
                df_test = df[df["__SPLIT__"] == 1].drop(columns=["__SPLIT__"])
                if not df_train.empty:
                    df_train.to_csv(out_train, index=False, mode="a", header=not os.path.exists(out_train))
                if not df_test.empty:
                    df_test.to_csv(out_test, index=False, mode="a", header=not os.path.exists(out_test))

    # close parquet writers
    if parquet and writers is not None:
        for key in ("train", "test"):
            if writers[key] is not None:
                writers[key].close()

    # save label names (compatible with existing loader)
    with open(os.path.join(save_dir, "label_dict.json"), "w", encoding="utf-8") as jf:
        json.dump({k: k for k in classes_sorted}, jf, indent=2)


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Memory-safe, streaming preprocessor for large NIDS CSVs")
    ap.add_argument("dataset_path", type=str, help="Path to a CSV file or a directory of CSVs")
    ap.add_argument("--label_col", type=str, default="Label",
                    help="Fallback label column if --class_col is not provided")
    ap.add_argument("--class_col", type=str, default=None,
                    help="Multi-class label column (e.g., 'Attack'). If set, overrides --label_col.")
    ap.add_argument("--save_dir", type=str, default="preprocessed", help="Output directory")
    ap.add_argument("--chunksize", type=int, default=200000, help="Rows per chunk to process")
    ap.add_argument("--test_size", type=float, default=0.2, help="Proportion of rows to assign to test split")
    ap.add_argument("--random_state", type=int, default=42, help="RNG seed for streaming stratified split")
    ap.add_argument("--parquet", action="store_true", help="Write Parquet instead of CSV (faster/smaller)")
    args = ap.parse_args()

    # Choose which column is the target (prefer class_col if given)
    target_col = args.class_col or args.label_col

    files = list_csvs(args.dataset_path)
    print(f"[pass1] scanning {len(files)} files with chunksize={args.chunksize} ...")
    minmax, label_counts, means = pass1_scan(files, target_col, args.chunksize)
    print(f"[pass1] features kept: {len(minmax)} | classes: {len(label_counts)}")

    if len(minmax) == 0:
        print("[warning] No numeric, non-constant columns detected. Check your data and target column.")
        # we still proceed to create empty outputs, but this likely indicates a column mismatch

    print("[pass2] transforming and writing outputs ...")
    pass2_transform_and_write(
        files=files,
        class_col=target_col,
        minmax=minmax,
        means=means,
        label_counts=label_counts,
        save_dir=args.save_dir,
        chunksize=args.chunksize,
        test_size=args.test_size,
        random_state=args.random_state,
        parquet=args.parquet,
    )
    print(f"[done] wrote preprocessed data under {args.save_dir}")


if __name__ == "__main__":
    main()