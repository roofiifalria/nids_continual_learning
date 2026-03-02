import sys
import torch
import os
import pandas as pd
import numpy as np

# Tambahkan folder 'src' ke path
sys.path.append('src')

from models.mlp_classifier import MLPClassifierTorch, MLPConfig
from data.dataloader import get_data

# --- KONFIGURASI ---
# Pastikan path ini mengarah ke folder output terbaru Anda
run_folder = 'outputs/20260127_161636' 
ckpt_path = os.path.join(run_folder, 'checkpoints', 'mlp_final.pt')
data_root = '.' 

# 1. Load Model
print(f"Loading model dari {ckpt_path}...")
if not os.path.exists(ckpt_path):
    print("Error: File checkpoint tidak ditemukan.")
    sys.exit(1)

checkpoint = torch.load(ckpt_path)
config = MLPConfig(**checkpoint['cfg'])
clf = MLPClassifierTorch(config)
clf.model.load_state_dict(checkpoint['state_dict'])
clf.model.eval()

print(f"Model dikonfigurasi untuk menerima {config.input_dim} fitur.")

# 2. Load Data Test
print("Loading data test (ini mungkin memakan waktu untuk 15 juta baris)...")
test_path = os.path.join(data_root, 'preprocessed', 'test_df.parquet')

if os.path.exists(test_path):
    test_df = pd.read_parquet(test_path)
else:
    print("File test_df.parquet tidak ditemukan, mencoba csv...")
    test_df = pd.read_csv(os.path.join(data_root, 'preprocessed', 'test_df.csv'))

# 3. Persiapkan Fitur (X) dan Label (y)
# PERBAIKAN: Hanya buang 'label' (id numerik) dan 'Attack' (string class). 
# Jangan buang 'Label' (binary) jika itu dianggap fitur oleh model saat training.
exclude_cols = ['label', 'Attack'] 
feature_cols = [c for c in test_df.columns if c not in exclude_cols]

# Validasi Dimensi
if len(feature_cols) != config.input_dim:
    print(f"PERINGATAN: Jumlah fitur data ({len(feature_cols)}) tidak sama dengan model ({config.input_dim}).")
    print("Mencoba menyesuaikan otomatis...")
    # Coba logika alternatif: ambil semua numerik kecuali 'label'
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != 'label']
    
    if len(feature_cols) != config.input_dim:
        print("ERROR FATAL: Tidak bisa mencocokkan jumlah fitur. Cek preprocessing Anda.")
        print(f"Fitur di Data ({len(feature_cols)}): {feature_cols[:5]} ...")
        sys.exit(1)
    else:
        print("Penyesuaian berhasil!")

print(f"Menggunakan {len(feature_cols)} fitur.")
print(f"Jumlah data test: {len(test_df)}")

# Konversi ke Array (Batching manual jika memori tidak cukup)
X_test = test_df[feature_cols].values.astype(np.float32)
y_test = test_df['label'].values.astype(int)

# 4. Hitung Metrik
# Karena data sangat besar (15 juta), evaluate() sekaligus bisa bikin Memory Error (OOM).
# Kita lakukan batching sederhana untuk evaluasi.

print("Menghitung akurasi (dengan batching)...")
batch_size = 50000
total_samples = len(y_test)
total_loss = 0.0
all_preds = []
all_targets = []

# Gunakan fungsi internal model jika memungkinkan, atau loop manual
# Di sini kita loop manual agar aman memori
criterion = torch.nn.CrossEntropyLoss()
clf.model.eval()

with torch.no_grad():
    for i in range(0, total_samples, batch_size):
        end = min(i + batch_size, total_samples)
        X_batch = torch.from_numpy(X_test[i:end]).to(clf.device)
        y_batch = torch.from_numpy(y_test[i:end]).long().to(clf.device)
        
        logits = clf.model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * (end - i)
        
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        
        # Progress bar sederhana
        if i % (batch_size * 10) == 0:
            print(f"Processed {i}/{total_samples} samples...")

avg_loss = total_loss / total_samples
all_preds = np.array(all_preds)

# Hitung akurasi manual
acc = (all_preds == y_test).mean()
# Hitung Macro-F1 manual (menggunakan helper di class MLP atau sklearn jika ada)
# Kita pakai cara simple sklearn untuk report akhir
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test, all_preds, average='macro')

print("-" * 30)
print("HASIL EVALUASI AKHIR")
print("-" * 30)
print(f"Loss      : {avg_loss:.4f}")
print(f"Akurasi   : {acc:.2%}")
print(f"Macro-F1  : {macro_f1:.4f}")
print("-" * 30)