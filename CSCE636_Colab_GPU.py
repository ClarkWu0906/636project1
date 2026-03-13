"""
CSCE 636 Project 1 - Mixture-of-Experts DNN for m-Height Estimation
===================================================================
Optimized for Google Colab GPU. Upload this file + data files to Colab.

Usage on Colab:
  1. Upload this .py file and the two data pickle files to /content/
  2. Run: !python CSCE636_Colab_GPU.py
  OR
  3. Copy-paste into a Colab notebook cell and run.

Outputs:
  - moe_model.pth          (saved model checkpoint)
  - predicted_mHeights      (pickle file with predictions)
  - training_curves.png     (loss plots)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import Counter, defaultdict
import os
import time
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FORCE GPU — will error out if no GPU available so you know
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️  No CUDA GPU detected! Make sure you enabled GPU in Colab:")
    print("   Runtime → Change runtime type → Hardware accelerator → GPU")
    print("   Falling back to CPU (will be slow)...")
    device = torch.device("cpu")

# Enable cuDNN benchmarking for faster training on GPU
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# ============================================================
# Section 1: Load Data
# ============================================================
x_file_path = 'CSCE-636-Project-1-Train-n_k_m_P'
y_file_path = 'CSCE-636-Project-1-Train-mHeights'

with open(x_file_path, 'rb') as f:
    X_raw = pickle.load(f)
with open(y_file_path, 'rb') as f:
    y_raw = pickle.load(f)

print(f"\nTotal samples loaded: {len(X_raw)}")
print(f"Sample m-height range: [{min(y_raw):.4f}, {max(y_raw):.4f}]")

param_counts = Counter([(s[0], s[1], s[2]) for s in X_raw])
print("\nSample counts per (n, k, m):")
for params, count in sorted(param_counts.items()):
    print(f"  n={params[0]}, k={params[1]}, m={params[2]}: {count} samples")


# ============================================================
# Section 2: LP-Based Data Augmentation
# ============================================================
from scipy.optimize import linprog
from itertools import combinations

def compute_m_height_lp(n, k, m, P):
    """
    Compute the m-height of the analog code with systematic generator
    matrix G = [I_k | P] using the LP-based algorithm.
    """
    G = np.hstack([np.eye(k), P])
    max_z = 1.0

    for S in combinations(range(n), m):
        S_set = set(S)
        S_bar = [t for t in range(n) if t not in S_set]

        for j in S:
            c = -G[:, j]
            A_ub = []
            b_ub = []
            for t in S_bar:
                A_ub.append(G[:, t])
                b_ub.append(1.0)
                A_ub.append(-G[:, t])
                b_ub.append(1.0)

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

            if result.success:
                z_sj = -result.fun
                if z_sj > max_z:
                    max_z = z_sj

    return max_z


def generate_augmented_samples(n, k, m, num_samples, p_range=10.0):
    """Generate new (P, m-height) samples using the LP algorithm."""
    new_X = []
    new_y = []

    for _ in range(num_samples):
        P = np.random.uniform(-p_range, p_range, size=(k, n - k))
        mh = compute_m_height_lp(n, k, m, P)
        if np.isfinite(mh) and mh >= 1.0:
            new_X.append([n, k, m, P])
            new_y.append(mh)

    return new_X, new_y


# Target: bring each group up to ~15000 samples
target_per_group = 15000
aug_X_all = []
aug_y_all = []

print("\nGenerating augmented samples via LP algorithm...")
print("(This may take a few minutes on Colab)\n")

for (n_val, k_val, m_val), count in sorted(param_counts.items()):
    deficit = max(0, target_per_group - count)
    if deficit > 0:
        num_to_gen = min(deficit, 3000)
        print(f"  (n={n_val}, k={k_val}, m={m_val}): existing={count}, generating {num_to_gen} new samples...", end=" ", flush=True)
        t0 = time.time()
        new_x, new_y = generate_augmented_samples(n_val, k_val, m_val, num_to_gen, p_range=10.0)
        aug_X_all.extend(new_x)
        aug_y_all.extend(new_y)
        print(f"done ({len(new_y)} valid, {time.time()-t0:.1f}s)")
    else:
        print(f"  (n={n_val}, k={k_val}, m={m_val}): existing={count}, no augmentation needed")

X_combined = X_raw + aug_X_all
y_combined = y_raw + aug_y_all

print(f"\nOriginal samples: {len(X_raw)}")
print(f"Augmented samples: {len(aug_X_all)}")
print(f"Total samples: {len(X_combined)}")

combined_counts = Counter([(s[0], s[1], s[2]) for s in X_combined])
print("\nUpdated sample counts per (n, k, m):")
for params, count in sorted(combined_counts.items()):
    print(f"  n={params[0]}, k={params[1]}, m={params[2]}: {count} samples")


# ============================================================
# Section 3: Feature Engineering & Preprocessing
# ============================================================

def extract_features(sample):
    """Extract a rich 54-dim feature vector from a single sample [n, k, m, P]."""
    n, k, m, P = sample
    max_k = 6
    max_n_minus_k = 5
    max_sv = 5

    P_padded = np.zeros((max_k, max_n_minus_k))
    P_padded[:k, :n-k] = P
    P_flat = P_padded.flatten()

    row_norms = np.zeros(max_k)
    for i in range(k):
        row_norms[i] = np.linalg.norm(P[i, :])

    col_norms = np.zeros(max_n_minus_k)
    for j in range(n - k):
        col_norms[j] = np.linalg.norm(P[:, j])

    sv = np.zeros(max_sv)
    try:
        singular_values = np.linalg.svd(P, compute_uv=False)
        sv[:len(singular_values)] = np.sort(singular_values)[::-1]
    except:
        pass

    frob_norm = np.linalg.norm(P, 'fro')
    max_abs = np.max(np.abs(P))
    mean_abs = np.mean(np.abs(P))

    try:
        cond = np.log1p(np.linalg.cond(P))
    except:
        cond = 0.0
    if not np.isfinite(cond):
        cond = 50.0

    mean_row = np.mean(row_norms[:k]) if k > 0 else 1.0
    mean_col = np.mean(col_norms[:n-k]) if (n-k) > 0 else 1.0
    ratio = mean_row / (mean_col + 1e-8)

    feat = np.concatenate([
        [n, k, m],
        P_flat,
        row_norms,
        col_norms,
        sv,
        [frob_norm],
        [max_abs],
        [mean_abs],
        [cond],
        [ratio],
    ])
    return feat  # total: 54


def preprocess_all(X_list, y_list):
    """Preprocess all samples into feature array and log2 target array."""
    features = [extract_features(s) for s in X_list]
    X_array = np.array(features, dtype=np.float32)
    y_array = np.log2(np.array(y_list, dtype=np.float32)).reshape(-1, 1)
    return X_array, y_array


X_array, y_log2 = preprocess_all(X_combined, y_combined)
FEATURE_DIM = X_array.shape[1]
print(f"\nFeature dimension: {FEATURE_DIM}")
print(f"Total samples: {X_array.shape[0]}")
print(f"Target log2(y) range: [{y_log2.min():.4f}, {y_log2.max():.4f}]")


# ============================================================
# Section 4: Dataset & DataLoaders (GPU-pinned)
# ============================================================

class MHeightDataset(Dataset):
    """PyTorch Dataset for (X, y) pairs."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


group_keys = sorted(set((s[1], s[2]) for s in X_combined))
print(f"\nMoE Groups (k, m): {group_keys}")

group_data = {}
batch_size = 256

# Use pin_memory for faster CPU→GPU transfer
use_pin = (device.type == "cuda")

for (k_val, m_val) in group_keys:
    indices = [i for i, s in enumerate(X_combined) if s[1] == k_val and s[2] == m_val]

    X_group = X_array[indices]
    y_group = y_log2[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_group)

    np.random.seed(42)
    perm = np.random.permutation(len(X_scaled))
    split = int(0.9 * len(perm))
    train_idx, val_idx = perm[:split], perm[split:]

    train_ds = MHeightDataset(X_scaled[train_idx], y_group[train_idx])
    val_ds = MHeightDataset(X_scaled[val_idx], y_group[val_idx])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=use_pin, num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=use_pin, num_workers=2
    )

    group_data[(k_val, m_val)] = {
        'scaler': scaler,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'feature_dim': X_scaled.shape[1],
    }

    print(f"  Group (k={k_val}, m={m_val}): {len(indices)} total, "
          f"{len(train_idx)} train, {len(val_idx)} val, "
          f"y_log2 range=[{y_group.min():.2f}, {y_group.max():.2f}]")


# ============================================================
# Section 5: Residual DNN Architecture
# ============================================================

class ResidualBlock(nn.Module):
    """A residual block: out = ReLU(BN(Linear(ReLU(BN(Linear(x)))))) + x"""
    def __init__(self, dim, dropout=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class ExpertDNN(nn.Module):
    """Residual DNN expert for a single (k, m) group."""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=3, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)


test_model = ExpertDNN(input_dim=FEATURE_DIM).to(device)
total_params = sum(p.numel() for p in test_model.parameters())
print(f"\nExpert DNN — Parameters per expert: {total_params:,}")
print(f"Total parameters (9 experts): {total_params * 9:,}")

# Verify model is on GPU
print(f"Model device: {next(test_model.parameters()).device}")
del test_model


# ============================================================
# Section 6: Train All Expert DNNs
# ============================================================

def train_expert(group_key, group_info, device, num_epochs=300, patience=20):
    """Train a single expert DNN for a (k, m) group on GPU."""
    k_val, m_val = group_key
    print(f"\n{'='*60}")
    print(f"Training Expert for (k={k_val}, m={m_val}) on {device}")
    print(f"  Train: {group_info['train_size']}, Val: {group_info['val_size']}")
    print(f"{'='*60}")

    model = ExpertDNN(
        input_dim=group_info['feature_dim'],
        hidden_dim=256,
        num_blocks=3,
        dropout=0.15
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_state = None

    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for bx, by in group_info['train_loader']:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            pred = model(bx)
            loss = criterion(pred, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= group_info['train_size']

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in group_info['val_loader']:
                bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= group_info['val_size']

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch [{epoch+1:3d}/{num_epochs}] | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f} | LR: {lr:.6f}")

        if patience_counter >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_state)
    model.to(device)
    print(f"  ✓ Best Val Loss: {best_val_loss:.6f}")

    return model, best_val_loss, train_losses, val_losses


print(f"\n{'#'*60}")
print(f"# TRAINING 9 EXPERT DNNs ON: {device}")
print(f"{'#'*60}")
start_time = time.time()

experts = {}
all_train_histories = {}

for group_key in group_keys:
    t0 = time.time()
    model, best_loss, t_hist, v_hist = train_expert(
        group_key, group_data[group_key], device
    )
    experts[group_key] = model
    all_train_histories[group_key] = (t_hist, v_hist)
    print(f"  ⏱ Time: {time.time()-t0:.1f}s")

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"ALL 9 EXPERTS TRAINED SUCCESSFULLY in {total_time:.1f}s")
print(f"{'='*60}")
for gk in group_keys:
    print(f"  Expert (k={gk[0]}, m={gk[1]}): best val loss = {min(all_train_histories[gk][1]):.6f}")

if device.type == "cuda":
    print(f"\nGPU memory used: {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")


# ============================================================
# Section 7: Plot Training Curves
# ============================================================

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Training & Validation Loss per Expert (k, m)', fontsize=14)

for idx, gk in enumerate(group_keys):
    ax = axes[idx // 3][idx % 3]
    t_hist, v_hist = all_train_histories[gk]
    ax.plot(t_hist, label='Train', linewidth=1)
    ax.plot(v_hist, label='Val', linewidth=1)
    ax.set_title(f'k={gk[0]}, m={gk[1]} (best={min(v_hist):.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log2)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print("Training curves saved to training_curves.png")


# ============================================================
# Section 8: Evaluate on Validation Set
# ============================================================

print("\nPer-Expert Validation Performance:")
print(f"{'Group':<12} {'Avg Cost':>10} {'RMSE_log2':>10} {'Samples':>8}")
print("-" * 45)

total_cost_sum = 0.0
total_val_count = 0

for gk in group_keys:
    model = experts[gk]
    model.eval()

    preds_list = []
    true_list = []

    with torch.no_grad():
        for bx, by in group_data[gk]['val_loader']:
            bx = bx.to(device, non_blocking=True)
            p = model(bx).cpu().numpy()
            preds_list.append(p)
            true_list.append(by.numpy())

    preds = np.concatenate(preds_list).flatten()
    trues = np.concatenate(true_list).flatten()

    costs = (trues - preds) ** 2
    avg_cost = costs.mean()
    rmse = np.sqrt(avg_cost)
    n_samples = len(trues)

    total_cost_sum += costs.sum()
    total_val_count += n_samples

    print(f"(k={gk[0]},m={gk[1]})  {avg_cost:10.6f} {rmse:10.6f} {n_samples:8d}")

overall_cost = total_cost_sum / total_val_count
print("-" * 45)
print(f"{'OVERALL':<12} {overall_cost:10.6f} {np.sqrt(overall_cost):10.6f} {total_val_count:8d}")
print(f"\n★ Overall average grading cost: {overall_cost:.6f}")


# ============================================================
# Section 9: Save MoE Model
# ============================================================

save_dict = {
    'feature_dim': FEATURE_DIM,
    'group_keys': group_keys,
}

for gk in group_keys:
    key_str = f"k{gk[0]}_m{gk[1]}"
    save_dict[f'expert_{key_str}_state'] = experts[gk].cpu().state_dict()
    experts[gk].to(device)  # move back to GPU
    save_dict[f'scaler_{key_str}'] = group_data[gk]['scaler']

torch.save(save_dict, 'moe_model.pth')

file_size = os.path.getsize('moe_model.pth') / 1024
print(f"\nMoE model saved to 'moe_model.pth'")
print(f"File size: {file_size:.1f} KB")
print(f"Contains {len(group_keys)} expert DNNs")


# ============================================================
# Section 10: Generate Predictions (on training data as test)
# ============================================================

test_file_path = 'CSCE-636-Project-1-Train-n_k_m_P'  # <-- Change to your actual test file
output_file_path = 'predicted_mHeights'

# Load test data
with open(test_file_path, 'rb') as f:
    test_data = pickle.load(f)
print(f"\nLoaded {len(test_data)} test samples.")

# Group by (k, m)
test_groups = defaultdict(list)
for i, sample in enumerate(test_data):
    n, k, m, P = sample
    feat = extract_features(sample)
    test_groups[(k, m)].append((i, feat))

# Predict
predicted_mheights = [0.0] * len(test_data)

for gk, items in test_groups.items():
    if gk not in experts:
        print(f"  WARNING: No expert for (k={gk[0]}, m={gk[1]}), using default=1.0")
        for idx, _ in items:
            predicted_mheights[idx] = 1.0
        continue

    indices = [item[0] for item in items]
    features = np.array([item[1] for item in items], dtype=np.float32)

    features_scaled = group_data[gk]['scaler'].transform(features)
    tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    model = experts[gk]
    model.eval()
    with torch.no_grad():
        all_preds = []
        for bi in range(0, len(tensor), 1024):
            batch = tensor[bi:bi+1024]
            pred = model(batch).cpu().numpy().flatten()
            all_preds.append(pred)
        preds_log2 = np.concatenate(all_preds)

    preds = np.maximum(2.0 ** preds_log2, 1.0)

    for idx, val in zip(indices, preds):
        predicted_mheights[idx] = float(val)

    print(f"  Expert (k={gk[0]}, m={gk[1]}): {len(indices)} samples, "
          f"pred range=[{preds.min():.4f}, {preds.max():.4f}]")

# Save predictions
with open(output_file_path, 'wb') as f:
    pickle.dump(predicted_mheights, f)

print(f"\nPredictions saved to '{output_file_path}'")
print(f"Total predictions: {len(predicted_mheights)}")
print(f"Range: [{min(predicted_mheights):.4f}, {max(predicted_mheights):.4f}]")
print(f"All >= 1: {all(h >= 1.0 for h in predicted_mheights)}")

print("\n" + "="*60)
print("DONE! Download 'moe_model.pth' and 'predicted_mHeights' from Colab.")
print("="*60)
