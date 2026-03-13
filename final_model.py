#!/usr/bin/env python3
"""
CSCE 636 Project 1 — Final Training Script for TAMU Grace HPC
==============================================================
Self-contained training pipeline:
  1. Load data + LP augmentation (massive, multi-range)
  2. 77-dim feature engineering
  3. Train 9 base MoE experts (scaled-up ResidualDNN)
  4. Identify weak groups (cost > threshold)
  5. Extra LP augmentation + cross-group transfer for weak groups
  6. Retrain base experts for weak groups (best-of-N)
  7. Diverse ensemble per weak group (16-20 models)
  8. Trimmed ensemble evaluation → pick best strategy per group
  9. Save everything to moe_model.pth + predicted_mHeights

Usage:
  python final_model.py --data_dir /path/to/data --output_dir /path/to/output

Grace HPC (SLURM):
  sbatch submit_job.sh
"""

import argparse
import os
import sys
import time
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
from itertools import combinations
from collections import Counter, defaultdict
from torch.optim.swa_utils import AveragedModel, SWALR
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CLI Arguments
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="CSCE636 Project 1 — Train MoE m-height model")
    p.add_argument('--data_dir', type=str, default='.', help='Directory containing data files')
    p.add_argument('--output_dir', type=str, default='.', help='Directory for model + predictions')
    p.add_argument('--seed', type=int, default=42, help='Global random seed')
    p.add_argument('--base_epochs', type=int, default=1000, help='Max epochs for base experts')
    p.add_argument('--weak_epochs', type=int, default=1200, help='Max epochs for weak group experts')
    p.add_argument('--ensemble_epochs', type=int, default=1200, help='Max epochs per ensemble member')
    p.add_argument('--base_runs', type=int, default=15, help='Best-of-N runs for easy base experts')
    p.add_argument('--hard_runs', type=int, default=15, help='Best-of-N runs for hard base experts')
    p.add_argument('--retrain_runs', type=int, default=15, help='Best-of-N runs for retrained experts')
    p.add_argument('--ensemble_size', type=int, default=15, help='Ensemble members per weak group')
    p.add_argument('--lp_per_group', type=int, default=10000, help='LP augment samples per group (initial)')
    p.add_argument('--lp_weak', type=int, default=15000, help='Extra LP samples per weak group')
    p.add_argument('--batch_size', type=int, default=512, help='Batch size')
    p.add_argument('--cost_threshold', type=float, default=0.0, help='Weak group threshold (0 = treat ALL groups)')
    p.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    p.add_argument('--no_augment', action='store_true', help='Skip initial LP augmentation')
    return p.parse_args()

# =============================================================================
# Dataset
# =============================================================================
class MHeightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# =============================================================================
# Architecture — Residual DNN
# =============================================================================
class ResidualBlock(nn.Module):
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
            nn.Linear(64, 1),
        )
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.output_head(x)

# =============================================================================
# LP-based m-height computation
# =============================================================================
def compute_m_height_lp(n, k, m, P):
    G = np.hstack([np.eye(k), P])
    max_z = 1.0
    for S in combinations(range(n), m):
        S_set = set(S)
        S_bar = [t for t in range(n) if t not in S_set]
        for j in S:
            c = -G[:, j]
            A_ub, b_ub = [], []
            for t in S_bar:
                A_ub.append(G[:, t]);  b_ub.append(1.0)
                A_ub.append(-G[:, t]); b_ub.append(1.0)
            result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), method='highs')
            if result.success:
                z = -result.fun
                if z > max_z:
                    max_z = z
    return max_z

def generate_lp_samples(n, k, m, num_samples, p_ranges=None):
    """Generate LP-augmented samples across multiple P-value ranges."""
    if p_ranges is None:
        p_ranges = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    new_X, new_y = [], []
    per_range = max(1, num_samples // len(p_ranges))
    for p_range in p_ranges:
        for _ in range(per_range):
            P = np.random.uniform(-p_range, p_range, size=(k, n - k))
            mh = compute_m_height_lp(n, k, m, P)
            if np.isfinite(mh) and mh >= 1.0:
                new_X.append([n, k, m, P])
                new_y.append(mh)
    return new_X, new_y

# =============================================================================
# Feature Engineering (77-dim)
# =============================================================================
FEATURE_DIM = 77

def extract_features(sample):
    n, k, m, P = sample
    max_k, max_nk, max_sv = 6, 5, 5

    # 1) Padded P (30)
    P_padded = np.zeros((max_k, max_nk))
    P_padded[:k, :n-k] = P
    P_flat = P_padded.flatten()

    # 2) Row norms (6)
    row_norms = np.zeros(max_k)
    for i in range(k):
        row_norms[i] = np.linalg.norm(P[i, :])

    # 3) Col norms (5)
    col_norms = np.zeros(max_nk)
    for j in range(n - k):
        col_norms[j] = np.linalg.norm(P[:, j])

    # 4) SVD (5)
    sv = np.zeros(max_sv)
    try:
        svs = np.linalg.svd(P, compute_uv=False)
        sv[:len(svs)] = np.sort(svs)[::-1]
    except:
        pass

    # 5) Stats (5)
    frob_norm = np.linalg.norm(P, 'fro')
    max_abs = np.max(np.abs(P))
    mean_abs = np.mean(np.abs(P))
    std_abs = np.std(np.abs(P))
    min_abs = np.min(np.abs(P))

    # 6) Cond (1)
    try:
        cond = np.log1p(np.linalg.cond(P))
    except:
        cond = 0.0
    if not np.isfinite(cond):
        cond = 50.0

    # 7) Ratio (1)
    mean_row = np.mean(row_norms[:k]) if k > 0 else 1.0
    mean_col = np.mean(col_norms[:n-k]) if (n - k) > 0 else 1.0
    ratio = mean_row / (mean_col + 1e-8)

    # 8) G=[I|P] col norms (9+4=13)
    G = np.hstack([np.eye(k), P])
    g_col_norms = np.zeros(9)
    for j in range(n):
        g_col_norms[j] = np.linalg.norm(G[:, j])
    g_col_max = np.max(g_col_norms)
    g_col_min = np.min(g_col_norms[:n])
    g_col_mean = np.mean(g_col_norms[:n])
    g_col_std = np.std(g_col_norms[:n])

    # 9) Rank (2)
    try:
        rank = np.linalg.matrix_rank(P)
    except:
        rank = min(k, n - k)
    effective_rank = np.sum(sv > 1e-6)

    # 10) SV ratios (3)
    sv_ratio = sv[0] / (sv[min(k, n-k)-1] + 1e-10) if sv[min(k, n-k)-1] > 1e-10 else 100.0
    sv_sum = np.sum(sv)
    sv_energy_ratio = sv[0] / (sv_sum + 1e-10)

    # 11) PtP features (2)
    try:
        PtP = P.T @ P
        ptP_trace = np.trace(PtP)
        ptP_frob = np.linalg.norm(PtP, 'fro')
    except:
        ptP_trace = frob_norm ** 2
        ptP_frob = frob_norm ** 2

    # 12) Det (1)
    try:
        sq = min(k, n - k)
        det_val = np.log1p(abs(np.linalg.det(P[:sq, :sq])))
    except:
        det_val = 0.0
    if not np.isfinite(det_val):
        det_val = 0.0

    return np.concatenate([
        [n, k, m], P_flat, row_norms, col_norms, sv,
        [frob_norm, max_abs, mean_abs, std_abs, min_abs],
        [cond, ratio],
        g_col_norms, [g_col_max, g_col_min, g_col_mean, g_col_std],
        [rank, effective_rank],
        [sv_ratio, sv_sum, sv_energy_ratio],
        [ptP_trace, ptP_frob],
        [det_val],
    ])

# =============================================================================
# Loss helpers
# =============================================================================
def log_cosh_loss(pred, target):
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))

def feature_noise(x, std=0.05):
    return x + torch.randn_like(x) * std

def combined_loss(pred, target, beta=0.5):
    """50% Huber + 30% MSE + 20% log-cosh"""
    return (0.5 * nn.functional.smooth_l1_loss(pred, target, beta=beta)
            + 0.3 * nn.functional.mse_loss(pred, target)
            + 0.2 * log_cosh_loss(pred, target))

# =============================================================================
# Training helpers
# =============================================================================
def train_single_run(model, train_loader, val_loader, train_size, val_size,
                     device, lr, weight_decay, max_epochs, patience,
                     noise_std=0.0, use_combined_loss=False, seed=42,
                     use_mixup=False, mixup_alpha=0.2):
    """Train a model for one run, return best val loss + state dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_combined_loss:
        criterion_fn = lambda p, t: combined_loss(p, t, beta=0.5)
    else:
        criterion = nn.SmoothL1Loss(beta=0.5)
        criterion_fn = lambda p, t: criterion(p, t)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=60, T_mult=2, eta_min=1e-7
    )

    best_val = float('inf')
    pat_cnt = 0
    best_state = None
    t_hist, v_hist = [], []

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            if noise_std > 0:
                bx = feature_noise(bx, std=noise_std)
            # Mixup augmentation: interpolate between random pairs
            if use_mixup and mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(bx.size(0), device=bx.device)
                bx = lam * bx + (1 - lam) * bx[idx]
                by = lam * by + (1 - lam) * by[idx]
            pred = model(bx)
            loss = criterion_fn(pred, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= train_size
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                val_loss += nn.functional.mse_loss(model(bx), by).item() * bx.size(0)
        val_loss /= val_size

        t_hist.append(train_loss)
        v_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            pat_cnt = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat_cnt += 1

        if (epoch + 1) % 200 == 0 or epoch == 0:
            print(f"      Ep [{epoch+1:4d}/{max_epochs}] Train: {train_loss:.6f} "
                  f"Val: {val_loss:.6f} Best: {best_val:.6f}")

        if pat_cnt >= patience:
            print(f"      Early stop ep {epoch+1}, best: {best_val:.6f}")
            break

    return best_val, best_state, t_hist, v_hist

def batch_predict(model, tensor, batch_size=2048):
    preds = []
    model.eval()
    with torch.no_grad():
        for bi in range(0, len(tensor), batch_size):
            preds.append(model(tensor[bi:bi+batch_size]).cpu().numpy().flatten())
    return np.concatenate(preds)

# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("=" * 70)
    print("STEP 1: Load Data")
    print("=" * 70)

    x_path = os.path.join(args.data_dir, 'CSCE-636-Project-1-Train-n_k_m_P')
    y_path = os.path.join(args.data_dir, 'CSCE-636-Project-1-Train-mHeights')

    with open(x_path, 'rb') as f:
        X_raw = pickle.load(f)
    with open(y_path, 'rb') as f:
        y_raw = pickle.load(f)

    print(f"Loaded {len(X_raw)} samples, y range [{min(y_raw):.4f}, {max(y_raw):.4f}]")
    param_counts = Counter([(s[0], s[1], s[2]) for s in X_raw])
    for params, count in sorted(param_counts.items()):
        print(f"  n={params[0]}, k={params[1]}, m={params[2]}: {count}")

    # =========================================================================
    # STEP 2: Initial LP Augmentation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: LP Augmentation (initial)")
    print("=" * 70)

    X_combined = list(X_raw)
    y_combined = list(y_raw)

    if not args.no_augment:
        target_per_group = 20000
        for (n_val, k_val, m_val), count in sorted(param_counts.items()):
            deficit = max(0, target_per_group - count)
            num_gen = min(deficit, args.lp_per_group)
            if num_gen > 0:
                print(f"  (n={n_val},k={k_val},m={m_val}): +{num_gen} LP samples...", end=" ", flush=True)
                t0 = time.time()
                new_x, new_y = generate_lp_samples(n_val, k_val, m_val, num_gen,
                                                    p_ranges=[1, 2, 5, 10, 20, 50, 100])
                X_combined.extend(new_x)
                y_combined.extend(new_y)
                print(f"{len(new_y)} valid ({time.time()-t0:.1f}s)")
            else:
                print(f"  (n={n_val},k={k_val},m={m_val}): {count} — skip")

    print(f"Total after augmentation: {len(X_combined)}")

    # =========================================================================
    # STEP 3: Feature Engineering
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Feature Engineering (77-dim)")
    print("=" * 70)

    X_array = np.array([extract_features(s) for s in X_combined], dtype=np.float32)
    y_log2 = np.log2(np.array(y_combined, dtype=np.float32)).reshape(-1, 1)
    X_array = np.nan_to_num(X_array, nan=0.0, posinf=100.0, neginf=-100.0)

    print(f"Features: {X_array.shape}, Targets: {y_log2.shape}")
    print(f"log2(y) range: [{y_log2.min():.4f}, {y_log2.max():.4f}]")

    # =========================================================================
    # STEP 4: Per-group Dataset Prep
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Per-group Dataset Preparation")
    print("=" * 70)

    group_keys = sorted(set((s[1], s[2]) for s in X_combined))
    print(f"Groups: {group_keys}")

    group_data = {}
    for (k_val, m_val) in group_keys:
        indices = [i for i, s in enumerate(X_combined) if s[1] == k_val and s[2] == m_val]
        X_grp = X_array[indices]
        y_grp = y_log2[indices]
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_grp)
        np.random.seed(42)
        perm = np.random.permutation(len(X_sc))
        split = int(0.9 * len(perm))
        train_idx, val_idx = perm[:split], perm[split:]
        train_ds = MHeightDataset(X_sc[train_idx], y_grp[train_idx])
        val_ds = MHeightDataset(X_sc[val_idx], y_grp[val_idx])
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
        group_data[(k_val, m_val)] = {
            'scaler': scaler, 'train_loader': train_loader, 'val_loader': val_loader,
            'train_size': len(train_idx), 'val_size': len(val_idx),
            'feature_dim': X_sc.shape[1],
        }
        print(f"  (k={k_val},m={m_val}): {len(indices)} total, "
              f"{len(train_idx)} train, {len(val_idx)} val")

    # =========================================================================
    # STEP 5: Train Base Experts (SCALED UP)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Train 9 Base Experts")
    print("=" * 70)

    # Scaled-up configs: (hidden_dim, num_blocks, dropout, lr, epochs, patience, num_runs)
    GROUP_HPARAMS = {
        # Easy groups — upgraded to 768/6, more patience
        (4, 3): (768,  6, 0.12, 5e-4, args.base_epochs, 60, args.base_runs),
        (5, 2): (768,  6, 0.12, 5e-4, args.base_epochs, 60, args.base_runs),
        (6, 2): (768,  6, 0.12, 5e-4, args.base_epochs, 60, args.base_runs),
        # Medium groups — upgraded to 1024/8
        (4, 2): (1024, 8, 0.10, 3e-4, args.base_epochs, 70, args.hard_runs),
        (4, 4): (1024, 8, 0.10, 3e-4, args.base_epochs, 70, args.hard_runs),
        (5, 3): (1024, 8, 0.10, 3e-4, args.base_epochs, 70, args.hard_runs),
        # Hard groups — maximum capacity
        (4, 5): (1024, 8, 0.10, 3e-4, args.base_epochs, 80, args.hard_runs),
        (5, 4): (1024, 8, 0.10, 3e-4, args.base_epochs, 80, args.hard_runs),
        (6, 3): (1024, 8, 0.10, 3e-4, args.base_epochs, 80, args.hard_runs),
    }

    experts = {}
    expert_hparams = {}
    t_total = time.time()

    for gk in group_keys:
        hparams = GROUP_HPARAMS.get(gk, (512, 5, 0.15, 8e-4, args.base_epochs, 40, args.base_runs))
        hidden_dim, num_blocks, dropout, lr, max_epochs, patience, num_runs = hparams
        expert_hparams[gk] = (hidden_dim, num_blocks, dropout)
        info = group_data[gk]

        print(f"\n{'='*60}")
        print(f"Expert (k={gk[0]},m={gk[1]}): h={hidden_dim}, b={num_blocks}, "
              f"d={dropout}, runs={num_runs}")
        print(f"  Data: {info['train_size']} train, {info['val_size']} val")
        print(f"{'='*60}")

        global_best_val = float('inf')
        global_best_state = None

        for run in range(num_runs):
            print(f"\n    --- Run {run+1}/{num_runs} ---")
            seed = args.seed + run * 777 + gk[0] * 100 + gk[1] * 10
            model = ExpertDNN(info['feature_dim'], hidden_dim, num_blocks, dropout).to(device)
            val_loss, state, _, _ = train_single_run(
                model, info['train_loader'], info['val_loader'],
                info['train_size'], info['val_size'], device,
                lr=lr, weight_decay=1e-4, max_epochs=max_epochs,
                patience=patience, noise_std=0.02,
                use_combined_loss=False, seed=seed,
                use_mixup=True, mixup_alpha=0.2,
            )
            print(f"    Run {run+1} best val: {val_loss:.6f}")
            if val_loss < global_best_val:
                global_best_val = val_loss
                global_best_state = state

        model = ExpertDNN(info['feature_dim'], hidden_dim, num_blocks, dropout).to(device)
        model.load_state_dict(global_best_state)
        model.eval()
        experts[gk] = model
        print(f"  ✓ (k={gk[0]},m={gk[1]}): best_val = {global_best_val:.6f}")

    print(f"\n[Base experts] Total time: {time.time()-t_total:.1f}s")

    # =========================================================================
    # STEP 6: Identify Weak Groups + Base Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: Evaluate Base Experts → Identify Weak Groups")
    print("=" * 70)

    group_costs = {}
    total_cost_sum, total_count = 0.0, 0
    for gk in group_keys:
        model = experts[gk]; model.eval()
        preds_l, trues_l = [], []
        with torch.no_grad():
            for bx, by in group_data[gk]['val_loader']:
                bx = bx.to(device)
                preds_l.append(model(bx).cpu().numpy())
                trues_l.append(by.numpy())
        pa = np.concatenate(preds_l).flatten()
        ta = np.concatenate(trues_l).flatten()
        cost = ((ta - pa) ** 2).mean()
        group_costs[gk] = cost
        total_cost_sum += ((ta - pa) ** 2).sum()
        total_count += len(ta)
        marker = "★ WEAK" if cost > args.cost_threshold else "✓ ok"
        print(f"  (k={gk[0]},m={gk[1]}): cost={cost:.4f} {marker}")

    overall_base = total_cost_sum / total_count
    print(f"\nOverall base cost: {overall_base:.6f}")

    weak_groups = [gk for gk in group_keys if group_costs[gk] > args.cost_threshold]
    print(f"Weak groups: {weak_groups}")

    if not weak_groups:
        print("\n✅ No weak groups — saving base model only.")
        # Skip to save
        retrained_experts = {}
        retrained_scalers = {}
        weak_ensembles = {}
        weak_scalers = {}
        ensemble_val_costs = {}
        use_ensemble_for = []
        best_strategy = {}
        ensemble_weights = {}
        improved_costs = {}
    else:
        # =====================================================================
        # STEP 7: Massive LP Augmentation for Weak Groups
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 7: Extra LP Augmentation for Weak Groups")
        print("=" * 70)

        weak_extra_X, weak_extra_y = {}, {}
        for gk in weak_groups:
            k_val, m_val = gk
            print(f"\n  Generating {args.lp_weak} LP samples for (k={k_val},m={m_val})...",
                  flush=True)
            t0 = time.time()
            new_x, new_y = generate_lp_samples(
                9, k_val, m_val, args.lp_weak,
                p_ranges=[1, 2, 5, 10, 20, 50, 100, 200]
            )
            weak_extra_X[gk] = new_x
            weak_extra_y[gk] = new_y
            print(f"    Generated {len(new_y)} samples in {time.time()-t0:.1f}s")

        # =====================================================================
        # STEP 8: Build Augmented Datasets + Cross-Group Transfer
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 8: Build Augmented Datasets with Cross-Group Transfer")
        print("=" * 70)

        cross_group_map = {
            # Every group borrows from neighbors with same k or adjacent m
            (4, 2): [(4, 3)],
            (4, 3): [(4, 2), (4, 4)],
            (4, 4): [(4, 3), (4, 5)],
            (4, 5): [(4, 4), (4, 3)],
            (5, 2): [(5, 3)],
            (5, 3): [(5, 2), (5, 4)],
            (5, 4): [(5, 3), (5, 2)],
            (6, 2): [(6, 3)],
            (6, 3): [(6, 2)],
        }

        weak_group_data = {}
        for gk in weak_groups:
            k_val, m_val = gk

            # Original data
            orig_indices = [i for i, s in enumerate(X_combined) if s[1] == k_val and s[2] == m_val]
            X_orig = X_array[orig_indices]
            y_orig = y_log2[orig_indices]

            # LP augmented
            extra_feats = np.array([extract_features(s) for s in weak_extra_X[gk]], dtype=np.float32)
            extra_y = np.log2(np.array(weak_extra_y[gk], dtype=np.float32)).reshape(-1, 1)

            # Cross-group transfer
            cross_X_list, cross_y_list = [], []
            for neighbor_gk in cross_group_map.get(gk, []):
                nk, nm = neighbor_gk
                nidx = [i for i, s in enumerate(X_combined) if s[1] == nk and s[2] == nm]
                if nidx:
                    np.random.seed(42)
                    n_borrow = max(500, len(nidx) // 10)
                    chosen = np.random.choice(nidx, size=min(n_borrow, len(nidx)), replace=False)
                    cross_X_list.append(X_array[chosen])
                    cross_y_list.append(y_log2[chosen])
                    print(f"    Borrowed {len(chosen)} from (k={nk},m={nm})")

            # Combine
            parts_X = [X_orig, extra_feats]
            parts_y = [y_orig, extra_y]
            if cross_X_list:
                parts_X.extend(cross_X_list)
                parts_y.extend(cross_y_list)

            X_all = np.vstack(parts_X)
            y_all = np.vstack(parts_y)
            X_all = np.nan_to_num(X_all, nan=0.0, posinf=100.0, neginf=-100.0)

            # Target clipping
            y_p5, y_p95 = np.percentile(y_all, 2), np.percentile(y_all, 98)
            y_clipped = np.clip(y_all, y_p5, y_p95)
            n_clipped = np.sum(y_all != y_clipped)
            if n_clipped > 0:
                print(f"    Clipped {n_clipped} targets to [{y_p5:.2f}, {y_p95:.2f}]")

            # Scaler on ALL data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_all)

            # Split: original val pure
            n_orig = len(X_orig)
            np.random.seed(42)
            orig_perm = np.random.permutation(n_orig)
            orig_val_size = int(0.1 * n_orig)
            orig_val_idx = orig_perm[:orig_val_size]
            orig_train_idx = orig_perm[orig_val_size:]

            train_indices = list(orig_train_idx) + list(range(n_orig, len(X_scaled)))
            val_indices = list(orig_val_idx)

            train_ds = MHeightDataset(X_scaled[train_indices], y_clipped[train_indices])
            val_ds = MHeightDataset(X_scaled[val_indices], y_all[val_indices])  # unclipped val

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                      drop_last=True, num_workers=args.num_workers, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)

            weak_group_data[gk] = {
                'scaler': scaler, 'train_loader': train_loader, 'val_loader': val_loader,
                'train_size': len(train_indices), 'val_size': len(val_indices),
                'feature_dim': X_scaled.shape[1],
            }
            print(f"  (k={k_val},m={m_val}): {n_orig} orig + {len(extra_feats)} LP + "
                  f"{sum(len(cx) for cx in cross_X_list) if cross_X_list else 0} cross "
                  f"= {len(X_all)} total, train={len(train_indices)}, val={len(val_indices)}")

        # =====================================================================
        # STEP 9: Retrain Base Experts for Weak Groups
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 9: Retrain Base Experts (Weak Groups) — Scaled Up")
        print("=" * 70)

        RETRAIN_CONFIGS = {
            # (hidden, blocks, dropout, wd, lr, epochs, patience, n_runs, noise_std)
            # Hard groups — maximum treatment
            (4, 5): (1024, 8, 0.18, 5e-4, 2e-4, args.weak_epochs, 100, args.retrain_runs, 0.06),
            (5, 4): (1024, 8, 0.18, 5e-4, 2e-4, args.weak_epochs, 100, args.retrain_runs, 0.06),
            (6, 3): (1024, 8, 0.15, 3e-4, 2e-4, args.weak_epochs, 100, args.retrain_runs, 0.05),
            # Medium groups
            (4, 2): (1024, 8, 0.15, 3e-4, 2e-4, args.weak_epochs, 80, args.retrain_runs, 0.04),
            (4, 4): (1024, 8, 0.15, 3e-4, 2e-4, args.weak_epochs, 80, args.retrain_runs, 0.05),
            (5, 3): (1024, 8, 0.15, 3e-4, 2e-4, args.weak_epochs, 80, args.retrain_runs, 0.05),
            # Easy groups — also retrain with bigger arch + augmented data
            (4, 3): (768,  6, 0.15, 3e-4, 3e-4, args.weak_epochs, 70, args.retrain_runs, 0.03),
            (5, 2): (768,  6, 0.15, 3e-4, 3e-4, args.weak_epochs, 70, args.retrain_runs, 0.03),
            (6, 2): (768,  6, 0.12, 3e-4, 3e-4, args.weak_epochs, 70, args.retrain_runs, 0.03),
        }

        retrained_experts = {}
        retrained_scalers = {}

        for gk in weak_groups:
            if gk not in RETRAIN_CONFIGS:
                # Fallback for unexpected weak groups
                RETRAIN_CONFIGS[gk] = (768, 6, 0.15, 3e-4, 3e-4, args.weak_epochs, 60, args.retrain_runs, 0.04)
            h_dim, n_blk, drop, wd, lr, max_ep, patience, n_runs, noise_std = RETRAIN_CONFIGS[gk]
            info = weak_group_data[gk]

            print(f"\n{'='*60}")
            print(f"RETRAIN Expert (k={gk[0]},m={gk[1]}): h={h_dim}, b={n_blk}, "
                  f"runs={n_runs}")
            print(f"  Data: {info['train_size']} train, {info['val_size']} val")
            print(f"  Old cost: {group_costs[gk]:.4f}")
            print(f"{'='*60}")

            global_best_val = float('inf')
            global_best_state = None

            for run in range(n_runs):
                print(f"\n    --- Run {run+1}/{n_runs} ---")
                seed = 1000 + run * 333 + gk[0] * 100 + gk[1] * 10
                model = ExpertDNN(info['feature_dim'], h_dim, n_blk, drop).to(device)
                val_loss, state, _, _ = train_single_run(
                    model, info['train_loader'], info['val_loader'],
                    info['train_size'], info['val_size'], device,
                    lr=lr, weight_decay=wd, max_epochs=max_ep,
                    patience=patience, noise_std=noise_std,
                    use_combined_loss=True, seed=seed,
                    use_mixup=True, mixup_alpha=0.2,
                )
                print(f"    Run {run+1} best val: {val_loss:.6f}")
                if val_loss < global_best_val:
                    global_best_val = val_loss
                    global_best_state = state

            model = ExpertDNN(info['feature_dim'], h_dim, n_blk, drop).to(device)
            model.load_state_dict(global_best_state)
            model.eval()
            retrained_experts[gk] = model
            retrained_scalers[gk] = info['scaler']
            print(f"  ✓ Best retrained val: {global_best_val:.6f}")

        # =====================================================================
        # STEP 10: Diverse Ensemble for Weak Groups (SCALED UP)
        # =====================================================================
        print("\n" + "=" * 70)
        print(f"STEP 10: Diverse Ensemble ({args.ensemble_size} models per weak group)")
        print("=" * 70)

        # Generate diverse architectures
        ENSEMBLE_CONFIGS = []
        arch_pool = [
            # (hidden_dim, num_blocks, dropout, lr, noise_std)
            (768,  6,  0.18, 2e-4, 0.06),
            (1024, 5,  0.18, 2e-4, 0.07),
            (768,  8,  0.15, 1e-4, 0.06),
            (1024, 6,  0.15, 2e-4, 0.06),
            (1280, 5,  0.18, 1e-4, 0.05),
            (768,  10, 0.12, 1e-4, 0.06),
            (1024, 8,  0.15, 1e-4, 0.05),
            (1536, 5,  0.18, 1e-4, 0.05),
            (1024, 10, 0.12, 1e-4, 0.05),
            (2048, 4,  0.20, 1e-4, 0.06),
            (768,  7,  0.18, 2e-4, 0.07),
            (1280, 6,  0.15, 1e-4, 0.05),
            (1024, 7,  0.15, 2e-4, 0.06),
            (1536, 6,  0.15, 1e-4, 0.05),
            (2048, 5,  0.18, 1e-4, 0.05),
            (768,  9,  0.12, 1e-4, 0.06),
            (1280, 8,  0.12, 1e-4, 0.05),
            (1536, 4,  0.20, 2e-4, 0.06),
            (2048, 6,  0.15, 1e-4, 0.04),
            (1024, 9,  0.12, 1e-4, 0.05),
        ]
        for i in range(args.ensemble_size):
            cfg = arch_pool[i % len(arch_pool)]
            seed_offset = (i + 1) * 100
            ENSEMBLE_CONFIGS.append((*cfg, seed_offset))

        weak_ensembles = {}
        weak_scalers = {}
        ensemble_val_costs = {}

        for gk in weak_groups:
            info = weak_group_data[gk]
            weak_scalers[gk] = info['scaler']
            ens_models = []
            ens_costs = []

            print(f"\n{'='*60}")
            print(f"ENSEMBLE (k={gk[0]},m={gk[1]}): {args.ensemble_size} models")
            print(f"{'='*60}")

            for eidx, (h_dim, n_blk, drop, lr, noise_std, seed_off) in enumerate(ENSEMBLE_CONFIGS):
                print(f"\n  Ens {eidx+1}/{args.ensemble_size} (h={h_dim}, b={n_blk}, d={drop})")
                seed = seed_off + gk[0] * 100 + gk[1] * 10

                model = ExpertDNN(info['feature_dim'], h_dim, n_blk, drop).to(device)
                val_loss, state, _, _ = train_single_run(
                    model, info['train_loader'], info['val_loader'],
                    info['train_size'], info['val_size'], device,
                    lr=lr, weight_decay=5e-4,
                    max_epochs=args.ensemble_epochs,
                    patience=80, noise_std=noise_std,
                    use_combined_loss=True, seed=seed,
                    use_mixup=True, mixup_alpha=0.2,
                )
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                ens_models.append(model)
                ens_costs.append(val_loss)
                print(f"    ✓ Best: {val_loss:.6f}")

            weak_ensembles[gk] = ens_models
            ensemble_val_costs[gk] = ens_costs

        # =====================================================================
        # STEP 11: Evaluate — Fair Comparison on Original Val Set
        # =====================================================================
        print("\n" + "=" * 70)
        print("STEP 11: Evaluate All Strategies (Trimmed, Weighted, TopK, ...)")
        print("=" * 70)

        improved_costs = {}
        ensemble_weights = {}
        best_strategy = {}

        for gk in weak_groups:
            k_val, m_val = gk

            # Get raw val features from ORIGINAL split
            orig_indices = [i for i, s in enumerate(X_combined) if s[1] == k_val and s[2] == m_val]
            X_grp = X_array[orig_indices]
            y_grp = y_log2[orig_indices]
            np.random.seed(42)
            perm = np.random.permutation(len(X_grp))
            split = int(0.9 * len(perm))
            val_idx = perm[split:]
            X_val_raw = X_grp[val_idx]
            y_val = y_grp[val_idx].flatten()

            # A) Original expert
            orig_scaler = group_data[gk]['scaler']
            orig_model = experts[gk]; orig_model.eval()
            t_orig = torch.tensor(orig_scaler.transform(X_val_raw), dtype=torch.float32).to(device)
            orig_preds = batch_predict(orig_model, t_orig)
            orig_cost = ((y_val - orig_preds) ** 2).mean()

            # B) Retrained expert
            ret_preds = None
            ret_cost = float('inf')
            if gk in retrained_experts:
                ret_scaler = retrained_scalers[gk]
                ret_model = retrained_experts[gk]; ret_model.eval()
                t_ret = torch.tensor(ret_scaler.transform(X_val_raw), dtype=torch.float32).to(device)
                ret_preds = batch_predict(ret_model, t_ret)
                ret_cost = ((y_val - ret_preds) ** 2).mean()

            # C) Ensemble
            ens_pred_arrays = []
            if gk in weak_ensembles:
                ens_scaler = weak_scalers[gk]
                t_ens = torch.tensor(ens_scaler.transform(X_val_raw), dtype=torch.float32).to(device)
                for em in weak_ensembles[gk]:
                    em.eval()
                    ens_pred_arrays.append(batch_predict(em, t_ens))

            # Build pools
            pool_no_orig = []
            if ret_preds is not None:
                pool_no_orig.append(ret_preds)
            pool_no_orig.extend(ens_pred_arrays)
            pool_with_orig = [orig_preds] + pool_no_orig

            results = {'original': orig_cost}
            if ret_preds is not None:
                results['retrained'] = ret_cost

            if pool_no_orig:
                arr_no = np.array(pool_no_orig)
                arr_full = np.array(pool_with_orig)

                # Simple (retrained + ensemble only)
                results['simple'] = ((y_val - np.mean(arr_no, axis=0)) ** 2).mean()

                # Full (all including original)
                results['full'] = ((y_val - np.mean(arr_full, axis=0)) ** 2).mean()

                # Median
                results['median'] = ((y_val - np.median(arr_full, axis=0)) ** 2).mean()

                # Weighted
                pool_costs = [orig_cost]
                if ret_preds is not None:
                    pool_costs.append(ret_cost)
                pool_costs.extend(ensemble_val_costs.get(gk, [1.0] * len(ens_pred_arrays)))
                inv = [1.0 / (c + 1e-8) for c in pool_costs]
                w = np.array([x / sum(inv) for x in inv])
                weighted_avg = np.average(arr_full, axis=0, weights=w)
                results['weighted'] = ((y_val - weighted_avg) ** 2).mean()

                # TopK
                K = min(5, len(pool_costs))
                si = np.argsort(pool_costs)[:K]
                tw = w[si]; tw = tw / tw.sum()
                topk_avg = np.average(arr_full[si], axis=0, weights=tw)
                results['topk'] = ((y_val - topk_avg) ** 2).mean()

                # Trimmed mean: remove worst N models
                for trim_n in [2, 3, 4]:
                    if len(arr_full) > trim_n + 3:
                        per_model = [((y_val - arr_full[i]) ** 2).mean() for i in range(len(arr_full))]
                        keep = np.argsort(per_model)[:len(arr_full) - trim_n]
                        trimmed = np.mean(arr_full[keep], axis=0)
                        tag = f'trimmed_{trim_n}'
                        results[tag] = ((y_val - trimmed) ** 2).mean()

            best_name = min(results, key=results.get)
            best_val = results[best_name]

            print(f"\n  (k={k_val},m={m_val}): orig={orig_cost:.4f} → best={best_name}({best_val:.4f})")
            for name, val in sorted(results.items(), key=lambda x: x[1]):
                print(f"    {name:>15}: {val:.4f}")

            # Store trimmed_keep for the winning strategy
            trimmed_keep = None
            if best_name.startswith('trimmed_'):
                trim_n = int(best_name.split('_')[1])
                per_model_final = [((y_val - np.array(pool_with_orig)[i]) ** 2).mean()
                                   for i in range(len(pool_with_orig))]
                trimmed_keep = np.argsort(per_model_final)[:len(pool_with_orig) - trim_n].tolist()
            elif best_name == 'simple':
                trimmed_keep = None  # use all non-orig
            elif best_name == 'full':
                trimmed_keep = list(range(len(pool_with_orig)))

            improved_costs[gk] = {
                'old_cost': float(orig_cost),
                'new_cost': float(best_val),
                'strategy': best_name,
                'weights': w.tolist() if 'weighted' in results else None,
                'topk_indices': si.tolist() if 'topk' in results else None,
                'topk_weights': tw.tolist() if 'topk' in results else None,
                'trimmed_keep': trimmed_keep,
            }
            best_strategy[gk] = best_name
            ensemble_weights[gk] = w.tolist() if 'weighted' in results else []

        # Determine which groups improved
        use_ensemble_for = []
        for gk in weak_groups:
            if improved_costs[gk]['new_cost'] < improved_costs[gk]['old_cost']:
                use_ensemble_for.append(gk)

        # Overall evaluation
        print("\n" + "-" * 60)
        print("OVERALL EVALUATION:")
        total_cost_sum_new, total_count_new = 0.0, 0
        for gk in group_keys:
            if gk in improved_costs and improved_costs[gk]['new_cost'] < improved_costs[gk]['old_cost']:
                cost = improved_costs[gk]['new_cost']
                k_val, m_val = gk
                orig_idx = [i for i, s in enumerate(X_combined) if s[1] == k_val and s[2] == m_val]
                n_samp = len(orig_idx) - int(0.9 * len(orig_idx))
                total_cost_sum_new += cost * n_samp
                total_count_new += n_samp
                source = improved_costs[gk]['strategy']
            else:
                model = experts[gk]; model.eval()
                preds_l, trues_l = [], []
                with torch.no_grad():
                    for bx, by in group_data[gk]['val_loader']:
                        bx = bx.to(device)
                        preds_l.append(model(bx).cpu().numpy())
                        trues_l.append(by.numpy())
                pa = np.concatenate(preds_l).flatten()
                ta = np.concatenate(trues_l).flatten()
                cost = ((ta - pa) ** 2).mean()
                total_cost_sum_new += ((ta - pa) ** 2).sum()
                total_count_new += len(ta)
                source = "original"
                n_samp = len(ta)
            print(f"  (k={gk[0]},m={gk[1]}): cost={cost:.6f} [{source}] ({n_samp} samples)")

        new_overall = total_cost_sum_new / total_count_new
        print(f"\n  Base overall:  {overall_base:.6f}")
        print(f"  New overall:   {new_overall:.6f}")
        print(f"  Improvement:   {overall_base - new_overall:+.6f}")

    # =========================================================================
    # STEP 12: Save Model
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 12: Save Model")
    print("=" * 70)

    ENSEMBLE_SIZE = args.ensemble_size

    save_dict = {
        'feature_dim': FEATURE_DIM,
        'group_keys': group_keys,
        'ensemble_groups': use_ensemble_for if weak_groups else [],
        'ensemble_size': ENSEMBLE_SIZE if weak_groups else 0,
    }

    # Base experts
    for gk in group_keys:
        ks = f"k{gk[0]}_m{gk[1]}"
        save_dict[f'expert_{ks}_state'] = experts[gk].state_dict()
        save_dict[f'scaler_{ks}'] = group_data[gk]['scaler']
        h, b, d = expert_hparams[gk]
        save_dict[f'expert_{ks}_hidden_dim'] = h
        save_dict[f'expert_{ks}_num_blocks'] = b
        save_dict[f'expert_{ks}_dropout'] = d

    # Retrained + ensemble for weak groups
    if weak_groups:
        for gk in use_ensemble_for:
            ks = f"k{gk[0]}_m{gk[1]}"
            strategy = best_strategy.get(gk, 'simple')
            save_dict[f'ensemble_{ks}_strategy'] = strategy

            # Retrained expert
            if gk in retrained_experts:
                save_dict[f'retrained_{ks}_state'] = retrained_experts[gk].state_dict()
                save_dict[f'retrained_scaler_{ks}'] = retrained_scalers[gk]
                cfg = RETRAIN_CONFIGS[gk]
                save_dict[f'retrained_{ks}_hidden_dim'] = cfg[0]
                save_dict[f'retrained_{ks}_num_blocks'] = cfg[1]
                save_dict[f'retrained_{ks}_dropout'] = cfg[2]

            # Ensemble models
            if gk in weak_ensembles:
                save_dict[f'ensemble_scaler_{ks}'] = weak_scalers[gk]
                for eidx, em in enumerate(weak_ensembles[gk]):
                    save_dict[f'ensemble_{ks}_model{eidx}_state'] = em.state_dict()
                    ea = ENSEMBLE_CONFIGS[eidx]
                    save_dict[f'ensemble_{ks}_model{eidx}_hidden_dim'] = ea[0]
                    save_dict[f'ensemble_{ks}_model{eidx}_num_blocks'] = ea[1]
                    save_dict[f'ensemble_{ks}_model{eidx}_dropout'] = ea[2]

            # Weights / topk / trimmed
            if gk in ensemble_weights:
                save_dict[f'ensemble_{ks}_weights'] = ensemble_weights[gk]
            if gk in improved_costs:
                ic = improved_costs[gk]
                if ic.get('topk_indices'):
                    save_dict[f'ensemble_{ks}_topk_indices'] = ic['topk_indices']
                    save_dict[f'ensemble_{ks}_topk_weights'] = ic['topk_weights']
                if ic.get('trimmed_keep'):
                    save_dict[f'ensemble_{ks}_trimmed_keep'] = ic['trimmed_keep']

    model_path = os.path.join(args.output_dir, 'moe_model.pth')
    torch.save(save_dict, model_path)
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model saved to '{model_path}' ({file_size:.1f} MB)")
    print(f"  {len(group_keys)} base experts")
    if weak_groups and use_ensemble_for:
        print(f"  {len(use_ensemble_for)} ensemble groups × {ENSEMBLE_SIZE} models")
        for gk in use_ensemble_for:
            print(f"    (k={gk[0]},m={gk[1]}): {best_strategy.get(gk, 'simple')}")

    # =========================================================================
    # STEP 13: Generate Predictions (test = training data for self-eval)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 13: Generate Predictions")
    print("=" * 70)

    test_path = os.path.join(args.data_dir, 'CSCE-636-Project-1-Train-n_k_m_P')
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    test_groups = defaultdict(list)
    for i, sample in enumerate(test_data):
        n, k, m, P = sample
        test_groups[(k, m)].append((i, extract_features(sample)))

    predicted_mheights = [0.0] * len(test_data)

    # Reload checkpoint to ensure test cell logic matches
    checkpoint = torch.save  # we already have everything in memory
    for gk, items in test_groups.items():
        indices = [it[0] for it in items]
        features = np.array([it[1] for it in items], dtype=np.float32)

        if gk not in experts:
            for idx in indices:
                predicted_mheights[idx] = 1.0
            continue

        # Original expert
        orig_sc = group_data[gk]['scaler'].transform(features)
        orig_t = torch.tensor(orig_sc, dtype=torch.float32).to(device)
        orig_preds = batch_predict(experts[gk], orig_t)

        if gk in (use_ensemble_for if weak_groups else []) and gk in weak_ensembles:
            strategy = best_strategy.get(gk, 'simple')

            # Retrained expert
            all_pred_arrays = []
            if gk in retrained_experts:
                ret_sc = retrained_scalers[gk].transform(features)
                ret_t = torch.tensor(ret_sc, dtype=torch.float32).to(device)
                all_pred_arrays.append(batch_predict(retrained_experts[gk], ret_t))

            # Ensemble
            ens_sc = weak_scalers[gk].transform(features)
            ens_t = torch.tensor(ens_sc, dtype=torch.float32).to(device)
            for em in weak_ensembles[gk]:
                all_pred_arrays.append(batch_predict(em, ens_t))

            arr = np.array(all_pred_arrays)

            if strategy == 'weighted' and gk in ensemble_weights:
                w = np.array(ensemble_weights[gk])
                # arr doesn't include original, but weights were computed on full pool
                # full pool = [orig, retrained, ens...]
                full_arr = np.vstack([orig_preds[np.newaxis, :], arr])
                preds_log2 = np.average(full_arr, axis=0, weights=w[:len(full_arr)])
            elif strategy == 'topk' and gk in improved_costs and improved_costs[gk].get('topk_indices'):
                tk_idx = improved_costs[gk]['topk_indices']
                tk_w = np.array(improved_costs[gk]['topk_weights'])
                full_arr = np.vstack([orig_preds[np.newaxis, :], arr])
                ti = [i for i in tk_idx if i < len(full_arr)]
                tw = tk_w[:len(ti)]
                tw = tw / tw.sum()
                preds_log2 = np.average(full_arr[ti], axis=0, weights=tw)
            elif strategy.startswith('trimmed') and gk in improved_costs and improved_costs[gk].get('trimmed_keep'):
                full_arr = np.vstack([orig_preds[np.newaxis, :], arr])
                keep_idx = [i for i in improved_costs[gk]['trimmed_keep'] if i < len(full_arr)]
                preds_log2 = np.mean(full_arr[keep_idx], axis=0)
            elif strategy == 'median':
                full_arr = np.vstack([orig_preds[np.newaxis, :], arr])
                preds_log2 = np.median(full_arr, axis=0)
            elif strategy == 'full':
                full_arr = np.vstack([orig_preds[np.newaxis, :], arr])
                preds_log2 = np.mean(full_arr, axis=0)
            else:
                preds_log2 = np.mean(arr, axis=0)

            source = f"ens-{strategy}({len(all_pred_arrays)})"
        else:
            preds_log2 = orig_preds
            source = "single"

        preds = np.maximum(2.0 ** preds_log2, 1.0)
        for idx, val in zip(indices, preds):
            predicted_mheights[idx] = float(val)

        print(f"  (k={gk[0]},m={gk[1]}) [{source:>25}]: {len(indices)} samples, "
              f"range=[{preds.min():.2f}, {preds.max():.2f}]")

    pred_path = os.path.join(args.output_dir, 'predicted_mHeights')
    with open(pred_path, 'wb') as f:
        pickle.dump(predicted_mheights, f)

    print(f"\nPredictions saved to '{pred_path}'")
    print(f"Total: {len(predicted_mheights)}")
    print(f"Range: [{min(predicted_mheights):.4f}, {max(predicted_mheights):.4f}]")
    print(f"All >= 1: {all(h >= 1.0 for h in predicted_mheights)}")

    print("\n" + "=" * 70)
    print("DONE! 🎉")
    print("=" * 70)

if __name__ == '__main__':
    main()
