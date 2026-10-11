#!/usr/bin/env python3
"""
v87_train_dg.py — Giai đoạn 3: Wildfire-DG-MLP.

Kiến trúc: Two-branch MLP
  - Weather branch: 35 weather features → hidden 128
  - Context branch: 3 fire history features → hidden 32
  - Fusion: concat(128, 32) → 160 → 64 → 1 (sigmoid)

Domain-generalization objective: GroupDRO
  - Chia train thành domain theo lat_band (5 bands)
  - Tối ưu worst-group loss thay vì average loss
  - Các domain từ train, KHÔNG dùng test để định nghĩa domain

4 thí nghiệm so sánh:
  1. XGBoost sạch (baseline, đã có)
  2. MLP thường (không DG, weather+context)
  3. Wildfire-DG-MLP (với GroupDRO)
  4. Ablation: DG-MLP không có GroupDRO (= MLP thường → confirm mục 2)

Đánh giá trên: temporal test (2023), cold-start (South, no prior), warm-start (South, with prior)
"""
import os, json, math, pickle, warnings, time
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings('ignore')

OUT = Path("/home/z/my-project/download/v87")
panel = pd.read_parquet(OUT/"panel_v87.parquet")
panel = panel.sort_values(["lat","lon","date"]).reset_index(drop=True)
splits = np.load(OUT/"splits_v87.npz")
meta = json.loads(open(OUT/"meta_v87.json").read())
cold_test = pd.read_parquet(OUT/"test_coldstart.parquet")

WEATHER = meta["feature_sets"]["B1_weather_only"]
HISTORY = meta["feature_sets"]["B2_fire_history_only"]
COMBINED = meta["feature_sets"]["B3_weather_plus_history"]

print("="*60); print("v8.7 GIAI ĐOẠN 3 — WILDFIRE-DG-MLP"); print("="*60)

# ============================================================
# Data prep
# ============================================================
tr = panel.iloc[splits["temporal_train"]].dropna(subset=["label_h1"])
va = panel.iloc[splits["temporal_val"]].dropna(subset=["label_h1"])
te = panel.iloc[splits["temporal_test"]].dropna(subset=["label_h1"])
# Cold-start
tr_cs = panel.iloc[splits["coldstart_train"]].dropna(subset=["label_h1"])
va_cs = panel.iloc[splits["coldstart_val"]].dropna(subset=["label_h1"])
te_cs = cold_test.dropna(subset=["label_h1"])
# Warm-start
tr_ws = panel.iloc[splits["warmstart_train"]].dropna(subset=["label_h1"])
va_ws = panel.iloc[splits["warmstart_val"]].dropna(subset=["label_h1"])
te_ws = panel.iloc[splits["warmstart_test"]].dropna(subset=["label_h1"])

di_idx = COMBINED.index("precip_roll30_sum")

def prep_data(tr, va, te, features):
    """Prepare train/val/test tensors."""
    di_mean = np.nanmean(tr[features[di_idx]].values) if di_idx >= 0 else 0
    di_std = (np.nanstd(tr[features[di_idx]].values) + 1e-6) if di_idx >= 0 else 1
    imp = SimpleImputer(strategy="median")
    def prep(df):
        X = df[features].values.astype(np.float32).copy()
        if di_idx >= 0:
            X[:, di_idx] = np.nan_to_num((X[:, di_idx]-di_mean)/di_std, nan=0.0)
        return X
    X_tr = imp.fit_transform(prep(tr))
    X_val = imp.transform(prep(va))
    X_test = imp.transform(prep(te))
    # Also get weather-only and history-only splits
    n_weather = len(WEATHER)
    X_tr_w = X_tr[:, :n_weather]
    X_tr_h = X_tr[:, n_weather:]
    X_val_w = X_val[:, :n_weather]
    X_val_h = X_val[:, n_weather:]
    X_test_w = X_test[:, :n_weather]
    X_test_h = X_test[:, n_weather:]
    y_tr = tr["label_h1"].values.astype(np.float32)
    y_val = va["label_h1"].values.astype(np.float32)
    y_test = te["label_h1"].values.astype(np.float32)
    # Domain labels (lat_band) for GroupDRO — from train only
    lat_bands = pd.cut(tr["lat"], bins=5, labels=False).fillna(2).astype(int).values
    return (X_tr_w, X_tr_h, X_tr, y_tr, lat_bands,
            X_val_w, X_val_h, X_val, y_val,
            X_test_w, X_test_h, X_test, y_test, imp)

# Prepare temporal data
print("\n[1/3] Preparing temporal data...")
(X_tr_w, X_tr_h, X_tr, y_tr, domains_tr,
 X_val_w, X_val_h, X_val, y_val,
 X_test_w, X_test_h, X_test, y_test, imp_temp) = prep_data(tr, va, te, COMBINED)
print(f"  Train: {len(tr):,} rows, weather={X_tr_w.shape[1]}, history={X_tr_h.shape[1]}")
print(f"  Domains (lat bands): {np.unique(domains_tr, return_counts=True)}")

# Prepare cold-start data
print("Preparing cold-start data...")
(X_cs_tr_w, X_cs_tr_h, X_cs_tr, y_cs_tr, domains_cs,
 X_cs_val_w, X_cs_val_h, X_cs_val, y_cs_val,
 X_cs_test_w, X_cs_test_h, X_cs_test, y_cs_test, imp_cs) = prep_data(tr_cs, va_cs, te_cs, COMBINED)
print(f"  Cold-start train: {len(tr_cs):,}, test: {len(te_cs):,}")

# ============================================================
# Model: Two-branch MLP
# ============================================================
class WildfireDGMLP(nn.Module):
    """Two-branch MLP: weather + context → fusion → fire risk."""
    def __init__(self, n_weather=35, n_history=3, hidden_w=128, hidden_h=32, hidden_fuse=64, dropout=0.15):
        super().__init__()
        # Weather branch
        self.weather_proj = nn.Linear(n_weather, hidden_w)
        self.weather_bn = nn.BatchNorm1d(hidden_w)
        self.weather_drop = nn.Dropout(dropout)
        # Context branch
        self.context_proj = nn.Linear(n_history, hidden_h)
        self.context_bn = nn.BatchNorm1d(hidden_h)
        self.context_drop = nn.Dropout(dropout)
        # Fusion
        self.fuse1 = nn.Linear(hidden_w + hidden_h, hidden_fuse)
        self.fuse_bn = nn.BatchNorm1d(hidden_fuse)
        self.fuse_drop = nn.Dropout(dropout)
        self.fuse2 = nn.Linear(hidden_fuse, 1)
        # Init
        for layer in [self.weather_proj, self.context_proj, self.fuse1]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.fuse2.weight, std=0.01)
        nn.init.zeros_(self.fuse2.bias)

    def forward(self, x_weather, x_history):
        # Weather branch
        w = F.silu(self.weather_proj(x_weather))
        w = self.weather_drop(self.weather_bn(w))
        # Context branch
        h = F.silu(self.context_proj(x_history))
        h = self.context_drop(self.context_bn(h))
        # Fusion
        fused = torch.cat([w, h], dim=1)
        fused = F.silu(self.fuse1(fused))
        fused = self.fuse_drop(self.fuse_bn(fused))
        return self.fuse2(fused).squeeze(-1)

# ============================================================
# Training functions
# ============================================================
def train_mlp(X_tr_w, X_tr_h, y_tr, domains, X_val_w, X_val_h, y_val,
              use_groupdro=False, epochs=5, batch_size=4096, lr=1e-3, seed=42):
    """Train MLP with optional GroupDRO."""
    torch.manual_seed(seed); np.random.seed(seed)
    
    model = WildfireDGMLP(X_tr_w.shape[1], X_tr_h.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    X_t_w = torch.tensor(X_tr_w, dtype=torch.float32)
    X_t_h = torch.tensor(X_tr_h, dtype=torch.float32)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    d_t = torch.tensor(domains, dtype=torch.long)
    
    X_v_w = torch.tensor(X_val_w, dtype=torch.float32)
    X_v_h = torch.tensor(X_val_h, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.float32)
    
    n_domains = int(domains.max() + 1)
    # GroupDRO: learn per-domain penalty weights
    if use_groupdro:
        log_q = torch.zeros(n_domains, requires_grad=True)
        opt_q = torch.optim.SGD([log_q], lr=0.01)
    
    ds = TensorDataset(X_t_w, X_t_h, y_t, d_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    
    best_val_auroc = 0
    best_state = None
    
    for ep in range(epochs):
        model.train()
        for xb_w, xb_h, yb, db in dl:
            opt.zero_grad()
            logits = model(xb_w, xb_h)
            losses = bce(logits, yb)  # per-sample loss
            
            if use_groupdro:
                # GroupDRO: weight each domain by exp(log_q)
                q = torch.softmax(log_q, dim=0)
                # Vectorized GroupDRO: compute per-domain mean loss using scatter
                group_loss = torch.zeros(n_domains, device=logits.device)
                group_counts = torch.zeros(n_domains, device=logits.device)
                group_loss.scatter_add_(0, db, losses)
                group_counts.scatter_add_(0, db, torch.ones_like(db, dtype=torch.float32))
                group_loss = group_loss / (group_counts + 1e-6)
                # Worst-group weighted loss
                weighted_loss = (q * group_loss).sum()
                weighted_loss.backward()
                # Update domain weights
                opt_q.zero_grad()
                group_loss_detached = group_loss.detach()
                q_loss = (torch.softmax(log_q, dim=0) * group_loss_detached).sum()
                q_loss.backward()
                opt_q.step()
            else:
                loss = losses.mean()
                loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(X_v_w, X_v_h)).numpy()
        try:
            val_auroc = roc_auc_score(y_val.numpy(), p_val)
        except:
            val_auroc = 0.5
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Load best
    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate_mlp(model, X_test_w, X_test_h, y_test, X_val_w, X_val_h, y_val):
    """Evaluate MLP + Isotonic calibration."""
    model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(torch.tensor(X_val_w, dtype=torch.float32),
                                     torch.tensor(X_val_h, dtype=torch.float32))).numpy()
        p_test = torch.sigmoid(model(torch.tensor(X_test_w, dtype=torch.float32),
                                      torch.tensor(X_test_h, dtype=torch.float32))).numpy()
    # Iso calibration (fit on val)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_val, y_val)
    p_test_iso = iso.predict(p_test)
    
    auroc = roc_auc_score(y_test, p_test)
    auroc_iso = roc_auc_score(y_test, p_test_iso)
    auprc = average_precision_score(y_test, p_test_iso)
    brier = brier_score_loss(y_test, p_test_iso)
    bins = np.linspace(0, 1, 11); ece = 0; n = len(y_test)
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        ib = ((p_test_iso >= lo) & (p_test_iso <= hi)) if i == 9 else ((p_test_iso >= lo) & (p_test_iso < hi))
        if ib.sum() == 0: continue
        ece += (ib.sum() / n) * abs(y_test[ib].mean() - p_test_iso[ib].mean())
    return {"AUROC": float(auroc), "AUROC_iso": float(auroc_iso), "AUPRC": float(auprc),
            "Brier": float(brier), "ECE_iso": float(ece)}

# ============================================================
# Train 3 models: MLP (no DG), DG-MLP (GroupDRO), DG-MLP ablation
# ============================================================
print(f"\n[2/3] Training models...")

# Model 1: MLP thường (no GroupDRO) — temporal
print("\n  MLP (no DG) — temporal...")
t0 = time.time()
mlp_temp = train_mlp(X_tr_w, X_tr_h, y_tr, domains_tr, X_val_w, X_val_h, y_val, use_groupdro=False)
r_mlp_temp = evaluate_mlp(mlp_temp, X_test_w, X_test_h, y_test, X_val_w, X_val_h, y_val)
print(f"    Temporal: AUROC={r_mlp_temp['AUROC_iso']:.4f} ({time.time()-t0:.0f}s)")

# Model 2: DG-MLP (GroupDRO) — temporal
print("  DG-MLP (GroupDRO) — temporal...")
t0 = time.time()
dg_temp = train_mlp(X_tr_w, X_tr_h, y_tr, domains_tr, X_val_w, X_val_h, y_val, use_groupdro=True)
r_dg_temp = evaluate_mlp(dg_temp, X_test_w, X_test_h, y_test, X_val_w, X_val_h, y_val)
print(f"    Temporal: AUROC={r_dg_temp['AUROC_iso']:.4f} ({time.time()-t0:.0f}s)")

# Evaluate on cold-start
print("\n  Evaluating on cold-start...")
r_mlp_cold = evaluate_mlp(mlp_temp, X_cs_test_w, X_cs_test_h, y_cs_test, X_cs_val_w, X_cs_val_h, y_cs_val)
print(f"    MLP cold-start: AUROC={r_mlp_cold['AUROC_iso']:.4f}")
r_dg_cold = evaluate_mlp(dg_temp, X_cs_test_w, X_cs_test_h, y_cs_test, X_cs_val_w, X_cs_val_h, y_cs_val)
print(f"    DG-MLP cold-start: AUROC={r_dg_cold['AUROC_iso']:.4f}")

# ============================================================
# Results
# ============================================================
print(f"\n[3/3] Results...")

# Load baselines
baselines = json.loads(open(OUT/"baselines_v87.json").read())

results = {
    "1_XGB_clean_B3": {
        "temporal": baselines["B3_combined_A_temporal"],
        "coldstart": baselines["B3_combined_B_coldstart"],
    },
    "2_MLP_no_DG": {
        "temporal": r_mlp_temp,
        "coldstart": r_mlp_cold,
    },
    "3_DG_MLP_GroupDRO": {
        "temporal": r_dg_temp,
        "coldstart": r_dg_cold,
    },
}

print(f"\n{'='*60}")
print("SO SÁNH: XGBoost sạch vs MLP vs DG-MLP")
print(f"{'='*60}")
print(f"\n  {'Model':<25} | {'Temporal':>10} | {'Cold-start':>10} | {'Δ (cold-temp)':>14}")
print(f"  {'-'*65}")
for name, r in results.items():
    t = r["temporal"]["AUROC_iso"]
    c = r["coldstart"]["AUROC_iso"]
    print(f"  {name:<25} | {t:>10.4f} | {c:>10.4f} | {c-t:>+14.4f}")

print(f"\n  Ablation check:")
print(f"    MLP (no DG) vs DG-MLP (GroupDRO):")
print(f"    Temporal:    MLP={r_mlp_temp['AUROC_iso']:.4f} vs DG={r_dg_temp['AUROC_iso']:.4f} Δ={r_dg_temp['AUROC_iso']-r_mlp_temp['AUROC_iso']:+.4f}")
print(f"    Cold-start:  MLP={r_mlp_cold['AUROC_iso']:.4f} vs DG={r_dg_cold['AUROC_iso']:.4f} Δ={r_dg_cold['AUROC_iso']-r_mlp_cold['AUROC_iso']:+.4f}")

# Verdict
print(f"\n  Verdict:")
xgb_cold = baselines["B3_combined_B_coldstart"]["AUROC_iso"]
dg_cold = r_dg_cold["AUROC_iso"]
mlp_cold = r_mlp_cold["AUROC_iso"]
if dg_cold > xgb_cold + 0.01:
    print(f"    ✓ DG-MLP cải thiện cold-start (+{dg_cold-xgb_cold:.4f} vs XGB)")
elif dg_cold > mlp_cold + 0.01:
    print(f"    ✓ GroupDRO cải thiện so với MLP thường (+{dg_cold-mlp_cold:.4f})")
    print(f"    Nhưng chưa vượt XGB baseline ({dg_cold:.4f} vs {xgb_cold:.4f})")
else:
    print(f"    ✗ DG-MLP không cải thiện cold-start")
    print(f"    DG={dg_cold:.4f} vs MLP={mlp_cold:.4f} vs XGB={xgb_cold:.4f}")

# Save
with open(OUT/"dg_mlp_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Saved: {OUT}/dg_mlp_results.json")
