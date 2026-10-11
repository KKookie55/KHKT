"""
Export trained model predictions to CSV for reproducibility audit.
Generates prediction CSVs for:
  - Test set (2022)
  - OOD temporal (2023)
  - OOD spatial (Khánh Hòa)

Output columns: y_true, y_pred, probability, date, lat, lon, split
Also exports raw XGBoost probabilities and per-ensemble-member probabilities
for full transparency.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/z/my-project/scripts")

from run_research import (
    build_dataset, make_splits, train_xgb, train_physics_model,
    predict_physics, train_physics_ensemble, predict_ensemble,
    compute_metrics, FEATURES, MASTER_SEED, set_seed,
)

OUT_DIR = "/home/z/my-project/download/predictions"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
set_seed(MASTER_SEED)
print("=" * 70)
print("EXPORTING MODEL PREDICTIONS TO CSV")
print("=" * 70)

# Step 1: Build dataset
print("\n[1/4] Building dataset...")
df = build_dataset(seed=MASTER_SEED)
splits = make_splits(df)

# Cast features to float64
for f in FEATURES:
    for split_name in ["train", "val", "test", "ood_time", "ood_space"]:
        splits[split_name][f] = splits[split_name][f].astype("float64")

train_df = splits["train"]
val_df = splits["val"]
test_df = splits["test"]
ood_time_df = splits["ood_time"]
ood_space_df = splits["ood_space"]

y_tr = train_df["label"]
y_val = val_df["label"]

# Step 2: Train XGBoost baseline
print("\n[2/4] Training XGBoost baseline...")
xgb_model, _ = train_xgb(train_df, y_tr, val_df, val_df["label"], seed=MASTER_SEED)

# Step 3: Train proposed ensemble (3 MLPs with seeds 42, 142, 242)
print("\n[3/4] Training proposed ensemble (3 MLP members)...")
ensemble = train_physics_ensemble(
    train_df, y_tr, val_df, val_df["label"],
    n_models=3, base_seed=MASTER_SEED, lam=0.20,
    epochs=25, batch_size=1024, lr=1e-3, verbose=False,
)

# Step 4: Generate predictions for each split and save to CSV
print("\n[4/4] Generating predictions and saving CSVs...")

def to_csv(split_df, split_name, xgb_model, ensemble):
    """Generate predictions CSV for a split."""
    # XGBoost raw predictions
    p_xgb = xgb_model.predict_proba(split_df[FEATURES])[:, 1]
    # Per-ensemble-member predictions (3 members)
    member_preds = []
    for i, (m, sc, xg, _) in enumerate(ensemble):
        p_member = predict_physics(m, sc, xg, split_df)
        member_preds.append(p_member)
    # Ensemble averaged predictions
    p_ensemble = np.mean(member_preds, axis=0)
    # Binary predictions at 0.5 threshold
    y_pred_xgb = (p_xgb >= 0.5).astype(int)
    y_pred_ensemble = (p_ensemble >= 0.5).astype(int)

    out_df = pd.DataFrame({
        "date": split_df["date"].astype(str).values,
        "lat": split_df["lat"].values,
        "lon": split_df["lon"].values,
        "year": split_df["year"].values,
        "month": split_df["month"].values,
        "doy": split_df["doy"].values,
        "y_true": split_df["label"].values,
        "y_pred_xgb": y_pred_xgb,
        "prob_xgb": p_xgb,
        "y_pred_ensemble": y_pred_ensemble,
        "prob_ensemble": p_ensemble,
        "prob_member1_seed42": member_preds[0],
        "prob_member2_seed142": member_preds[1],
        "prob_member3_seed242": member_preds[2],
        "prob_true": split_df["prob_true"].values,
        "split": split_name,
    })

    csv_path = f"{OUT_DIR}/predictions_{split_name}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}: {len(out_df):,} rows × {len(out_df.columns)} cols")

    # Print quick metrics
    y = split_df["label"].values
    m_xgb = compute_metrics(y, p_xgb)
    m_ens = compute_metrics(y, p_ensemble)
    print(f"    XGB:       AUROC={m_xgb['AUROC']:.4f}  ECE={m_xgb['ECE']:.4f}  Brier={m_xgb['Brier']:.4f}")
    print(f"    Ensemble:  AUROC={m_ens['AUROC']:.4f}  ECE={m_ens['ECE']:.4f}  Brier={m_ens['Brier']:.4f}")
    return out_df

# Generate predictions for each split
test_out = to_csv(test_df, "test_2022", xgb_model, ensemble)
ood_time_out = to_csv(ood_time_df, "ood_time_2023", xgb_model, ensemble)
if len(ood_space_df) > 0:
    ood_space_out = to_csv(ood_space_df, "ood_space_khanh_hoa", xgb_model, ensemble)

# Combined CSV with all splits
combined = pd.concat([test_out, ood_time_out], ignore_index=True)
if len(ood_space_df) > 0:
    combined = pd.concat([combined, ood_space_out], ignore_index=True)
combined_path = f"{OUT_DIR}/predictions_ALL.csv"
combined.to_csv(combined_path, index=False)
print(f"\n  Combined CSV: {combined_path}: {len(combined):,} rows")

# Step 5: Also export the results.json (already exists, copy to download)
import shutil
results_src = "/home/z/my-project/scripts/results.json"
results_dst = "/home/z/my-project/download/results.json"
shutil.copy(results_src, results_dst)
print(f"\n  Copied results JSON: {results_dst}")

# Step 6: Summary
print("\n" + "=" * 70)
print("EXPORT COMPLETE")
print("=" * 70)
print(f"\nAll files saved to: {OUT_DIR}/")
print(f"\nFiles:")
for fn in sorted(os.listdir(OUT_DIR)):
    fp = os.path.join(OUT_DIR, fn)
    print(f"  {fn}: {os.path.getsize(fp)/1024:.1f} KB")

print(f"\nTotal samples exported:")
print(f"  Test set (2022):              {len(test_out):,}")
print(f"  OOD temporal (2023):          {len(ood_time_out):,}")
print(f"  OOD spatial (Khánh Hòa):      {len(ood_space_out) if len(ood_space_df)>0 else 0:,}")
print(f"  Total:                         {len(combined):,}")
print(f"\nCSV columns:")
print(f"  date, lat, lon, year, month, doy,")
print(f"  y_true (binary label),")
print(f"  y_pred_xgb, prob_xgb,")
print(f"  y_pred_ensemble, prob_ensemble,")
print(f"  prob_member1_seed42, prob_member2_seed142, prob_member3_seed242,")
print(f"  prob_true (synthetic ground-truth probability — for audit only),")
print(f"  split (test_2022 / ood_time_2023 / ood_space_khanh_hoa)")
