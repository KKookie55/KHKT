"""Compute simple accuracy for both baseline and proposed models."""
import sys, os, json
sys.path.insert(0, "/home/z/my-project/scripts")
from run_research import (
    build_dataset, make_splits, train_xgb, train_physics_model, predict_physics,
    FEATURES, MASTER_SEED, set_seed,
)
import numpy as np
from sklearn.metrics import accuracy_score
import math

set_seed(MASTER_SEED)
print("Building dataset...")
df = build_dataset(seed=MASTER_SEED)
splits = make_splits(df)
train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
y_tr = train_df["label"]; y_val = val_df["label"]; y_test = test_df["label"]

# Ensure float64
for f in FEATURES:
    train_df[f] = train_df[f].astype("float64")
    val_df[f] = val_df[f].astype("float64")
    test_df[f] = test_df[f].astype("float64")

print("\nTraining baseline (XGBoost)...")
xgb_model, _ = train_xgb(train_df, y_tr, val_df, y_val, seed=MASTER_SEED)
p_base = xgb_model.predict_proba(test_df[FEATURES])[:, 1]

print("Training proposed (GBDT + Physics MLP)...")
prop_model, prop_scaler, prop_xgb, _ = train_physics_model(
    train_df, y_tr, val_df, y_val, seed=MASTER_SEED, lam=0.20,
    epochs=40, batch_size=1024, lr=1e-3, verbose=False,
)
p_prop = predict_physics(prop_model, prop_scaler, prop_xgb, test_df)

# Compute accuracy at multiple thresholds
y_true = y_test.values
print("\n" + "="*60)
print("ACCURACY (test set)")
print("="*60)
print(f"\nTotal test samples: {len(y_true):,}")
print(f"Positives (fire):   {int(y_true.sum()):,} ({100*y_true.mean():.2f}%)")
print(f"Negatives (no fire): {int((1-y_true).sum()):,} ({100*(1-y_true.mean()):.2f}%)")
print(f"\nNaive baseline (always predict 'no fire'): {100*(1-y_true.mean()):.2f}%")

print(f"\n{'Threshold':<12} {'Baseline Acc':<14} {'Proposed Acc':<14}")
print("-" * 40)
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    b_pred = (p_base >= thr).astype(int)
    p_pred = (p_prop >= thr).astype(int)
    b_acc = accuracy_score(y_true, b_pred)
    p_acc = accuracy_score(y_true, p_pred)
    print(f"{thr:<12.2f} {b_acc*100:<13.2f}% {p_acc*100:<13.2f}%")

# F1-optimal threshold accuracy
from sklearn.metrics import precision_recall_curve, f1_score
prec, rec, thr = precision_recall_curve(y_true, p_base)
f1s = 2*prec*rec/(prec+rec+1e-12)
best_idx = int(np.argmax(f1s[:-1] if len(f1s) > 1 else f1s))
best_thr_b = float(thr[best_idx]) if len(thr) > best_idx else 0.5
prec, rec, thr = precision_recall_curve(y_true, p_prop)
f1s = 2*prec*rec/(prec+rec+1e-12)
best_idx = int(np.argmax(f1s[:-1] if len(f1s) > 1 else f1s))
best_thr_p = float(thr[best_idx]) if len(thr) > best_idx else 0.5

print(f"\n--- At F1-optimal threshold ---")
print(f"Baseline threshold:  {best_thr_b:.4f}")
print(f"  Accuracy:          {accuracy_score(y_true, (p_base >= best_thr_b).astype(int))*100:.2f}%")
print(f"Proposed threshold:  {best_thr_p:.4f}")
print(f"  Accuracy:          {accuracy_score(y_true, (p_prop >= best_thr_p).astype(int))*100:.2f}%")

# Summary
print("\n" + "="*60)
print("ONE-LINE SUMMARY")
print("="*60)
print(f"\nBaseline Accuracy @ 0.5: {accuracy_score(y_true, (p_base >= 0.5).astype(int))*100:.2f}%")
print(f"Proposed Accuracy @ 0.5: {accuracy_score(y_true, (p_prop >= 0.5).astype(int))*100:.2f}%")
print(f"\nBaseline Accuracy @ F1-optimal: {accuracy_score(y_true, (p_base >= best_thr_b).astype(int))*100:.2f}%")
print(f"Proposed Accuracy @ F1-optimal: {accuracy_score(y_true, (p_prop >= best_thr_p).astype(int))*100:.2f}%")
