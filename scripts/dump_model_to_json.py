#!/usr/bin/env python3
"""
Dump the XGBoost model + isotonic calibrator to a compact JSON format
that can be loaded by a JS inference module.

Output: /home/z/my-project/model/wildfire_v86_model/v86_compact.json
"""
import json
import pickle
import os
import numpy as np
import xgboost as xgb

MODEL_DIR = '/home/z/my-project/model/wildfire_v86_model'
XGB_PATH = os.path.join(MODEL_DIR, 'xgb_modis_iid.json')
ISO_PATH = os.path.join(MODEL_DIR, 'iso_calibrator.pkl')
IMP_PATH = os.path.join(MODEL_DIR, 'imputer.pkl')
META_PATH = os.path.join(MODEL_DIR, 'model_meta.json')
OUT_PATH = os.path.join(MODEL_DIR, 'v86_compact.json')

# Load model
model = xgb.XGBClassifier()
model.load_model(XGB_PATH)

# Load calibrator
with open(ISO_PATH, 'rb') as f:
    iso = pickle.load(f)

# Load imputer
with open(IMP_PATH, 'rb') as f:
    imp = pickle.load(f)

# Load meta
with open(META_PATH) as f:
    meta = json.load(f)

# Extract trees from the XGBoost model
# model.get_booster().get_dump() gives us text format
# But we can use the JSON dump for precise structure
booster = model.get_booster()
tree_dumps = booster.get_dump(dump_format='json')
trees = [json.loads(td) for td in tree_dumps]

print(f"Number of trees: {len(trees)}")
print(f"First tree sample structure:")
print(json.dumps(trees[0], indent=2)[:500])

# Extract isotonic calibrator data
# IsotonicRegression has attributes:
# - X_min_, X_max_ (the input range)
# - X_thresholds_ (the unique x values)
# - y_thresholds_ (the corresponding y values)
iso_data = {
    "x_thresholds": iso.X_thresholds_.tolist() if hasattr(iso, 'X_thresholds_') else None,
    "y_thresholds": iso.y_thresholds_.tolist() if hasattr(iso, 'y_thresholds_') else None,
    "x_min": float(iso.X_min_) if hasattr(iso, 'X_min_') else None,
    "x_max": float(iso.X_max_) if hasattr(iso, 'X_max_') else None,
}
print(f"\nIsotonic: {len(iso_data['x_thresholds'])} thresholds")
print(f"  x range: [{iso_data['x_min']}, {iso_data['x_max']}]")
print(f"  y range: [{min(iso_data['y_thresholds'])}, {max(iso_data['y_thresholds'])}]")

# Extract imputer data
# SimpleImputer stores: statistics_ (per-feature fill values)
if hasattr(imp, 'statistics_'):
    imp_data = {
        "statistics": imp.statistics_.tolist(),
        "strategy": imp.strategy,
    }
    print(f"\nImputer: strategy={imp.strategy}, {len(imp_data['statistics'])} features")
else:
    imp_data = {"statistics": [], "strategy": "unknown"}

# Simplify trees for compact JS representation
# Each tree node: {nodeid, split (feature idx), split_condition, yes (child), no (child), leaf (if leaf)}
# We'll convert to a flat array per tree: [(feature_idx, threshold, left_child, right_child, leaf_value), ...]
def simplify_tree(node):
    """Convert XGBoost tree node to compact format."""
    if 'leaf' in node:
        return {
            "id": node['nodeid'],
            "leaf": node['leaf'],
            "cover": node.get('cover', 0),
        }
    else:
        return {
            "id": node['nodeid'],
            "split": node.get('split', -1),
            "threshold": node.get('split_condition', 0),
            "yes": node.get('yes', -1),
            "no": node.get('no', -1),
            "missing": node.get('missing', node.get('yes', -1)),
            "children": [
                simplify_tree(c) for c in node.get('children', [])
            ]
        }

compact_trees = []
for t in trees:
    compact_trees.append(simplify_tree(t))

print(f"\nCompact trees: {len(compact_trees)}")
print(f"Sample first tree keys: {list(compact_trees[0].keys()) if compact_trees else 'empty'}")

# Combine into compact JSON
output = {
    "version": "v8.6_canonical_model",
    "n_features": meta['n_features'],
    "feature_names": meta['features'],
    "drought_idx": meta['drought_idx'],
    "drought_mean": meta['drought_mean'],
    "drought_std": meta['drought_std'],
    "num_trees": len(compact_trees),
    "trees": compact_trees,
    "iso_calibrator": iso_data,
    "imputer": imp_data,
    "performance": meta['performance'],
    "region_penalty": {"north": 1.0, "central": 0.95, "south": 0.65},
}

# Save
with open(OUT_PATH, 'w') as f:
    json.dump(output, f)

file_size = os.path.getsize(OUT_PATH)
print(f"\n[OK] Saved: {OUT_PATH}")
print(f"     Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
