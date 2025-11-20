import pandas as pd
import numpy as np

INPUT_FILE = 'clustered_selected_features.csv'
OUTPUT_FILE = 'cluster_descriptive_statistics.xlsx'
FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]

# Load
df = pd.read_csv(INPUT_FILE)

# Validate required columns
missing = [c for c in ['cluster'] + FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Aggregations
agg_funcs = {
    'count': 'count',
    'mean': 'mean',
    'std': 'std',
    'min': 'min',
    'q25': lambda s: s.quantile(0.25),
    'median': 'median',
    'q75': lambda s: s.quantile(0.75),
    'max': 'max',
    'dispersion': 'var',
}

# Build MultiIndex columns cleanly
pieces = []
for feat in FEATURES:
    g = df.groupby('cluster')[[feat]]
    part = g.agg(**{
        'count': (feat, 'count'),
        'mean': (feat, 'mean'),
        'std': (feat, 'std'),
        'min': (feat, 'min'),
        'q25': (feat, lambda s: s.quantile(0.25)),
        'median': (feat, 'median'),
        'q75': (feat, lambda s: s.quantile(0.75)),
        'max': (feat, 'max'),
        'dispersion': (feat, 'var'),
    })
    part.index.name = 'Cluster'
    part.columns = pd.MultiIndex.from_product([[feat], part.columns])
    pieces.append(part)

wide = pd.concat(pieces, axis=1).sort_index()

# Round numeric stats except count
for feat in FEATURES:
    for stat in ['mean','std','min','q25','median','q75','max','dispersion']:
        col = (feat, stat)
        if col in wide.columns:
            wide[col] = wide[col].astype(float).round(3)

# Excel writers often dislike nested renamers; flatten just for export
flat_cols = ['{} | {}'.format(top, sub) for top, sub in wide.columns.to_flat_index()]
wide_export = wide.copy()
wide_export.columns = flat_cols

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    wide_export.to_excel(writer, sheet_name='Cluster_Stats', index=True)

print(f"Wrote one-sheet cluster stats to {OUTPUT_FILE} with shape {wide.shape}")
