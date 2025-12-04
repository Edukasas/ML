import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1]

REMOVE_OUTLIERS_BEFORE_NORMALIZATION = False
USE_TSNE_FOR_CLUSTERING = True
TSNE_PERPLEXITY = 50
RANDOM_STATE = 42

# ============================================================================
# DATA PREPARATION
# ============================================================================

def load_data():
    """Load and combine CSV files for all labels."""
    dfs = [pd.read_csv(f"sampled_label_{label}.csv").assign(label=label) for label in LABELS]
    return pd.concat(dfs, ignore_index=True)

def remove_outliers(df, features):
    """Remove outliers using IQR method. Returns cleaned data and masks."""
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1

    inner_mask = ((df[features] < Q1 - 1.5 * IQR) |
                  (df[features] > Q3 + 1.5 * IQR)).any(axis=1)
    outer_mask = ((df[features] < Q1 - 3 * IQR) |
                  (df[features] > Q3 + 3 * IQR)).any(axis=1)

    outlier_mask = inner_mask | outer_mask
    cleaned = df[~outlier_mask].copy()

    return cleaned, outlier_mask, inner_mask, outer_mask

def prepare_data(df, features, remove_outliers_first=False):
    """Fill missing values and normalize features. Optionally remove outliers first."""
    df = df.copy()

    # Fill missing values with group median
    for feature in features:
        if feature in df.columns:
            df[feature] = df.groupby("label")[feature].transform(
                lambda x: x.fillna(x.median())
            )

    # Remove outliers before normalization if requested
    outlier_info = None
    if remove_outliers_first:
        df, outlier_mask, inner_mask, outer_mask = remove_outliers(df, features)
        outlier_info = {
            'outlier_mask': outlier_mask,
            'inner_mask': inner_mask,
            'outer_mask': outer_mask,
            'n_outliers': outlier_mask.sum(),
            'n_mild': (inner_mask & ~outer_mask).sum(),
            'n_extreme': outer_mask.sum()
        }
        print(f"  Removed {outlier_info['n_outliers']} outliers "
              f"({outlier_info['n_mild']} inner, {outlier_info['n_extreme']} outer)")

    # Normalize to [0, 1]
    for feature in features:
        if feature in df.columns:
            x_min, x_max = df[feature].min(), df[feature].max()
            if x_max != x_min:
                df[feature] = (df[feature] - x_min) / (x_max - x_min)

    return (df, outlier_info) if remove_outliers_first else df

# 1) Load the two files into one DataFrame
df = pd.concat([pd.read_csv(f"sampled_label_{l}.csv").assign(label=l) for l in LABELS],
               ignore_index=True)  # read and concat CSVs [web:79][web:73]

error_tokens = ["#NAME?", "#DIV/0!", "#VALUE!", "#N/A", "#NULL!", "#REF!", "#NUM!"]
df = df.replace(error_tokens, np.nan)  # normalize errors to NaN [web:136]

# 2) Coerce all non-label columns to numeric where possible
for c in df.columns:
    if c != "label":
        df[c] = pd.to_numeric(df[c], errors="coerce")  # non-numeric → NaN [web:136]
        
# select all numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns

# avoid scaling the label itself
feat_cols = num_cols.drop("label") if "label" in num_cols else num_cols

# 2) Prepare once
df_prepared = prepare_data(df, FEATURES, remove_outliers_first=False)  # run your function [web:24]

# 3) Save the result as a new CSV (or overwrite if you prefer)
df_prepared.to_csv("sampled_all_data.csv", index=False)  # simple single write [web:87][web:81]

# X = df_prepared[FEATURES].values
# tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=RANDOM_STATE, init="random", learning_rate="auto")  # API usage [web:108]
# Z = tsne.fit_transform(X)  # returns 2D coordinates [web:108]

# df_tsne = df_prepared.copy()
# df_tsne["tsne_x"] = Z[:, 0]
# df_tsne["tsne_y"] = Z[:, 1]
# df_tsne.to_csv("sampled_tsne_prepared.csv", index=False)