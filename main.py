import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]

def load_data():
    dfs = []
    for label in LABELS:
        df = pd.read_csv(f"sampled_label_{label}.csv")
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_values(data, method='median', features=FEATURES):
    """Fill missing values for specified features (defaults to FEATURES)"""
    data_filled = data.copy()
    for feature in features:
        if feature in data.columns:
            if method == 'mean':
                if 'label' in data.columns:
                    data_filled[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.mean()))
                else:
                    data_filled[feature] = data_filled[feature].fillna(data_filled[feature].mean())
            else:
                if 'label' in data.columns:
                    data_filled[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.median()))
                else:
                    data_filled[feature] = data_filled[feature].fillna(data_filled[feature].median())
    return data_filled

def normalize_data(data, method='minmax', features=FEATURES):
    """Normalize data for specified features (defaults to FEATURES)"""
    data_normalized = data.copy()
    for feature in features:
        if feature in data.columns:
            if method == 'minmax':
                x_min = data[feature].min()
                x_max = data[feature].max()
                if x_max != x_min:  # Avoid division by zero
                    data_normalized[feature] = (data[feature] - x_min) / (x_max - x_min)
            else: 
                x_mean = data[feature].mean()
                x_std = data[feature].std()
                if x_std != 0:  # Avoid division by zero
                    data_normalized[feature] = (data[feature] - x_mean) / x_std
    return data_normalized

def remove_outliers(data):
    """
    Detects both inner (mild) and outer (extreme) outliers using the IQR method.
    Returns:
        cleaned_data: DataFrame with no outer or inner outliers
        outliers_data: DataFrame with all detected outliers (inner + outer)
        outlier_mask: Boolean mask for all outliers (inner + outer)
        inner_outlier_mask: Boolean mask for inner (mild) outliers only
        outer_outlier_mask: Boolean mask for outer (extreme) outliers only
    """
    data_features = data[FEATURES]
    Q1 = data_features.quantile(0.25)
    Q3 = data_features.quantile(0.75)
    IQR = Q3 - Q1

    # Inner outliers: 1.5*IQR
    lower_inner = Q1 - 1.5 * IQR
    upper_inner = Q3 + 1.5 * IQR
    inner_outlier_mask = ((data_features < lower_inner) | (data_features > upper_inner)).any(axis=1)

    # Outer outliers: 3*IQR
    lower_outer = Q1 - 3 * IQR
    upper_outer = Q3 + 3 * IQR
    outer_outlier_mask = ((data_features < lower_outer) | (data_features > upper_outer)).any(axis=1)

    # Any outlier (inner or outer)
    outlier_mask = inner_outlier_mask | outer_outlier_mask

    cleaned_data = data[~outlier_mask].copy()
    outliers_data = data[outlier_mask].copy()

    return cleaned_data, outliers_data, outlier_mask, inner_outlier_mask, outer_outlier_mask

# def perform_tsne(data, normalized=False, perplexity=30, metric='euclidean', learning_rate='auto':
#     X = data[FEATURES]
#     y = data['label']
#     if normalized:
#         X = normalize_data(X)
#     tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, learning_rate=learning_rate, random_state=42)
def evaluate_clusters_libs(data, features, k_min=2, k_max=10, random_state=42, excel_path="cluster_selection.xlsx"):
    X = data[features].values
    ks = range(k_min, k_max + 1)
    
    results = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        sil_score = silhouette_score(X, km.labels_) if k > 1 else float('nan')
        results.append({
            'k': k,
            'inertia': km.inertia_,
            'silhouette': sil_score
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_path, index=False)
    
    return {
        'elbow': results_df,
        'optimal_k': results_df.loc[results_df['silhouette'].idxmax(), 'k']
    }


data_load = load_data()
data_filled = fill_missing_values(data_load)
data_normal = normalize_data(data_filled)

results = evaluate_clusters_libs(data_normal, FEATURES, k_min=2, k_max=10, random_state=42, excel_path="cluster_selection.xlsx")

print("Optimal k:", results.get("optimal_k"))
print("Elbow table:\n", results["elbow"].head())

