import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
import time

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]

def load_data():
    dfs = []
    for label in LABELS:
        df = pd.read_csv(f"sampled_label_{label}.csv")
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def fill_missing_values(data, method='median', features=None):
    """Fill missing values for specified features (defaults to FEATURES)"""
    if features is None:
        features = FEATURES
    
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

def normalize_data(data, method='minmax', features=None):
    """Normalize data for specified features (defaults to FEATURES)"""
    if features is None:
        features = FEATURES
        
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

def perform_tsne(data, normalized=False, perplexity=30, metric='euclidean', learning_rate='auto'):
    X = data[FEATURES]
    y = data['label']
    if normalized:
        X = normalize_data(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, learning_rate=learning_rate, random_state=42)
    X_tsne = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
    df_tsne['label'] = y.values
    return df_tsne, tsne.kl_divergence_

# Load and process data
data = load_data()
data_filled = fill_missing_values(data)
data_no_outliers, outliers_removed, outlier_mask, inner_outlier_mask, outer_outlier_mask = remove_outliers(data_filled)

# 1. t-SNE on raw data (no normalization)
raw_start_time = time.time()
df_tsne_raw, kl_div_raw = perform_tsne(data_filled, normalized=False)
raw_calculation_time = time.time() - raw_start_time
trust_raw = trustworthiness(data_filled[FEATURES], df_tsne_raw[['Dim1','Dim2']].values, n_neighbors=12)

# 2. t-SNE on normalized data (default params)
norm_start_time = time.time()
data_filled_norm = normalize_data(data_filled)
df_tsne_norm, kl_div_norm = perform_tsne(data_filled, normalized=True)
norm_calculation_time = time.time() - norm_start_time
trust_norm = trustworthiness(data_filled_norm[FEATURES], df_tsne_norm[['Dim1','Dim2']].values, n_neighbors=12)

# 3. Separate graphs for raw and normalized t-SNE with time counting

# plt.figure(figsize=(10, 8))
# for label in sorted(df_tsne_raw['label'].unique()):
#     subset = df_tsne_raw[df_tsne_raw['label'] == label]
#     plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
# plt.title(f't-SNE Raw Data\nKL={kl_div_raw:.4f} | Trust={trust_raw:.4f} | Time={raw_calculation_time:.3f}s')
# plt.xlabel('Dim1')
# plt.ylabel('Dim2')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # Normalized data graph
# plt.figure(figsize=(10, 8))
# for label in sorted(df_tsne_norm['label'].unique()):
#     subset = df_tsne_norm[df_tsne_norm['label'] == label]
#     plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
# plt.title(f't-SNE Normalized Data\nKL={kl_div_norm:.4f} | Trust={trust_norm:.4f} | Time={norm_calculation_time:.3f}s')
# plt.xlabel('Dim1')
# plt.ylabel('Dim2')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.show()

# # 4. Independent t-SNE graphs with normalized data and different perplexities/learning rates
# perplexities = [5, 25, 50]
# metrics = ['euclidean', 'cosine', 'manhattan']

# for perplexity in perplexities:
#     for metric in metrics:
#         start_time = time.time()
#         df_tsne, kl_div = perform_tsne(data_filled_norm, normalized=False, perplexity=perplexity, metric=metric, learning_rate='auto')
#         elapsed_time = time.time() - start_time
#         df_tsne['is_outer_outlier'] = outer_outlier_mask.values
#         df_tsne['is_inner_outlier'] = inner_outlier_mask.values & ~outer_outlier_mask.values
#         trust = trustworthiness(data_filled_norm[FEATURES], df_tsne[['Dim1','Dim2']].values, n_neighbors=12)
#         plt.figure(figsize=(10, 8))
#         colors = ['blue', 'orange', 'green']
#         for i, label in enumerate(sorted(df_tsne['label'].unique())):
#             subset = df_tsne[df_tsne['label'] == label]
#             color = colors[i]
#             normal_points = subset[~subset['is_inner_outlier'] & ~subset['is_outer_outlier']]
#             inner_points = subset[subset['is_inner_outlier']]
#             outer_points = subset[subset['is_outer_outlier']]
#             if len(normal_points) > 0:
#                 plt.scatter(normal_points['Dim1'], normal_points['Dim2'], c=color, alpha=0.7, marker='o', s=50)
#             if len(inner_points) > 0:
#                 plt.scatter(inner_points['Dim1'], inner_points['Dim2'], c=color, alpha=0.9, marker='s', s=80, edgecolors='black', linewidth=1)
#             if len(outer_points) > 0:
#                 plt.scatter(outer_points['Dim1'], outer_points['Dim2'], c=color, alpha=0.9, marker='^', s=80, edgecolors='black', linewidth=1)
#         for i in range(3):
#             plt.scatter([], [], c=colors[i], marker='o', s=50, label=f'Label {i} Normal')
#             plt.scatter([], [], c=colors[i], marker='s', s=80, edgecolors='black', linewidth=1, label=f'Label {i} Inner Outlier')
#             plt.scatter([], [], c=colors[i], marker='^', s=80, edgecolors='black', linewidth=1, label=f'Label {i} Outer Outlier')
#         plt.title(f't-SNE (perplexity={perplexity}, metric={metric})\nKL={kl_div:.4f} | Trust={trust:.4f}\nTime={elapsed_time:.2f}s')
#         plt.xlabel('Dim1')
#         plt.ylabel('Dim2')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()

#     total_points = len(df_tsne)
#     total_outliers = outlier_mask.sum()
#     total_inner = (inner_outlier_mask & ~outer_outlier_mask).sum()
#     total_outer = outer_outlier_mask.sum()
#     percent_outliers = 100 * total_outliers / total_points
#     percent_inner = 100 * total_inner / total_points
#     percent_outer = 100 * total_outer / total_points
#     print(f"Total points: {total_points}")
#     print(f"Total outliers: {total_outliers} ({percent_outliers:.2f}%)")
#     print(f"  Inner outliers: {total_inner} ({percent_inner:.2f}%)")
#     print(f"  Outer outliers: {total_outer} ({percent_outer:.2f}%)")

# sample_data = load_data()
# if 'label' in sample_data.columns:
#     numeric_columns = sample_data.select_dtypes(include=[np.number]).columns.tolist()
#     if 'label' in numeric_columns:
#         numeric_columns.remove('label')
#     all_features = numeric_columns
#     if len(all_features) > 0:

#         sample_data_clean = sample_data[all_features + ['label']].dropna()
#         sample_filled = fill_missing_values(sample_data_clean, features=all_features)
#         sample_raw_start = time.time()
#         X_sample_raw = sample_filled[all_features]
#         y_sample_raw = sample_filled['label']
#         tsne_sample_raw = TSNE(n_components=2, random_state=42)
#         X_tsne_sample_raw = tsne_sample_raw.fit_transform(X_sample_raw)
#         df_tsne_sample_raw = pd.DataFrame(X_tsne_sample_raw, columns=['Dim1', 'Dim2'])
#         df_tsne_sample_raw['label'] = y_sample_raw.values
#         kl_div_sample_raw = tsne_sample_raw.kl_divergence_
#         trust_sample_raw = trustworthiness(X_sample_raw, df_tsne_sample_raw[['Dim1','Dim2']].values, n_neighbors=12)
#         sample_raw_time = time.time() - sample_raw_start


#         sample_norm_start = time.time()
#         sample_normalized = normalize_data(sample_filled, features=all_features)
#         X_sample_norm = sample_normalized[all_features]
#         y_sample_norm = sample_normalized['label']
#         tsne_sample_norm = TSNE(n_components=2, random_state=42)
#         X_tsne_sample_norm = tsne_sample_norm.fit_transform(X_sample_norm)
#         df_tsne_sample_norm = pd.DataFrame(X_tsne_sample_norm, columns=['Dim1', 'Dim2'])
#         df_tsne_sample_norm['label'] = y_sample_norm.values
#         kl_div_sample_norm = tsne_sample_norm.kl_divergence_
#         trust_sample_norm = trustworthiness(X_sample_norm, df_tsne_sample_norm[['Dim1','Dim2']].values, n_neighbors=12)
#         sample_norm_time = time.time() - sample_norm_start


#         plt.figure(figsize=(10, 8))
#         for label in sorted(df_tsne_sample_raw['label'].unique()):
#             subset = df_tsne_sample_raw[df_tsne_sample_raw['label'] == label]
#             plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
#         plt.title(f'Sample t-SNE Raw Data (ALL FEATURES)\nKL={kl_div_sample_raw:.4f} | Trust={trust_sample_raw:.4f} | Time={sample_raw_time:.3f}s')
#         plt.xlabel('Dim1')
#         plt.ylabel('Dim2')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()
#         plt.figure(figsize=(10, 8))
#         for label in sorted(df_tsne_sample_norm['label'].unique()):
#             subset = df_tsne_sample_norm[df_tsne_sample_norm['label'] == label]
#             plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
#         plt.title(f'Sample t-SNE Normalized Data (ALL FEATURES)\nKL={kl_div_sample_norm:.4f} | Trust={trust_sample_norm:.4f} | Time={sample_norm_time:.3f}s')
#         plt.xlabel('Dim1')
#         plt.ylabel('Dim2')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No numeric features found in sample data")
# else:
#     print("'label' column not found in sample data")

# Simple concise summaries (like .describe) for filled and normalized data
print("data_filled (features) summary:")
desc_filled = data_filled[FEATURES].describe().T
desc_filled['median'] = data_filled[FEATURES].median()
desc_filled['dispersion'] = data_filled[FEATURES].var()
print(desc_filled[['count','mean','std','min','25%','50%','75%','max','median','dispersion']])

print("\ndata_filled_norm (features) summary:")
desc_norm = data_filled_norm[FEATURES].describe().T
desc_norm['median'] = data_filled_norm[FEATURES].median()
desc_norm['dispersion'] = data_filled_norm[FEATURES].var()
print(desc_norm[['count','mean','std','min','25%','50%','75%','max','median','dispersion']])
