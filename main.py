import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

def fill_missing_values(data, method='median'):
    data_filled = data.copy()
    for feature in FEATURES:
        if method == 'mean':
            data_filled[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.mean()))
        else:
            data_filled[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.median()))
    return data_filled

def normalize_data(data, method='minmax'):
    data_normalized = data.copy()
    for feature in FEATURES:
        if method == 'minmax':
            x_min = data[feature].min()
            x_max = data[feature].max()
            data_normalized[feature] = (data[feature] - x_min) / (x_max - x_min)
        else: 
            x_mean = data[feature].mean()
            x_std = data[feature].std()
            data_normalized[feature] = (data[feature] - x_mean) / x_std
    return data_normalized

def perform_tsne(data, normalized=False):
    X = data[FEATURES]
    y = data['label']
    if normalized:
        X = normalize_data(X)
    tsne = TSNE(n_components=2, perplexity=50, random_state=42)
    X_tsne = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
    df_tsne['label'] = y.values
    plt.figure(figsize=(8, 6))
    for label in sorted(df_tsne['label'].unique()):
        subset = df_tsne[df_tsne['label'] == label]
        plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
    plt.title(f't-SNE Visualization ({"Normalized" if normalized else "Raw"} Data)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Make sure data_no_outliers is defined before using it in the t-SNE grid!
data = load_data()
data_filled = fill_missing_values(data)

# 1. t-SNE on raw data (no normalization)
X_raw = data_filled[FEATURES]
y_raw = data_filled['label']
tsne_raw = TSNE(n_components=2, random_state=42)
X_tsne_raw = tsne_raw.fit_transform(X_raw)
df_tsne_raw = pd.DataFrame(X_tsne_raw, columns=['Dim1', 'Dim2'])
df_tsne_raw['label'] = y_raw.values
kl_div_raw = tsne_raw.kl_divergence_

# 2. t-SNE on normalized data (default params)
data_filled_norm = normalize_data(data_filled)
X_norm = data_filled_norm[FEATURES]
y_norm = data_filled_norm['label']
tsne_norm = TSNE(n_components=2, random_state=42)
X_tsne_norm = tsne_norm.fit_transform(X_norm)
df_tsne_norm = pd.DataFrame(X_tsne_norm, columns=['Dim1', 'Dim2'])
df_tsne_norm['label'] = y_norm.values
kl_div_norm = tsne_norm.kl_divergence_

# 3. Compare raw vs normalized t-SNE
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
for label in sorted(df_tsne_raw['label'].unique()):
    subset = df_tsne_raw[df_tsne_raw['label'] == label]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
plt.title(f't-SNE Raw Data\nKL={kl_div_raw:.4f}')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for label in sorted(df_tsne_norm['label'].unique()):
    subset = df_tsne_norm[df_tsne_norm['label'] == label]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
plt.title(f't-SNE Normalized Data\nKL={kl_div_norm:.4f}')
plt.xlabel('Dim1')
plt.ylabel('Dim2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nt-SNE embedding (raw data, default params):")
print(df_tsne_raw.head())
print(f"KL Divergence (raw): {kl_div_raw:.4f}")

print("\nt-SNE embedding (normalized data, default params):")
print(df_tsne_norm.head())
print(f"KL Divergence (normalized): {kl_div_norm:.4f}")

# 4. Independent t-SNE graphs with normalized data and different perplexities/learning rates
perplexities = [5, 25, 50]
learning_rates = [250, 500, 750]

for perplexity in perplexities:
    for lr in learning_rates:
        start_time = time.time()
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, random_state=42)
        X = data_filled_norm[FEATURES]
        y = data_filled_norm['label']
        X_tsne = tsne.fit_transform(X)
        elapsed_time = time.time() - start_time
        df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
        df_tsne['label'] = y.values
        plt.figure(figsize=(8, 6))
        for label in sorted(df_tsne['label'].unique()):
            subset = df_tsne[df_tsne['label'] == label]
            plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
        kl_divergence = tsne.kl_divergence_
        plt.title(f't-SNE (perplexity={perplexity}, lr={lr})\nKL={kl_divergence:.4f}\nTime={elapsed_time:.2f}s')
        plt.xlabel('Dim1')
        plt.ylabel('Dim2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print(f"\nt-SNE embedding (perplexity={perplexity}, learning_rate={lr}):")
        print(df_tsne.head())
        print(f"KL Divergence: {kl_divergence:.4f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")