import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

def remove_outliers(data):
    Q1 = data[FEATURES].quantile(0.25)
    Q3 = data[FEATURES].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    mask = ~((data[FEATURES] < lower_bound) | (data[FEATURES] > upper_bound)).any(axis=1)
    return data[mask].reset_index(drop=True)

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
data_no_outliers = remove_outliers(data_filled)

# t-SNE parameter grids
perplexities = [5, 25, 50]         # low, mid, high
learning_rates = [10, 500, 1000]  # low, mid, high

# Prepare normalized data
data_no_outliers_norm = normalize_data(data_no_outliers)

# Combine all t-SNE results into one matrix of subplots
fig, axes = plt.subplots(len(perplexities), len(learning_rates), figsize=(18, 18), squeeze=False)

for i, perplexity in enumerate(perplexities):
    for j, lr in enumerate(learning_rates):
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=lr, random_state=42)
        X = data_no_outliers_norm[FEATURES]
        y = data_no_outliers_norm['label']
        X_tsne = tsne.fit_transform(X)
        df_tsne = pd.DataFrame(X_tsne, columns=['Dim1', 'Dim2'])
        df_tsne['label'] = y.values
        ax = axes[i, j]
        for label in sorted(df_tsne['label'].unique()):
            subset = df_tsne[df_tsne['label'] == label]
            ax.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
        kl_divergence = tsne.kl_divergence_
        ax.set_title(f'perplexity={perplexity}, lr={lr}\nKL={kl_divergence:.2f}')
        ax.set_xlabel('Dim1')
        ax.set_ylabel('Dim2')
        ax.grid(True, alpha=0.3)
        if i == 0 and j == 0:
            ax.legend()
        # Print the numbers for each configuration
        print(f"\nt-SNE embedding (perplexity={perplexity}, learning_rate={lr}):")
        print(df_tsne.head())
        print(f"KL Divergence: {kl_divergence:.4f}")

plt.suptitle("t-SNE Grid: Perplexity vs Learning Rate (KL Divergence Shown)", y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# Default t-SNE for comparison
tsne_default = TSNE(n_components=2, random_state=42)
X_default = data_no_outliers_norm[FEATURES]
y_default = data_no_outliers_norm['label']
X_tsne_default = tsne_default.fit_transform(X_default)
df_tsne_default = pd.DataFrame(X_tsne_default, columns=['Dim1', 'Dim2'])
df_tsne_default['label'] = y_default.values
plt.figure(figsize=(8, 6))
for label in sorted(df_tsne_default['label'].unique()):
    subset = df_tsne_default[df_tsne_default['label'] == label]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=f'Label {label}', alpha=0.7)
kl_div_default = tsne_default.kl_divergence_
plt.title(f't-SNE (default parameters)\nKL={kl_div_default:.2f}')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
print("\nt-SNE embedding (default parameters):")
print(df_tsne_default.head())
print(f"KL Divergence: {kl_div_default:.4f}")