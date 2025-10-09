import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]

def load_data():
    dfs = []
    for label in LABELS:
        df = pd.read_csv(f"sampled_label_{label}.csv")
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def minmax_normalize(data):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data[FEATURES])
    norm_df = pd.DataFrame(normalized, columns=FEATURES)
    norm_df['label'] = data['label'].values
    return norm_df

def perform_pca(data, normalized=False):
    X = data[FEATURES]
    y = data['label']

    # PCA iki 2 dimensijų
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['label'] = y

    plt.figure(figsize=(8, 6))
    for label in sorted(df_pca['label'].unique()):
        subset = df_pca[df_pca['label'] == label]
        plt.scatter(subset['PC1'], subset['PC2'], label=f'Label {label}', alpha=0.7)

    plt.title(f'PCA Visualization ({"Normalized" if normalized else "Raw"} Data)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    explained = pca.explained_variance_ratio_
    print(f"Explained variance ratio: PC1={explained[0]:.3f}, PC2={explained[1]:.3f}")

def main():
    data = load_data()

    print("=== PCA on Raw Data ===")
    perform_pca(data, normalized=False)

    print("\n=== PCA on Min-Max Normalized Data ===")
    normalized_data = minmax_normalize(data)
    perform_pca(normalized_data, normalized=True)

if __name__ == "__main__":
    main()