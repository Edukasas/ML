import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

def perform_pca(data, normalized=False):
    X = data[FEATURES]
    y = data['label']

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['label'] = y.values

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

    data = fill_missing_values(data)

    data = remove_outliers(data)

    perform_pca(data, normalized=False)

    normalized_data = normalize_data(data)
    perform_pca(normalized_data, normalized=True)

if __name__ == "__main__":
    main()