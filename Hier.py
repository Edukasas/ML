import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster

# Constants
FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]
LINKAGE_METHOD = 'ward'
DISTANCE_METRIC = 'euclidean'


def load_data():
    dfs = [pd.read_csv(f"sampled_label_{label}.csv").assign(label=label)
           for label in LABELS]
    return pd.concat(dfs, ignore_index=True)


def fill_missing_values(df, features=None):
    df_filled = df.copy()

    if features is None:
        features = [col for col in df_filled.select_dtypes(include=[np.number]).columns
                    if col != 'label']

    for feature in features:
        if feature in df_filled.columns:
            df_filled[feature] = df_filled.groupby("label")[feature].transform(
                lambda x: x.fillna(x.median())
            )

    return df_filled


def normalize_data(data, method='minmax', features=FEATURES):
    data_normalized = data.copy()

    for feature in features:
        if feature not in data.columns:
            continue

        if method == 'minmax':
            x_min, x_max = data[feature].min(), data[feature].max()
            if x_max != x_min:
                data_normalized[feature] = (data[feature] - x_min) / (x_max - x_min)
        else:  # z-score
            x_mean, x_std = data[feature].mean(), data[feature].std()
            if x_std != 0:
                data_normalized[feature] = (data[feature] - x_mean) / x_std

    return data_normalized


def remove_outliers(data):
    data_features = data[FEATURES]
    Q1, Q3 = data_features.quantile(0.25), data_features.quantile(0.75)
    IQR = Q3 - Q1

    lower_inner, upper_inner = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    inner_outlier_mask = ((data_features < lower_inner) |
                          (data_features > upper_inner)).any(axis=1)

    lower_outer, upper_outer = Q1 - 3 * IQR, Q3 + 3 * IQR
    outer_outlier_mask = ((data_features < lower_outer) |
                          (data_features > upper_outer)).any(axis=1)

    outlier_mask = inner_outlier_mask | outer_outlier_mask
    cleaned_data = data[~outlier_mask].copy()
    outliers_data = data[outlier_mask].copy()

    return cleaned_data, outliers_data, outlier_mask, inner_outlier_mask, outer_outlier_mask


def perform_tsne(data, normalized=False, perplexity=30, metric='euclidean', learning_rate='auto'):
    X = data[FEATURES]
    y = data['label']

    if normalized:
        X = normalize_data(X)

    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric,
                learning_rate=learning_rate, random_state=42)
    X_embedded = tsne.fit_transform(X)

    return X_embedded, y


def evaluate_hierarchical_clustering(df, features, linkage_method='ward',
                                     distance_metric='euclidean', k_min=2,
                                     k_max=10, excel_path="hierarchical_analysis.xlsx"):
    X = df[features].values
    Z = linkage(X, method=linkage_method, metric=distance_metric)

    results = []
    for k in range(k_min, k_max + 1):
        clusters = fcluster(Z, k)
        sil = silhouette_score(X, clusters) if k > 1 else np.nan
        results.append({'k': k, 'silhouette': sil})

    results_df = pd.DataFrame(results)
    results_df.to_excel(excel_path, index=False)

    last_merges = Z[-(k_max - 1):, 2]
    gaps = np.diff(last_merges)
    optimal_k = k_min + np.argmax(gaps)

    return Z, results_df, optimal_k


def main():
    data = load_data()
    data = fill_missing_values(data)
    data = normalize_data(data)
    data = remove_outliers(data)

    _, results_df, optimal_k = evaluate_hierarchical_clustering(
        data, FEATURES, LINKAGE_METHOD, DISTANCE_METRIC
    )

    X_embedded = perform_tsne(data)
    results_df.to_excel("hierarchical_analysis.xlsx", index=False)

    print("Analysis complete.")
    print(f"Optimal k: {optimal_k}")
    print("Results saved to hierarchical_analysis.xlsx")



if __name__ == "__main__":
    main()
