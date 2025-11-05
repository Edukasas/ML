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


def evaluate_hierarchical_clustering(data, features, linkage_method='ward',
                                     distance_metric='euclidean', k_min=2,
                                     k_max=10, excel_path="hierarchical_analysis.xlsx"):
    X = data[features].values

    if linkage_method == 'ward' and distance_metric != 'euclidean':
        print("Warning: Ward linkage requires euclidean distance. Using 'euclidean'")
        distance_metric = 'euclidean'

    print(f"Computing linkage matrix (method={linkage_method}, metric={distance_metric})...")
    Z = linkage(X, method=linkage_method, metric=distance_metric)

    # Evaluate different k values
    results = []
    for k in range(k_min, k_max + 1):
        clusters = fcluster(Z, k, criterion='maxclust')
        sil_score = silhouette_score(X, clusters) if k > 1 else np.nan

        results.append({
            'k': k,
            'silhouette': sil_score
        })

    results_df = pd.DataFrame(results)

    # Find optimal k using gap in merge distances
    last_merges = Z[-(k_max - 1):, 2]  # Last k_max-1 merges
    gaps = np.diff(last_merges)
    optimal_k_dendro = k_min + np.argmax(gaps)

    # Alternative: second derivative method
    if len(last_merges) >= 3:
        second_diff = np.diff(last_merges, 2)
        optimal_k_dendro_alt = k_min + np.argmax(second_diff)
    else:
        optimal_k_dendro_alt = optimal_k_dendro

    results_df.to_excel(excel_path, index=False)

    optimal_k_silhouette = results_df.loc[results_df['silhouette'].idxmax(), 'k']

    print(f"\n{'=' * 60}")
    print("HIERARCHICAL CLUSTERING ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Linkage method: {linkage_method}")
    print(f"Distance metric: {distance_metric}")
    print(f"\nOptimal k (dendrogram - largest gap): {optimal_k_dendro}")
    print(f"Optimal k (dendrogram - acceleration): {optimal_k_dendro_alt}")
    print(f"Optimal k (silhouette score): {optimal_k_silhouette}")
    print(f"\nNote: For hierarchical clustering, dendrogram interpretation")
    print(f"      is typically more reliable than silhouette scores alone.")
    print(f"{'=' * 60}\n")

    return {
        'linkage_matrix': Z,
        'results': results_df,
        'optimal_k_dendro': optimal_k_dendro,
        'optimal_k_silhouette': optimal_k_silhouette
    }


def main():
    """Main analysis pipeline (no plotting)."""
    print("Loading and preprocessing data...")
    data_load = load_data()
    data_filled = fill_missing_values(data_load)
    data_normal = normalize_data(data_filled)

    # Step 1: Hierarchical clustering evaluation
    print("\n" + "=" * 60)
    print("STEP 1: HIERARCHICAL CLUSTERING EVALUATION")
    print("=" * 60)
    results = evaluate_hierarchical_clustering(data_normal, FEATURES,
                                               linkage_method=LINKAGE_METHOD,
                                               distance_metric=DISTANCE_METRIC,
                                               k_min=2, k_max=10)
    print("\nResults table:")
    print(results["results"])

    print("Performing t-SNE dimensionality reduction...")
    X_embedded, y = perform_tsne(data_normal, normalized=False, perplexity=30)



if __name__ == "__main__":
    main()
