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

def fill_missing_values(df, features=None):
    df_filled = df.copy()
    
    if features is None:
        features = [col for col in df_filled.select_dtypes(include=[np.number]).columns if col != 'label']
    
    for feature in features:
        if feature in df_filled.columns:
            df_filled[feature] = df_filled.groupby("label")[feature].transform(
                lambda x: x.fillna(x.median())
            )
    
    return df_filled

def normalize_data(data, method='minmax', features=FEATURES):
    data_normalized = data.copy()
    for feature in features:
        if feature in data.columns:
            if method == 'minmax':
                x_min = data[feature].min()
                x_max = data[feature].max()
                if x_max != x_min:
                    data_normalized[feature] = (data[feature] - x_min) / (x_max - x_min)
            else: 
                x_mean = data[feature].mean()
                x_std = data[feature].std()
                if x_std != 0:
                    data_normalized[feature] = (data[feature] - x_mean) / x_std
    return data_normalized

def remove_outliers(data):
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


def main():
    data_load = load_data()
    data_filled = fill_missing_values(data_load)
    data_normal = normalize_data(data_filled)

    # evaluate on selected FEATURES
    results_selected = evaluate_clusters_libs(data_normal, FEATURES, k_min=2, k_max=10, random_state=42)
    print("Optimal k (selected features):", results_selected.get("optimal_k"))
    print("Elbow table (selected features):\n", results_selected["elbow"].head())

    # prepare all numeric features (exclude 'label')
    numeric_columns = data_load.select_dtypes(include=[np.number]).columns.tolist()
    if 'label' in numeric_columns:
        numeric_columns.remove('label')
    all_features = numeric_columns

    if len(all_features) == 0:
        raise RuntimeError("No numeric features found for clustering with all_features.")

    data_filled_all = fill_missing_values(data_load, features=all_features)
    data_normal_all = normalize_data(data_filled_all, features=all_features)

    results_all = evaluate_clusters_libs(data_normal_all, all_features, k_min=2, k_max=10, random_state=42)
    print("Optimal k (all features):", results_all.get("optimal_k"))
    print("Elbow table (all features):\n", results_all["elbow"].head())

    # clusterize both datasets with their respective optimal k (fallback 3)
    opt_k_sel = int(results_selected.get("optimal_k")) if results_selected.get("optimal_k") is not None else 3
    opt_k_all = int(results_all.get("optimal_k")) if results_all.get("optimal_k") is not None else 3

    X_sel = data_normal[FEATURES].values
    X_all = data_normal_all[all_features].values

    kmeans_sel = KMeans(n_clusters=opt_k_sel, random_state=42, n_init=10).fit(X_sel)
    kmeans_all = KMeans(n_clusters=opt_k_all, random_state=42, n_init=10).fit(X_all)

    data_normal['cluster_sel'] = kmeans_sel.labels_
    data_normal_all['cluster_all'] = kmeans_all.labels_

    sizes_sel = dict(enumerate(np.bincount(kmeans_sel.labels_)))
    sizes_all = dict(enumerate(np.bincount(kmeans_all.labels_)))

    sil_sel = silhouette_score(X_sel, kmeans_sel.labels_) if opt_k_sel > 1 else float('nan')
    sil_all = silhouette_score(X_all, kmeans_all.labels_) if opt_k_all > 1 else float('nan')

    print(f"KMeans (selected FEATURES) k={opt_k_sel} sizes={sizes_sel} inertia={kmeans_sel.inertia_} silhouette={sil_sel}")
    print(f"KMeans (all features)      k={opt_k_all} sizes={sizes_all} inertia={kmeans_all.inertia_} silhouette={sil_all}")

    # save results
    data_normal.to_csv("data_normal_clustered_selected_features.csv", index=False)
    data_normal_all.to_csv("data_normal_clustered_all_features.csv", index=False)
    pd.DataFrame(kmeans_sel.cluster_centers_, columns=FEATURES).to_csv("kmeans_centers_selected_features.csv", index_label="cluster")
    pd.DataFrame(kmeans_all.cluster_centers_, columns=all_features).to_csv("kmeans_centers_all_features.csv", index_label="cluster")

    # compute 2D embeddings for visualization
    tsne_sel = TSNE(n_components=2, random_state=42, perplexity=50)
    X_tsne_sel = tsne_sel.fit_transform(X_sel)
    df_tsne_sel = pd.DataFrame(X_tsne_sel, columns=['Dim1','Dim2'])
    df_tsne_sel['cluster'] = kmeans_sel.labels_
    df_tsne_sel['label'] = data_normal['label'].values

    tsne_all = TSNE(n_components=2, random_state=42, perplexity=50)
    X_tsne_all = tsne_all.fit_transform(X_all)
    df_tsne_all = pd.DataFrame(X_tsne_all, columns=['Dim1','Dim2'])
    df_tsne_all['cluster'] = kmeans_all.labels_
    df_tsne_all['label'] = data_normal_all['label'].values

    # detect outliers only for selected FEATURES (do NOT run remove_outliers for all_features)
    cleaned_sel, outliers_sel, outlier_mask_sel, inner_mask_sel, outer_mask_sel = remove_outliers(data_normal)

    # Re-evaluate optimal k on cleaned selected dataset
    results_clean_selected = evaluate_clusters_libs(cleaned_sel, FEATURES, k_min=2, k_max=10, random_state=42, excel_path="cluster_selection_cleaned_selected.xlsx")
    print("Optimal k (selected features) on cleaned data:", results_clean_selected.get("optimal_k"))
    print("Elbow table (selected features, cleaned):\n", results_clean_selected["elbow"].head())

    # prepare cleaned arrays for assignment/re-fit
    X_clean = cleaned_sel[FEATURES].values

    # assign cleaned points to original clusters (predict) and compute silhouette
    labels_clean_pred = kmeans_sel.predict(X_clean)
    silhouette_predict = silhouette_score(X_clean, labels_clean_pred) if len(np.unique(labels_clean_pred)) > 1 else float('nan')

    # recompute clustering on cleaned data (fit) and compute silhouette
    kmeans_clean_recomputed = KMeans(n_clusters=opt_k_sel, random_state=42, n_init=10).fit(X_clean)
    labels_clean_recomputed = kmeans_clean_recomputed.labels_
    sil_clean_recomputed = silhouette_score(X_clean, labels_clean_recomputed) if opt_k_sel > 1 else float('nan')

    # t-SNE for cleaned data (for both predict and recomputed visualizations we'll use same embedding)
    tsne_clean = TSNE(n_components=2, random_state=42, perplexity=50)
    X_tsne_clean = tsne_clean.fit_transform(X_clean)
    df_tsne_clean = pd.DataFrame(X_tsne_clean, columns=['Dim1','Dim2'])
    df_tsne_clean['cluster_pred'] = labels_clean_pred
    df_tsne_clean['cluster_recomputed'] = labels_clean_recomputed
    df_tsne_clean['label'] = cleaned_sel['label'].values

    # First figure: selected FEATURES with outliers highlighted and all-features plot (no outliers overlay)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.get_cmap("tab10")

    # Selected FEATURES: normal points by cluster, inner outliers= squares, outer outliers= triangles
    ax = axes[0]
    for c in sorted(df_tsne_sel['cluster'].unique()):
        mask_normal = (~outlier_mask_sel) & (df_tsne_sel['cluster'] == c)
        pts = df_tsne_sel[mask_normal]
        ax.scatter(pts['Dim1'], pts['Dim2'], s=30, color=colors(c % 10), label=f"cluster {c}", alpha=0.7)
    for c in sorted(df_tsne_sel['cluster'].unique()):
        mask_inner = inner_mask_sel & (df_tsne_sel['cluster'] == c)
        pts = df_tsne_sel[mask_inner]
        if not pts.empty:
            ax.scatter(pts['Dim1'], pts['Dim2'], s=90, marker='s', facecolors=colors(c % 10), edgecolors='k', linewidths=0.8, label=f"cluster {c} inner outlier")
    for c in sorted(df_tsne_sel['cluster'].unique()):
        mask_outer = outer_mask_sel & (df_tsne_sel['cluster'] == c)
        pts = df_tsne_sel[mask_outer]
        if not pts.empty:
            ax.scatter(pts['Dim1'], pts['Dim2'], s=90, marker='^', facecolors=colors(c % 10), edgecolors='k', linewidths=0.8, label=f"cluster {c} outer outlier")

    ax.set_title(f"t-SNE (selected FEATURES), k={opt_k_sel}\nSilhouette (predict)={silhouette_predict:.4f}")
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
    handles, labels_ = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small')

    # All features plot (no outlier overlay)
    ax = axes[1]
    for c in sorted(df_tsne_all['cluster'].unique()):
        sub = df_tsne_all[df_tsne_all['cluster'] == c]
        ax.scatter(sub['Dim1'], sub['Dim2'], s=30, color=colors(c % 10), label=f"cluster {c}", alpha=0.7)
    ax.set_title(f"t-SNE (all FEATURES), k={opt_k_all}")
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
    ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

    # Second figure: cleaned data - left = assignment-to-original-clusters (predict), right = recomputed (fit)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.set_title(f"Assigned to original KMeans (predict) k={opt_k_sel}\nSilhouette={silhouette_predict:.4f}")
    for c in sorted(df_tsne_clean['cluster_pred'].unique()):
        sub = df_tsne_clean[df_tsne_clean['cluster_pred'] == c]
        ax.scatter(sub['Dim1'], sub['Dim2'], s=30, color=colors(c % 10), label=f"cluster {c}", alpha=0.7)
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
    ax.legend(loc='best', fontsize='small')

    ax = axes[1]
    ax.set_title(f"Recomputed on cleaned data (fit) k={opt_k_sel}\nSilhouette={sil_clean_recomputed:.4f}")
    for c in sorted(df_tsne_clean['cluster_recomputed'].unique()):
        sub = df_tsne_clean[df_tsne_clean['cluster_recomputed'] == c]
        ax.scatter(sub['Dim1'], sub['Dim2'], s=30, color=colors(c % 10), label=f"cluster {c}", alpha=0.7)
    ax.set_xlabel("Dim1"); ax.set_ylabel("Dim2")
    ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.show()

    print(f"Recomputed KMeans on cleaned data: k={opt_k_sel}, inertia={kmeans_clean_recomputed.inertia_:.4f}, silhouette={sil_clean_recomputed:.4f}")
    pd.DataFrame(kmeans_clean_recomputed.cluster_centers_, columns=FEATURES).to_csv("kmeans_centers_cleaned_recomputed.csv", index_label="cluster")


if __name__ == "__main__":
    main()