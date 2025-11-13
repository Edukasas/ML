import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import warnings
warnings.filterwarnings('ignore')

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABEL = ["label"]
LABELS = [0, 1, 2]

LINKAGE_METHODS = ['ward', 'complete', 'average']
DISTANCE_METRICS = {
    'ward': ['euclidean'],
    'complete': ['euclidean', 'cityblock'],
    'average': ['euclidean', 'cityblock']
}

METRIC_DISPLAY_NAMES = {
    'euclidean': 'Euclidean',
    'cityblock': 'Manhattan'
}

IQR_MULTIPLIER = 3.0
INNER_MULTIPLIER = 1.5


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


def clean_excel_errors(df):
    df_cleaned = df.copy()
    
    excel_errors = ['#NAME?', '#VALUE!', '#REF!', '#DIV/0!', '#N/A', '#NUM!', '#NULL!']
    
    for col in df_cleaned.columns:
        if col != 'label':
            df_cleaned[col] = df_cleaned[col].replace(excel_errors, np.nan)
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    return df_cleaned


def normalize_features(df, features=None):
    df_normalized = df.copy()
    
    if features is None:
        features = [col for col in df_normalized.select_dtypes(include=[np.number]).columns if col != 'label']
    
    for feature in features:
        if feature in df_normalized.columns:
            x_min = df[feature].min()
            x_max = df[feature].max()
            if x_max != x_min:
                df_normalized[feature] = (df[feature] - x_min) / (x_max - x_min)
    
    return df_normalized


def remove_outliers(data):
    data_features = data[FEATURES]
    Q1 = data_features.quantile(0.25)
    Q3 = data_features.quantile(0.75)
    IQR = Q3 - Q1

    lower_inner = Q1 - 1.5 * IQR
    upper_inner = Q3 + 1.5 * IQR
    inner_outlier_mask = ((data_features < lower_inner) | (data_features > upper_inner)).any(axis=1)

    lower_outer = Q1 - 3 * IQR
    upper_outer = Q3 + 3 * IQR
    outer_outlier_mask = ((data_features < lower_outer) | (data_features > upper_outer)).any(axis=1)

    outlier_mask = inner_outlier_mask | outer_outlier_mask

    cleaned_data = data[~outlier_mask].copy()
    outliers_data = data[outlier_mask].copy()

    return cleaned_data, outliers_data, outlier_mask, inner_outlier_mask, outer_outlier_mask


def detect_outliers_iqr(X, multiplier=3.0, inner_multiplier=1.5):
    outer_outlier_mask = np.zeros(len(X), dtype=bool)
    inner_outlier_mask = np.zeros(len(X), dtype=bool)
    
    for feature in X.columns:
        q1 = X[feature].quantile(0.25)
        q3 = X[feature].quantile(0.75)
        iqr = q3 - q1

        outer_outlier_mask |= (X[feature] < q1 - multiplier * iqr) | (X[feature] > q3 + multiplier * iqr)
        is_beyond_inner = (X[feature] < q1 - inner_multiplier * iqr) | (X[feature] > q3 + inner_multiplier * iqr)
        is_within_outer = (X[feature] >= q1 - multiplier * iqr) & (X[feature] <= q3 + multiplier * iqr)
        inner_outlier_mask |= (is_beyond_inner & is_within_outer)

    return outer_outlier_mask, inner_outlier_mask


def evaluate_clustering(X, clusters):
    n_clusters = len(np.unique(clusters))
    
    metrics = {
        'silhouette': silhouette_score(X, clusters) if n_clusters > 1 else np.nan,
        'davies_bouldin': davies_bouldin_score(X, clusters) if n_clusters > 1 else np.nan,
        'calinski_harabasz': calinski_harabasz_score(X, clusters) if n_clusters > 1 else np.nan
    }
    
    return metrics


def find_optimal_k_multi_metric(X, Z, k_min=2, k_max=10):
    results = []
    for k in range(k_min, k_max + 1):
        clusters = fcluster(Z, k, criterion='maxclust')
        metrics = evaluate_clustering(X, clusters)
        
        results.append({
            'k': k,
            'n_clusters': len(np.unique(clusters)),
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    
    results_df['silhouette_norm'] = (results_df['silhouette'] + 1) / 2
    
    db_min, db_max = results_df['davies_bouldin'].min(), results_df['davies_bouldin'].max()
    if db_max > db_min:
        results_df['davies_bouldin_norm'] = 1 - (results_df['davies_bouldin'] - db_min) / (db_max - db_min)
    else:
        results_df['davies_bouldin_norm'] = 0.5
    
    ch_min, ch_max = results_df['calinski_harabasz'].min(), results_df['calinski_harabasz'].max()
    if ch_max > ch_min:
        results_df['calinski_harabasz_norm'] = (results_df['calinski_harabasz'] - ch_min) / (ch_max - ch_min)
    else:
        results_df['calinski_harabasz_norm'] = 0.5
    
    results_df['combined_score'] = (
        0.33 * results_df['silhouette_norm'] +
        0.33 * results_df['davies_bouldin_norm'] +
        0.34 * results_df['calinski_harabasz_norm']
    )
    
    optimal_idx = results_df['combined_score'].idxmax()
    
    recommendations = {
        'optimal_k': int(results_df.loc[optimal_idx, 'k']),
        'best_silhouette_k': int(results_df.loc[results_df['silhouette'].idxmax(), 'k']),
        'best_davies_bouldin_k': int(results_df.loc[results_df['davies_bouldin'].idxmin(), 'k']),
        'best_calinski_harabasz_k': int(results_df.loc[results_df['calinski_harabasz'].idxmax(), 'k']),
        'evaluation_df': results_df
    }
    
    return recommendations


def analyze_cluster_label_relationship(data_with_labels, clusters):
    df_analysis = pd.DataFrame({
        'cluster': clusters,
        'label': data_with_labels['label'].values
    })
    
    contingency = pd.crosstab(df_analysis['cluster'], df_analysis['label'])
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    
    dominant_labels = contingency_pct.idxmax(axis=1)
    dominant_percentages = contingency_pct.max(axis=1)
    
    cluster_summary = pd.DataFrame({
        'cluster_size': contingency.sum(axis=1),
        'dominant_label': dominant_labels,
        'dominant_percentage': dominant_percentages,
        'label_diversity': contingency.apply(lambda x: (x > 0).sum(), axis=1)
    })
    overall_purity = (cluster_summary['cluster_size'] * cluster_summary['dominant_percentage']).sum() / cluster_summary['cluster_size'].sum()
    
    return {
        'contingency_table': contingency,
        'contingency_percentages': contingency_pct,
        'cluster_summary': cluster_summary,
        'overall_purity': overall_purity,
    }


def plot_optimization_metrics(recommendations, linkage_method, distance_metric, save_path=None):
    results_df = recommendations['evaluation_df']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    fig.suptitle(f'{linkage_method.capitalize()} / {display_metric}', fontsize=14, fontweight='bold')

    axes[0, 0].plot(results_df['k'], results_df['silhouette'], 'o-')
    axes[0, 0].set_title('Silhouette')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(results_df['k'], results_df['davies_bouldin'], 'o-', color='orange')
    axes[0, 1].set_title('Davies-Bouldin')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(results_df['k'], results_df['calinski_harabasz'], 'o-', color='green')
    axes[1, 0].set_title('Calinski-Harabasz')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(results_df['k'], results_df['combined_score'], 'o-', color='red')
    axes[1, 1].axvline(recommendations['optimal_k'], color='darkred', linestyle='--')
    axes[1, 1].set_title(f'Combined (k={recommendations["optimal_k"]})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_label_distribution(label_analysis, linkage_method, distance_metric, optimal_k, save_path=None):
    contingency_pct = label_analysis['contingency_percentages']
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    contingency_pct.plot(kind='bar', stacked=True, ax=ax, colormap='viridis', edgecolor='black')
    ax.set_title(f'{linkage_method.capitalize()} / {display_metric} (k={optimal_k})', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Percentage (%)')
    ax.legend(title='Label', bbox_to_anchor=(1.05, 1))
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dendrogram(Z, linkage_method, distance_metric, optimal_k=None, save_path=None):
    plt.figure(figsize=(15, 8))
    
    threshold = None
    if optimal_k is not None and optimal_k > 1:
        threshold = (Z[-optimal_k, 2] + Z[-optimal_k+1, 2]) / 2
    
    dendrogram(Z, truncate_mode=None, color_threshold=threshold, above_threshold_color='gray')
    
    if threshold:
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'k={optimal_k}')
        plt.legend()
    
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    plt.title(f'{linkage_method.capitalize()} / {display_metric}', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def characterize_clusters(df, clusters):
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    return df_with_clusters['cluster'].value_counts().sort_index()


def plot_cluster_characteristics(cluster_sizes, linkage_method, distance_metric, optimal_k, save_path=None):
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cluster_sizes.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{linkage_method.capitalize()} / {display_metric} (k={optimal_k})', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne_clusters(df, clusters, linkage_method, distance_metric, optimal_k, save_path=None):
    X = df.values
    df_plot = df.copy()
    df_plot['cluster'] = clusters

    outer_outlier_mask, inner_outlier_mask = detect_outliers_iqr(df)
    df_plot['is_outer'] = outer_outlier_mask
    df_plot['is_inner'] = inner_outlier_mask

    tsne = TSNE(n_components=2, perplexity=50, random_state=42, metric=distance_metric)
    X_embedded = tsne.fit_transform(X)
    df_plot['tsne_1'] = X_embedded[:, 0]
    df_plot['tsne_2'] = X_embedded[:, 1]

    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    plt.figure(figsize=(12, 8))

    cmap = plt.cm.get_cmap('tab10', len(np.unique(clusters)))

    for cluster_id in sorted(df_plot['cluster'].unique()):
        cluster_data = df_plot[df_plot['cluster'] == cluster_id]
        base_color = cmap(cluster_id - 1)

        normal = cluster_data[~cluster_data['is_outer'] & ~cluster_data['is_inner']]
        plt.scatter(normal['tsne_1'], normal['tsne_2'], color=[base_color], alpha=0.6, s=50, 
                   edgecolors='black', linewidth=0.3, label=f"Cluster {cluster_id}")

        inner = cluster_data[cluster_data['is_inner'] & ~cluster_data['is_outer']]
        if not inner.empty:
            plt.scatter(inner['tsne_1'], inner['tsne_2'], color=[base_color], marker='+', s=80, linewidth=1.2)

        outer = cluster_data[cluster_data['is_outer']]
        if not outer.empty:
            plt.scatter(outer['tsne_1'], outer['tsne_2'], color=[base_color], marker='^', s=60, 
                       edgecolor='black', linewidth=0.8)

    plt.title(f'{linkage_method.capitalize()} / {display_metric} (k={optimal_k})', fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def perform_hierarchical_clustering(df, linkage_method='ward', distance_metric='euclidean', k_min=2, k_max=10):
    X = df.values
    Z = linkage(X, method=linkage_method, metric=distance_metric)
    recommendations = find_optimal_k_multi_metric(X, Z, k_min, k_max)
    return Z, recommendations['evaluation_df'], recommendations['optimal_k'], recommendations


def main():
    data = load_data()
    data = clean_excel_errors(data)
    data = fill_missing_values(data)
    
    # data, outliers_data, outlier_mask, inner_outlier_mask, outer_outlier_mask = remove_outliers(data)

    data_normalized = normalize_features(data)
    data_normalized = data_normalized.replace([np.inf, -np.inf], np.nan)
    data_normalized = data_normalized.dropna(axis=0, how="any")

    labels_for_analysis = data_normalized['label'].copy()
    X = data_normalized[FEATURES]

    # X = data_normalized.drop(columns=LABEL)

    all_results = {}
    all_optimal_k = {}
    all_recommendations = {}

    for linkage_method in LINKAGE_METHODS:
        for distance_metric in DISTANCE_METRICS[linkage_method]:
            print(f"\n{'='*80}")
            print(f"Testing: {linkage_method.upper()} linkage + {distance_metric.upper()} distance")
            print('='*80)
            
            Z, results_df, optimal_k, recommendations = perform_hierarchical_clustering(
                X, linkage_method, distance_metric
            )
            
            key = (linkage_method, distance_metric)
            all_results[key] = results_df
            all_optimal_k[key] = optimal_k
            all_recommendations[key] = recommendations
            
            opt_metrics_path = f"optimization_metrics_{linkage_method}_{distance_metric}.png"
            plot_optimization_metrics(recommendations, linkage_method, distance_metric, opt_metrics_path)
            
            clusters = fcluster(Z, optimal_k, criterion='maxclust')
            metrics = evaluate_clustering(X, clusters)
            print(f"  - Silhouette Score: {metrics['silhouette']:.4f}")
            print(f"  - Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
            print(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
            
            data_for_label_analysis = pd.DataFrame({'label': labels_for_analysis})
            label_analysis = analyze_cluster_label_relationship(data_for_label_analysis, clusters)

            for idx, row in label_analysis['cluster_summary'].iterrows():
                print(f"   Cluster {idx}: {int(row['cluster_size'])} samples, "
                      f"dominant label={int(row['dominant_label'])} ({row['dominant_percentage']:.1f}%)")

            dendrogram_path = f"dendrogram_{linkage_method}_{distance_metric}.png"
            plot_dendrogram(Z, linkage_method, distance_metric, optimal_k, dendrogram_path)
            
            cluster_sizes = characterize_clusters(X, clusters)
            char_path = f"cluster_characteristics_{linkage_method}_{distance_metric}.png"
            plot_cluster_characteristics(cluster_sizes, linkage_method, distance_metric, optimal_k, char_path)
            
            label_dist_path = f"label_distribution_{linkage_method}_{distance_metric}.png"
            plot_label_distribution(label_analysis, linkage_method, distance_metric, optimal_k, label_dist_path)
            
            tsne_path = f"tsne_clusters_{linkage_method}_{distance_metric}.png"
            plot_tsne_clusters(X, clusters, linkage_method, distance_metric, optimal_k, tsne_path)
            
            df_clusters = X.copy()
            df_clusters['Cluster'] = clusters
            df_clusters['Original_Label'] = labels_for_analysis.values
            
            descriptive_stats = df_clusters.groupby('Cluster').agg(['mean', 'std', 'min', 'max', 'count'])
            stats_path = f"cluster_statistics_{linkage_method}_{distance_metric}.xlsx"
            
            with pd.ExcelWriter(stats_path, engine='openpyxl') as writer:
                descriptive_stats.to_excel(writer, sheet_name='Feature_Statistics')
                label_analysis['contingency_table'].to_excel(writer, sheet_name='Label_Counts')
                label_analysis['contingency_percentages'].to_excel(writer, sheet_name='Label_Percentages')
                label_analysis['cluster_summary'].to_excel(writer, sheet_name='Cluster_Summary')
                recommendations['evaluation_df'].to_excel(writer, sheet_name='K_Optimization', index=False)
            
if __name__ == "__main__":
    main()