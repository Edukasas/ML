import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from scipy.stats import f_oneway, chi2_contingency
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

        outer_lower = q1 - multiplier * iqr
        outer_upper = q3 + multiplier * iqr
        inner_lower = q1 - inner_multiplier * iqr
        inner_upper = q3 + inner_multiplier * iqr

        outer_outlier_mask |= (X[feature] < outer_lower) | (X[feature] > outer_upper)
        is_beyond_inner = (X[feature] < inner_lower) | (X[feature] > inner_upper)
        is_within_outer = (X[feature] >= outer_lower) & (X[feature] <= outer_upper)
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
    """
    Find optimal k by evaluating multiple metrics and combining their recommendations.
    Returns optimal k and detailed evaluation results.
    """
    results = []
    
    # Evaluate clustering metrics for each k
    for k in range(k_min, k_max + 1):
        clusters = fcluster(Z, k, criterion='maxclust')
        metrics = evaluate_clustering(X, clusters)
        
        results.append({
            'k': k,
            'n_clusters': len(np.unique(clusters)),
            **metrics
        })
    
    results_df = pd.DataFrame(results)
    
    # Normalize metrics to [0, 1] for comparison
    # Silhouette: higher is better (already in [-1, 1], normalize to [0, 1])
    results_df['silhouette_norm'] = (results_df['silhouette'] + 1) / 2
    
    # Davies-Bouldin: lower is better (invert and normalize)
    db_min, db_max = results_df['davies_bouldin'].min(), results_df['davies_bouldin'].max()
    if db_max > db_min:
        results_df['davies_bouldin_norm'] = 1 - (results_df['davies_bouldin'] - db_min) / (db_max - db_min)
    else:
        results_df['davies_bouldin_norm'] = 0.5
    
    # Calinski-Harabasz: higher is better (normalize)
    ch_min, ch_max = results_df['calinski_harabasz'].min(), results_df['calinski_harabasz'].max()
    if ch_max > ch_min:
        results_df['calinski_harabasz_norm'] = (results_df['calinski_harabasz'] - ch_min) / (ch_max - ch_min)
    else:
        results_df['calinski_harabasz_norm'] = 0.5
    
    # Elbow method on dendrogram heights
    last_merges = Z[-(k_max-1):, 2]
    heights = last_merges[::-1]
    k_values = np.arange(k_max, k_min-1, -1)
    heights = heights[:len(k_values)]
    k_values = k_values[:len(heights)]
    
    # Calculate elbow score (perpendicular distance to line)
    if len(heights) > 2:
        heights_norm = (heights - heights.min()) / (heights.max() - heights.min() + 1e-10)
        k_norm = (k_values - k_values.min()) / (k_values.max() - k_values.min() + 1e-10)
        
        distances = []
        for i in range(len(k_values)):
            point = np.array([k_norm[i], heights_norm[i]])
            line_start = np.array([k_norm[0], heights_norm[0]])
            line_end = np.array([k_norm[-1], heights_norm[-1]])
            
            line_vec = line_end - line_start
            point_vec = point - line_start
            
            line_len = np.linalg.norm(line_vec)
            if line_len > 0:
                distance = abs(np.cross(line_vec, point_vec)) / line_len
            else:
                distance = 0
            distances.append(distance)
        
        elbow_scores = np.array(distances)
        elbow_scores_norm = elbow_scores / (elbow_scores.max() + 1e-10)
        
        # Add elbow scores to results
        for i, k in enumerate(k_values):
            idx = results_df[results_df['k'] == k].index
            if len(idx) > 0:
                results_df.loc[idx[0], 'elbow_score'] = elbow_scores_norm[i]
    else:
        results_df['elbow_score'] = 0.0
    
    # Combined score (weighted average)
    # Weights: silhouette (0.3), davies_bouldin (0.2), calinski_harabasz (0.3), elbow (0.2)
    results_df['combined_score'] = (
        0.3 * results_df['silhouette_norm'] +
        0.2 * results_df['davies_bouldin_norm'] +
        0.3 * results_df['calinski_harabasz_norm'] +
        0.2 * results_df['elbow_score']
    )
    
    # Find k with best combined score
    optimal_idx = results_df['combined_score'].idxmax()
    optimal_k = results_df.loc[optimal_idx, 'k']
    
    # Individual metric recommendations
    best_silhouette_k = results_df.loc[results_df['silhouette'].idxmax(), 'k']
    best_db_k = results_df.loc[results_df['davies_bouldin'].idxmin(), 'k']
    best_ch_k = results_df.loc[results_df['calinski_harabasz'].idxmax(), 'k']
    best_elbow_k = results_df.loc[results_df['elbow_score'].idxmax(), 'k'] if 'elbow_score' in results_df else None
    
    recommendations = {
        'optimal_k': int(optimal_k),
        'best_silhouette_k': int(best_silhouette_k),
        'best_davies_bouldin_k': int(best_db_k),
        'best_calinski_harabasz_k': int(best_ch_k),
        'best_elbow_k': int(best_elbow_k) if best_elbow_k else None,
        'evaluation_df': results_df
    }
    
    return recommendations


def analyze_cluster_label_relationship(data_with_labels, clusters):
    """
    Analyze relationship between clusters and original labels.
    Returns detailed analysis of label distribution in clusters.
    """
    df_analysis = pd.DataFrame({
        'cluster': clusters,
        'label': data_with_labels['label'].values
    })
    
    # Contingency table
    contingency = pd.crosstab(df_analysis['cluster'], df_analysis['label'])
    
    # Calculate percentages
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
    
    # Dominant label per cluster
    dominant_labels = contingency_pct.idxmax(axis=1)
    dominant_percentages = contingency_pct.max(axis=1)
    
    # Purity metrics
    cluster_summary = pd.DataFrame({
        'cluster_size': contingency.sum(axis=1),
        'dominant_label': dominant_labels,
        'dominant_percentage': dominant_percentages,
        'label_diversity': contingency.apply(lambda x: (x > 0).sum(), axis=1)
    })
    
    # Chi-square test for independence
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Overall purity (weighted average of dominant percentages)
    overall_purity = (cluster_summary['cluster_size'] * cluster_summary['dominant_percentage']).sum() / cluster_summary['cluster_size'].sum()
    
    analysis_results = {
        'contingency_table': contingency,
        'contingency_percentages': contingency_pct,
        'cluster_summary': cluster_summary,
        'overall_purity': overall_purity,
        'chi2_statistic': chi2,
        'chi2_p_value': p_value
    }
    
    return analysis_results


def analyze_feature_differences(X, clusters):
    """
    Analyze feature value differences between clusters.
    Returns statistical tests and meaningful interpretations.
    """
    df_features = X.copy()
    df_features['cluster'] = clusters
    
    # Statistical tests (ANOVA) for each feature
    feature_analysis = []
    
    for feature in X.columns:
        cluster_groups = [df_features[df_features['cluster'] == c][feature].values 
                         for c in sorted(df_features['cluster'].unique())]
        
        # ANOVA F-test
        f_stat, p_value = f_oneway(*cluster_groups)
        
        # Effect size (eta-squared)
        grand_mean = df_features[feature].mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean)**2 for group in cluster_groups)
        ss_total = sum((df_features[feature] - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Cluster means and std
        cluster_stats = df_features.groupby('cluster')[feature].agg(['mean', 'std', 'min', 'max'])
        
        feature_analysis.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant': p_value < 0.05,
            'cluster_means': cluster_stats['mean'].to_dict(),
            'cluster_stds': cluster_stats['std'].to_dict()
        })
    
    feature_df = pd.DataFrame(feature_analysis)
    
    # Identify most discriminative features
    feature_df = feature_df.sort_values('eta_squared', ascending=False)
    
    return feature_df


def generate_interpretation_report(linkage_method, distance_metric, optimal_k, 
                                  label_analysis, feature_analysis, save_path=None):
    """
    Generate comprehensive interpretation report based on expert analysis.
    """
    report = []
    report.append("=" * 80)
    report.append(f"CLUSTER INTERPRETATION REPORT")
    report.append(f"Method: {linkage_method.capitalize()} / {METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)}")
    report.append(f"Optimal k: {optimal_k}")
    report.append("=" * 80)
    report.append("")
    
    # 1. Cluster-Label Relationship Analysis
    report.append("1. RELATIONSHIP WITH ORIGINAL LABELS")
    report.append("-" * 80)
    report.append(f"Overall Cluster Purity: {label_analysis['overall_purity']:.2f}%")
    report.append(f"Chi-square test: χ² = {label_analysis['chi2_statistic']:.2f}, p = {label_analysis['chi2_p_value']:.4f}")
    if label_analysis['chi2_p_value'] < 0.05:
        report.append("   → Clusters are SIGNIFICANTLY associated with original labels")
    else:
        report.append("   → Clusters are NOT significantly associated with original labels")
    report.append("")
    
    report.append("Label Distribution in Clusters:")
    for cluster_id in label_analysis['cluster_summary'].index:
        row = label_analysis['cluster_summary'].loc[cluster_id]
        report.append(f"\n  Cluster {cluster_id}:")
        report.append(f"    - Size: {int(row['cluster_size'])} samples")
        report.append(f"    - Dominant Label: {int(row['dominant_label'])} ({row['dominant_percentage']:.1f}%)")
        report.append(f"    - Label Diversity: {int(row['label_diversity'])} different labels")
        
        # Show label breakdown
        label_dist = label_analysis['contingency_percentages'].loc[cluster_id]
        report.append(f"    - Label breakdown: {', '.join([f'Label {int(l)}: {v:.1f}%' for l, v in label_dist.items() if v > 0])}")
    
    report.append("")
    report.append("")
    
    # 2. Feature-based Cluster Differentiation
    report.append("2. FEATURE-BASED CLUSTER DIFFERENTIATION")
    report.append("-" * 80)
    
    significant_features = feature_analysis[feature_analysis['significant']]
    report.append(f"Number of statistically significant features: {len(significant_features)}/{len(feature_analysis)}")
    report.append("")
    
    report.append("Most Discriminative Features (by effect size):")
    for idx, row in feature_analysis.head(len(feature_analysis)).iterrows():
        report.append(f"\n  {row['feature']}:")
        report.append(f"    - Effect size (η²): {row['eta_squared']:.4f} {'(LARGE)' if row['eta_squared'] > 0.14 else '(MEDIUM)' if row['eta_squared'] > 0.06 else '(SMALL)'}")
        report.append(f"    - Statistical significance: p = {row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'}")
        report.append(f"    - Cluster means: {', '.join([f'C{k}: {v:.3f}' for k, v in sorted(row['cluster_means'].items())])}")
    
    report.append("")
    report.append("")
    
    # 3. Interpretation and Conclusions
    report.append("3. EXPERT INTERPRETATION")
    report.append("-" * 80)
    
    # Check if clusters are meaningful
    high_purity = label_analysis['overall_purity'] > 70
    significant_association = label_analysis['chi2_p_value'] < 0.05
    discriminative_features = len(significant_features) >= len(feature_analysis) * 0.5
    
    report.append("Quality Assessment:")
    report.append(f"  ✓ High cluster purity (>70%): {'YES' if high_purity else 'NO'}")
    report.append(f"  ✓ Significant label association: {'YES' if significant_association else 'NO'}")
    report.append(f"  ✓ Strong feature discrimination: {'YES' if discriminative_features else 'NO'}")
    report.append("")
    
    if high_purity and significant_association:
        report.append("CONCLUSION: Clusters show STRONG correspondence with original labels.")
        report.append("The clustering successfully recovers the underlying class structure.")
    elif high_purity or significant_association:
        report.append("CONCLUSION: Clusters show MODERATE correspondence with original labels.")
        report.append("Some class structure is recovered, but with limitations.")
    else:
        report.append("CONCLUSION: Clusters show WEAK correspondence with original labels.")
        report.append("The clustering reveals alternative patterns not aligned with original classes.")
    
    report.append("")
    
    # Identify key differentiating features
    top_features = feature_analysis.head(3)
    report.append("Key Differentiating Features:")
    for idx, row in top_features.iterrows():
        report.append(f"  - {row['feature']}: Explains {row['eta_squared']*100:.1f}% of variance between clusters")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
    
    return report_text


def plot_optimization_metrics(recommendations, linkage_method, distance_metric, save_path=None):
    """
    Visualize the multi-metric optimization process.
    """
    results_df = recommendations['evaluation_df']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Optimal k Selection - {linkage_method.capitalize()} / {METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)}',
                 fontsize=16, fontweight='bold')
    
    # Silhouette Score
    ax = axes[0, 0]
    ax.plot(results_df['k'], results_df['silhouette'], 'o-', linewidth=2, markersize=8)
    ax.axvline(recommendations['best_silhouette_k'], color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score (higher is better)')
    ax.grid(True, alpha=0.3)
    
    # Davies-Bouldin Index
    ax = axes[0, 1]
    ax.plot(results_df['k'], results_df['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
    ax.axvline(recommendations['best_davies_bouldin_k'], color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Davies-Bouldin Index')
    ax.set_title('Davies-Bouldin Index (lower is better)')
    ax.grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    ax = axes[0, 2]
    ax.plot(results_df['k'], results_df['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='green')
    ax.axvline(recommendations['best_calinski_harabasz_k'], color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Calinski-Harabasz Score')
    ax.set_title('Calinski-Harabasz Score (higher is better)')
    ax.grid(True, alpha=0.3)
    
    # Elbow Score
    ax = axes[1, 0]
    ax.plot(results_df['k'], results_df['elbow_score'], 'o-', linewidth=2, markersize=8, color='purple')
    if recommendations['best_elbow_k']:
        ax.axvline(recommendations['best_elbow_k'], color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Elbow Score')
    ax.set_title('Elbow Method Score (higher is better)')
    ax.grid(True, alpha=0.3)
    
    # Combined Score
    ax = axes[1, 1]
    ax.plot(results_df['k'], results_df['combined_score'], 'o-', linewidth=2, markersize=8, color='red')
    ax.axvline(recommendations['optimal_k'], color='darkred', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Combined Score')
    ax.set_title(f'Combined Score (Optimal k={recommendations["optimal_k"]})')
    ax.grid(True, alpha=0.3)
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary_data = [
        ['Metric', 'Best k'],
        ['Silhouette', f"{recommendations['best_silhouette_k']}"],
        ['Davies-Bouldin', f"{recommendations['best_davies_bouldin_k']}"],
        ['Calinski-Harabasz', f"{recommendations['best_calinski_harabasz_k']}"],
        ['Elbow', f"{recommendations['best_elbow_k']}"],
        ['', ''],
        ['COMBINED', f"{recommendations['optimal_k']}"]
    ]
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Highlight combined result
    for i in range(2):
        table[(6, i)].set_facecolor('#ffcccc')
        table[(6, i)].set_text_props(weight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_label_distribution(label_analysis, linkage_method, distance_metric, 
                           optimal_k, save_path=None):
    """
    Visualize label distribution across clusters.
    """
    contingency_pct = label_analysis['contingency_percentages']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    
    # Stacked bar chart
    ax = axes[0]
    contingency_pct.plot(kind='bar', stacked=True, ax=ax, 
                         colormap='viridis', alpha=0.8, edgecolor='black')
    ax.set_title('Label Distribution in Clusters (%)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Cluster', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', alpha=0.3)
    
    # Heatmap
    ax = axes[1]
    sns.heatmap(contingency_pct, annot=True, fmt='.1f', cmap='YlOrRd', 
               ax=ax, cbar_kws={'label': 'Percentage (%)'}, 
               linewidths=1, linecolor='gray')
    ax.set_title('Label Distribution Heatmap (%)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Label', fontsize=11)
    ax.set_ylabel('Cluster', fontsize=11)
    
    plt.suptitle(f'Cluster-Label Relationship - {linkage_method.capitalize()} / {display_metric} (k={optimal_k})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_analysis, linkage_method, distance_metric,
                           optimal_k, save_path=None):
    """
    Visualize feature importance for cluster differentiation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric)
    
    # Effect sizes
    ax = axes[0]
    colors = ['red' if p < 0.05 else 'gray' for p in feature_analysis['p_value']]
    ax.barh(feature_analysis['feature'], feature_analysis['eta_squared'], color=colors, alpha=0.7)
    ax.set_xlabel('Effect Size (η²)', fontsize=11)
    ax.set_title('Feature Discriminative Power', fontweight='bold', fontsize=12)
    ax.axvline(0.06, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
    ax.axvline(0.14, color='red', linestyle='--', alpha=0.5, label='Large effect')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # P-values
    ax = axes[1]
    colors = ['green' if p < 0.05 else 'orange' for p in feature_analysis['p_value']]
    ax.barh(feature_analysis['feature'], -np.log10(feature_analysis['p_value']), color=colors, alpha=0.7)
    ax.set_xlabel('-log10(p-value)', fontsize=11)
    ax.set_title('Statistical Significance', fontweight='bold', fontsize=12)
    ax.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Feature Analysis - {linkage_method.capitalize()} / {display_metric} (k={optimal_k})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dendrogram(Z, linkage_method, distance_metric, optimal_k=None, save_path=None):
    plt.figure(figsize=(15, 8))
    
    if optimal_k is not None:
        threshold = (Z[-optimal_k, 2] + Z[-optimal_k+1, 2]) / 2 if optimal_k > 1 else None
    else:
        threshold = None
    
    dendrogram(Z, 
               truncate_mode=None,
               color_threshold=threshold,
               above_threshold_color='gray')
    
    if threshold:
        plt.axhline(y=threshold, color='r', linestyle='--', 
                   label=f'Cut for k={optimal_k}')
        plt.legend()
    
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric.capitalize())
    
    plt.title(f'Dendrogram - {linkage_method.capitalize()} Linkage, {display_metric} Distance', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def characterize_clusters(df, clusters):
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    cluster_sizes = df_with_clusters['cluster'].value_counts().sort_index()
    return cluster_sizes


def plot_cluster_characteristics(cluster_sizes, 
                                 linkage_method, distance_metric, optimal_k, save_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric.capitalize())
    
    cluster_sizes.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Cluster Sizes', fontweight='bold')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Samples')
    ax.tick_params(axis='x', rotation=0)
    
    plt.suptitle(f'Cluster Characterization - {linkage_method.capitalize()} / {display_metric} (k={optimal_k})',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_tsne_clusters(df, clusters, linkage_method, distance_metric, 
                      optimal_k, save_path=None):
    X = df.values
    df_plot = df.copy()
    df_plot['cluster'] = clusters

    outer_outlier_mask, inner_outlier_mask = detect_outliers_iqr(df)
    df_plot['is_outer_outlier'] = outer_outlier_mask
    df_plot['is_inner_outlier'] = inner_outlier_mask

    tsne = TSNE(n_components=2, perplexity=50, random_state=42, metric=distance_metric)
    X_embedded = tsne.fit_transform(X)
    df_plot['tsne_1'] = X_embedded[:, 0]
    df_plot['tsne_2'] = X_embedded[:, 1]

    display_metric = METRIC_DISPLAY_NAMES.get(distance_metric, distance_metric.capitalize())
    plt.figure(figsize=(12, 8))

    cmap = plt.cm.get_cmap('tab10', len(np.unique(clusters)))

    for cluster_id in sorted(df_plot['cluster'].unique()):
        cluster_data = df_plot[df_plot['cluster'] == cluster_id]
        base_color = cmap(cluster_id - 1)

        normal_points = cluster_data[~cluster_data['is_outer_outlier'] & ~cluster_data['is_inner_outlier']]
        plt.scatter(normal_points['tsne_1'], normal_points['tsne_2'],
                    color=[base_color], alpha=0.6, s=50, edgecolors='black', linewidth=0.3,
                    label=f"Cluster {cluster_id}")

        inner_outliers = cluster_data[cluster_data['is_inner_outlier'] & ~cluster_data['is_outer_outlier']]
        if not inner_outliers.empty:
            plt.scatter(inner_outliers['tsne_1'], inner_outliers['tsne_2'],
                        color=[base_color], marker='+', s=80, linewidth=1.2)

        outer_outliers = cluster_data[cluster_data['is_outer_outlier']]
        if not outer_outliers.empty:
            plt.scatter(outer_outliers['tsne_1'], outer_outliers['tsne_2'],
                        color=[base_color], marker='^', s=60, edgecolor='black', linewidth=0.8)

    plt.title(f't-SNE Visualization with Cluster-Colored Outliers\n'
              f'{linkage_method.capitalize()} / {display_metric} (k={optimal_k})',
              fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def perform_hierarchical_clustering(df, linkage_method='ward',
                                    distance_metric='euclidean', k_min=2, k_max=10):
    X = df.values
    Z = linkage(X, method=linkage_method, metric=distance_metric)
    
    # Use multi-metric optimization
    recommendations = find_optimal_k_multi_metric(X, Z, k_min, k_max)
    optimal_k = recommendations['optimal_k']
    results_df = recommendations['evaluation_df']
    
    return Z, results_df, optimal_k, recommendations


def main():
    data = load_data()
    data = clean_excel_errors(data)
    data = fill_missing_values(data)

    # data, outliers_sel, outlier_mask_sel, inner_mask_sel, outer_mask_sel = remove_outliers(data)
    
    data_normalized = normalize_features(data)

    # Clean data and remove infinities / NaNs
    data_normalized = data_normalized.replace([np.inf, -np.inf], np.nan)
    data_normalized = data_normalized.dropna(axis=0, how="any")

    # Keep track of labels for later analysis
    labels_for_analysis = data_normalized['label'].copy()

    # ✅ FIX: properly drop label column — LABEL is already a list
    X = data_normalized.drop(columns=LABEL)

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
            
            print(f"\nOptimal k determination:")
            print(f"  - Silhouette recommends: k={recommendations['best_silhouette_k']}")
            print(f"  - Davies-Bouldin recommends: k={recommendations['best_davies_bouldin_k']}")
            print(f"  - Calinski-Harabasz recommends: k={recommendations['best_calinski_harabasz_k']}")
            print(f"  - Elbow method recommends: k={recommendations['best_elbow_k']}")
            print(f"  → COMBINED OPTIMAL k: {optimal_k}")
            
            # Plot optimization metrics
            opt_metrics_path = f"optimization_metrics_{linkage_method}_{distance_metric}.png"
            plot_optimization_metrics(recommendations, linkage_method, distance_metric, opt_metrics_path)
            
            # Generate clusters
            clusters = fcluster(Z, optimal_k, criterion='maxclust')
            metrics = evaluate_clustering(X, clusters)
            print(f"\nClustering Quality Metrics:")
            print(f"  - Silhouette Score: {metrics['silhouette']:.4f}")
            print(f"  - Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
            print(f"  - Calinski-Harabasz Score: {metrics['calinski_harabasz']:.2f}")
            
            # EXPERT INTERPRETATION ANALYSIS
            print(f"\n{'='*80}")
            print("EXPERT INTERPRETATION ANALYSIS")
            print('='*80)
            
            # 1. Analyze cluster-label relationship
            data_for_label_analysis = pd.DataFrame({'label': labels_for_analysis})
            label_analysis = analyze_cluster_label_relationship(data_for_label_analysis, clusters)
            
            print(f"\n1. Cluster-Label Relationship:")
            print(f"   - Overall Purity: {label_analysis['overall_purity']:.2f}%")
            print(f"   - Chi-square: χ²={label_analysis['chi2_statistic']:.2f}, p={label_analysis['chi2_p_value']:.4f}")
            print(f"\n   Cluster Summary:")
            for idx, row in label_analysis['cluster_summary'].iterrows():
                print(f"   Cluster {idx}: {int(row['cluster_size'])} samples, "
                      f"dominant label={int(row['dominant_label'])} ({row['dominant_percentage']:.1f}%)")
            
            # 2. Analyze feature differences
            feature_analysis = analyze_feature_differences(X, clusters)
            
            print(f"\n2. Feature-Based Differentiation:")
            print(f"   - Significant features: {len(feature_analysis[feature_analysis['significant']])}/{len(feature_analysis)}")
            print(f"\n   Top discriminative features:")
            for idx, row in feature_analysis.head(3).iterrows():
                print(f"   - {row['feature']}: η²={row['eta_squared']:.4f}, p={row['p_value']:.4f}")
            
            # Generate interpretation report
            report_path = f"interpretation_report_{linkage_method}_{distance_metric}.txt"
            report_text = generate_interpretation_report(
                linkage_method, distance_metric, optimal_k,
                label_analysis, feature_analysis, report_path
            )
            print(f"\n   → Full interpretation report saved to: {report_path}")
            
            # Visualizations
            dendrogram_path = f"dendrogram_{linkage_method}_{distance_metric}.png"
            plot_dendrogram(Z, linkage_method, distance_metric, optimal_k, dendrogram_path)
            
            cluster_sizes = characterize_clusters(X, clusters)
            char_path = f"cluster_characteristics_{linkage_method}_{distance_metric}.png"
            plot_cluster_characteristics(cluster_sizes, linkage_method, distance_metric, optimal_k, char_path)
            
            # Label distribution visualization
            label_dist_path = f"label_distribution_{linkage_method}_{distance_metric}.png"
            plot_label_distribution(label_analysis, linkage_method, distance_metric, optimal_k, label_dist_path)
            
            # Feature importance visualization
            feature_imp_path = f"feature_importance_{linkage_method}_{distance_metric}.png"
            plot_feature_importance(feature_analysis, linkage_method, distance_metric, optimal_k, feature_imp_path)
            
            # t-SNE visualization
            tsne_path = f"tsne_clusters_{linkage_method}_{distance_metric}.png"
            plot_tsne_clusters(X, clusters, linkage_method, distance_metric, optimal_k, tsne_path)
            
            # Save detailed statistics
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
                feature_analysis.to_excel(writer, sheet_name='Feature_Analysis', index=False)
                recommendations['evaluation_df'].to_excel(writer, sheet_name='K_Optimization', index=False)
            
            print(f"\n   → Statistics saved to: {stats_path}")
            print(f"\n{'='*80}\n")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files for each method combination:")
    print("  - optimization_metrics_*.png: Multi-metric k optimization")
    print("  - dendrogram_*.png: Hierarchical clustering dendrogram")
    print("  - cluster_characteristics_*.png: Cluster size distribution")
    print("  - label_distribution_*.png: Label distribution in clusters")
    print("  - feature_importance_*.png: Feature discriminative power")
    print("  - tsne_clusters_*.png: t-SNE visualization")
    print("  - interpretation_report_*.txt: Expert interpretation report")
    print("  - cluster_statistics_*.xlsx: Comprehensive statistics")


if __name__ == "__main__":
    main()