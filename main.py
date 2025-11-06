import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]
K_RANGE = range(2, 30)


# Configuration: Set to True to remove outliers before normalization
REMOVE_OUTLIERS_BEFORE_NORMALIZATION = False


# ============================================================================
# DATA PREPARATION
# ============================================================================


def load_data():
    """Load and combine CSV files for all labels."""
    dfs = [pd.read_csv(f"sampled_label_{label}.csv").assign(label=label) 
           for label in LABELS]
    return pd.concat(dfs, ignore_index=True)


def prepare_data(df, features, remove_outliers_first=False):
    """Fill missing values and normalize features. Optionally remove outliers first."""
    df = df.copy()
    
    # Fill missing values with group median
    for feature in features:
        if feature in df.columns:
            df[feature] = df.groupby("label")[feature].transform(
                lambda x: x.fillna(x.median())
            )
    
    # Remove outliers before normalization if requested
    outlier_info = None
    if remove_outliers_first:
        df, outlier_mask, inner_mask, outer_mask = remove_outliers(df, features)
        outlier_info = {
            'outlier_mask': outlier_mask,
            'inner_mask': inner_mask,
            'outer_mask': outer_mask,
            'n_outliers': outlier_mask.sum(),
            'n_mild': (inner_mask & ~outer_mask).sum(),
            'n_extreme': outer_mask.sum()  
        }
        print(f"  Removed {outlier_info['n_outliers']} outliers "
              f"({outlier_info['n_mild']} inner, {outlier_info['n_extreme']} outer)")
    
    # Normalize to [0, 1]
    for feature in features:
        if feature in df.columns:
            x_min, x_max = df[feature].min(), df[feature].max()
            if x_max != x_min:
                df[feature] = (df[feature] - x_min) / (x_max - x_min)
    
    return (df, outlier_info) if remove_outliers_first else df


def remove_outliers(df, features):
    """Remove outliers using IQR method. Returns cleaned data and masks."""
    Q1 = df[features].quantile(0.25)
    Q3 = df[features].quantile(0.75)
    IQR = Q3 - Q1
    
    inner_mask = ((df[features] < Q1 - 1.5 * IQR) | 
                  (df[features] > Q3 + 1.5 * IQR)).any(axis=1)
    outer_mask = ((df[features] < Q1 - 3 * IQR) | 
                  (df[features] > Q3 + 3 * IQR)).any(axis=1)
    
    outlier_mask = inner_mask | outer_mask
    cleaned = df[~outlier_mask].copy()
    
    return cleaned, outlier_mask, inner_mask, outer_mask


# ============================================================================
# CLUSTER ANALYSIS
# ============================================================================


def compute_empirical_k(m):
    """Compute optimal k using empirical formula: k ≈ sqrt(m/2)."""
    return int(np.sqrt(m / 2))


def compute_clustering_metrics(df, features, k_range=K_RANGE):
    """Compute inertia and silhouette scores for range of k values."""
    X = df[features].values
    results = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_) if k > 1 else np.nan
        results.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
    
    return pd.DataFrame(results)


def find_elbow_k(metrics_df):
    """Find elbow point using perpendicular distance to line method."""
    xs = metrics_df['k'].values.astype(float)
    ys = metrics_df['inertia'].values.astype(float)
    
    # Line from first to last point
    x1, y1 = xs[0], ys[0]
    x2, y2 = xs[-1], ys[-1]
    
    # Distance from each point to line
    a, b = y2 - y1, -(x2 - x1)
    c = x2 * y1 - y2 * x1
    denom = np.sqrt(a**2 + b**2)
    
    if denom == 0:
        return int(xs[0])
    
    distances = [abs(a * x + b * y + c) / denom for x, y in zip(xs, ys)]
    return int(xs[np.argmax(distances)])


def choose_optimal_k(metrics_df, elbow_k, empirical_k, sil_k):
    """Choose k based on elbow, silhouette, and empirical method agreement."""
    
    # If all three agree
    if elbow_k == sil_k == empirical_k:
        return elbow_k, 'all methods agree'
    
    # If two methods agree
    if elbow_k == empirical_k:
        return elbow_k, 'elbow and empirical agree'
    if elbow_k == sil_k:
        return elbow_k, 'elbow and silhouette agree'
    if sil_k == empirical_k:
        return sil_k, 'silhouette and empirical agree'
    
    # Check silhouette scores for elbow vs empirical
    sil_at_elbow = metrics_df.loc[metrics_df['k'] == elbow_k, 'silhouette'].values
    sil_at_empirical = metrics_df.loc[metrics_df['k'] == empirical_k, 'silhouette'].values
    sil_best = metrics_df.loc[metrics_df['k'] == sil_k, 'silhouette'].values[0]
    
    # Prioritize empirical if it has good silhouette
    if len(sil_at_empirical) > 0 and not np.isnan(sil_at_empirical[0]):
        if sil_best - sil_at_empirical[0] < 0.02:
            return empirical_k, 'empirical chosen (reasonable silhouette)'
    
    # Fall back to elbow
    if len(sil_at_elbow) > 0 and not np.isnan(sil_at_elbow[0]):
        if not np.isnan(sil_best) and (sil_best - sil_at_elbow[0]) > 0.02:
            return sil_k, 'silhouette significantly better'
    
    return elbow_k, 'elbow chosen (default)'


def fit_kmeans(df, features, k):
    """Fit KMeans and return model and metrics."""
    X = df[features].values
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sil = silhouette_score(X, km.labels_)
    counts = np.bincount(km.labels_)
    sizes = {int(i): int(v) for i, v in enumerate(counts)}
    
    return km, {'silhouette': sil, 'inertia': km.inertia_, 'sizes': sizes}


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_label_distribution_in_clusters(df, kmeans, title="Label Distribution in Clusters"):
    """Plot bar chart showing how original labels are distributed across clusters."""
    # Create a DataFrame with cluster and label info
    cluster_label_counts = pd.crosstab(kmeans.labels_, df['label'])
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Stacked bar chart
    cluster_label_counts.plot(kind='bar', stacked=True, ax=axes[0], 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_xlabel('Cluster')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_title(f'{title} - Stacked')
    axes[0].legend(title='Original Label', labels=[f'Label {i}' for i in LABELS])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Plot 2: Grouped bar chart
    cluster_label_counts.plot(kind='bar', ax=axes[1], 
                             color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_xlabel('Cluster')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title(f'{title} - Grouped')
    axes[1].legend(title='Original Label', labels=[f'Label {i}' for i in LABELS])
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", bbox_inches='tight', dpi=100)
    plt.show()
    
    # Print detailed statistics
    print(f"\n{title}:")
    print("="*70)
    print(cluster_label_counts)
    print("\nPercentage distribution:")
    print((cluster_label_counts.div(cluster_label_counts.sum(axis=1), axis=0) * 100).round(2))


def plot_metrics(metrics_df, title, elbow_k=None, empirical_k=None, prefix=""):
    """Plot elbow and silhouette curves with empirical k marked."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Elbow plot
    axes[0].plot(metrics_df['k'], metrics_df['inertia'], '-o')
    if elbow_k:
        y = metrics_df.loc[metrics_df['k'] == elbow_k, 'inertia'].values[0]
        axes[0].scatter([elbow_k], [y], color='red', s=100, zorder=5, label=f'Elbow k={elbow_k}')
        axes[0].annotate(f"Elbow\nk={elbow_k}", (elbow_k, y), 
                        xytext=(10, -10), textcoords="offset points", ha='left')
    if empirical_k:
        y = metrics_df.loc[metrics_df['k'] == empirical_k, 'inertia'].values[0]
        axes[0].scatter([empirical_k], [y], color='green', s=100, marker='^', zorder=5, label=f'Empirical k={empirical_k}')
        axes[0].annotate(f"Empirical\nk={empirical_k}", (empirical_k, y), 
                        xytext=(10, 10), textcoords="offset points", ha='left')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title(f"{title} - Elbow Method")
    axes[0].legend()
    
    # Silhouette plot
    axes[1].plot(metrics_df['k'], metrics_df['silhouette'], '-o')
    if elbow_k:
        y = metrics_df.loc[metrics_df['k'] == elbow_k, 'silhouette'].values[0]
        axes[1].scatter([elbow_k], [y], color='red', s=100, zorder=5, label=f'Elbow k={elbow_k}')
    if empirical_k:
        y = metrics_df.loc[metrics_df['k'] == empirical_k, 'silhouette'].values[0]
        axes[1].scatter([empirical_k], [y], color='green', s=100, marker='^', zorder=5, label=f'Empirical k={empirical_k}')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f"{title} - Silhouette Score")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{prefix}_metrics.png", bbox_inches='tight', dpi=100)
    plt.show()


def compute_tsne(df, features):
    """Compute t-SNE embedding."""
    X = df[features].values
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    embedding = tsne.fit_transform(X)
    return pd.DataFrame(embedding, columns=['Dim1', 'Dim2'], index=df.index)


def plot_with_outliers(ax, df_tsne, kmeans_labels, outlier_mask, inner_mask, outer_mask, 
                      title, k, silhouette):
    """Helper function to plot clusters with inner/outer outlier markers."""
    colors = plt.cm.tab10
    
    # Align masks with df_tsne index
    outlier_mask_aligned = pd.Series(outlier_mask, index=outlier_mask.index).reindex(df_tsne.index).fillna(False)
    inner_mask_aligned = pd.Series(inner_mask, index=inner_mask.index).reindex(df_tsne.index).fillna(False)
    outer_mask_aligned = pd.Series(outer_mask, index=outer_mask.index).reindex(df_tsne.index).fillna(False)
    
    for c in sorted(set(kmeans_labels)):
        label_mask = pd.Series(kmeans_labels == c, index=df_tsne.index)
        
        # Normal points (not outliers)
        mask = (~outlier_mask_aligned) & label_mask
        pts = df_tsne.loc[mask]
        ax.scatter(pts['Dim1'], pts['Dim2'], s=30, color=colors(c), 
                  label=f"Cluster {c}", alpha=0.7)
        
        # Inner outliers (square markers)
        mask_inner = inner_mask_aligned & label_mask
        pts = df_tsne.loc[mask_inner]
        if not pts.empty:
            ax.scatter(pts['Dim1'], pts['Dim2'], s=90, marker='s', 
                      facecolors=colors(c), edgecolors='k', linewidths=0.8)
        
        # Outer outliers (triangle markers)
        mask_outer = outer_mask_aligned & label_mask
        pts = df_tsne.loc[mask_outer]
        if not pts.empty:
            ax.scatter(pts['Dim1'], pts['Dim2'], s=90, marker='^', 
                      facecolors=colors(c), edgecolors='k', linewidths=0.8)
    
    ax.set_title(f"{title} (k={k})\nSilhouette={silhouette:.4f}")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(loc='best', fontsize='small')


def plot_clusters_comparison(df_sel, df_all, kmeans_sel, kmeans_all, 
                            outlier_mask, inner_mask, outer_mask, 
                            opt_k_sel, opt_k_all, sil_sel, sil_all):
    """Plot selected features (with outliers) vs all features."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Selected features with outliers
    plot_with_outliers(axes[0], df_sel, kmeans_sel.labels_, outlier_mask, inner_mask, outer_mask,
                      "Selected Features", opt_k_sel, sil_sel)
    
    # All features with outliers
    plot_with_outliers(axes[1], df_all, kmeans_all.labels_, outlier_mask, inner_mask, outer_mask,
                      "All Features", opt_k_all, sil_all)
    
    plt.tight_layout()
    plt.savefig("comparison_clusters.png", bbox_inches='tight', dpi=100)
    plt.show()


def plot_cleaned_comparison(df_tsne, labels_pred, labels_recomp, 
                           original_data, features,
                           k, sil_pred, sil_recomp):
    """Plot cleaned data: predicted vs recomputed clusters with outlier markers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Detect outliers in original data for visualization
    _, outlier_mask, inner_mask, outer_mask = remove_outliers(original_data, features)
    
    # Filter masks to cleaned data indices
    cleaned_indices = df_tsne.index
    outlier_mask_cleaned = outlier_mask.reindex(cleaned_indices).fillna(False)
    inner_mask_cleaned = inner_mask.reindex(cleaned_indices).fillna(False)
    outer_mask_cleaned = outer_mask.reindex(cleaned_indices).fillna(False)
    
    # Predicted clusters
    plot_with_outliers(axes[0], df_tsne, labels_pred, outlier_mask_cleaned, 
                      inner_mask_cleaned, outer_mask_cleaned,
                      "Assigned to Original", k, sil_pred)
    
    # Recomputed clusters
    plot_with_outliers(axes[1], df_tsne, labels_recomp, outlier_mask_cleaned,
                      inner_mask_cleaned, outer_mask_cleaned,
                      "Recomputed on Cleaned", k, sil_recomp)
    
    plt.tight_layout()
    plt.savefig("cleaned_comparison.png", bbox_inches='tight', dpi=100)
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    # Load and prepare data
    print("Loading data...")
    data = load_data()
    
    # Get all numeric features
    all_features = [col for col in data.select_dtypes(include=[np.number]).columns 
                   if col != 'label']
    
    # ========================================================================
    # ANALYSIS 1: Selected FEATURES
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYZING SELECTED FEATURES")
    print("="*70)
    
    if REMOVE_OUTLIERS_BEFORE_NORMALIZATION:
        print("\nRemoving outliers before normalization...")
        data_sel, outlier_info_sel = prepare_data(data, FEATURES, remove_outliers_first=True)
    else:
        data_sel = prepare_data(data, FEATURES)
        outlier_info_sel = None
    
    # Compute empirical k
    m_sel = len(data_sel)
    empirical_k_sel = compute_empirical_k(m_sel)
    print(f"\nEmpirical k (sqrt(m/2)): {empirical_k_sel} (m={m_sel})")
    
    metrics_sel = compute_clustering_metrics(data_sel, FEATURES)
    
    elbow_k_sel = find_elbow_k(metrics_sel)
    sil_k_sel = int(metrics_sel.loc[metrics_sel['silhouette'].idxmax(), 'k'])
    opt_k_sel, reason_sel = choose_optimal_k(metrics_sel, elbow_k_sel, empirical_k_sel, sil_k_sel)
    
    print(f"Elbow k: {elbow_k_sel}")
    print(f"Silhouette best k: {sil_k_sel}")
    print(f"Optimal k: {opt_k_sel} ({reason_sel})")
    
    plot_metrics(metrics_sel, "Selected Features", elbow_k_sel, empirical_k_sel, "selected")
    
    kmeans_sel, metrics_sel_final = fit_kmeans(data_sel, FEATURES, opt_k_sel)
    print(f"Silhouette: {metrics_sel_final['silhouette']:.4f}")
    print(f"Cluster sizes: {metrics_sel_final['sizes']}")
    
    # ========================================================================
    # ANALYSIS 2: All FEATURES
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYZING ALL FEATURES")
    print("="*70)
    
    if REMOVE_OUTLIERS_BEFORE_NORMALIZATION:
        print("\nRemoving outliers before normalization...")
        data_all, outlier_info_all = prepare_data(data, all_features, remove_outliers_first=True)
    else:
        data_all = prepare_data(data, all_features)
        outlier_info_all = None
    
    # Compute empirical k
    m_all = len(data_all)
    empirical_k_all = compute_empirical_k(m_all)
    print(f"\nEmpirical k (sqrt(m/2)): {empirical_k_all} (m={m_all})")
    
    metrics_all = compute_clustering_metrics(data_all, all_features)
    
    elbow_k_all = find_elbow_k(metrics_all)
    sil_k_all = int(metrics_all.loc[metrics_all['silhouette'].idxmax(), 'k'])
    opt_k_all, reason_all = choose_optimal_k(metrics_all, elbow_k_all, empirical_k_all, sil_k_all)
    
    print(f"Elbow k: {elbow_k_all}")
    print(f"Silhouette best k: {sil_k_all}")
    print(f"Optimal k: {opt_k_all} ({reason_all})")
    
    plot_metrics(metrics_all, "All Features", elbow_k_all, empirical_k_all, "all")
    
    kmeans_all, metrics_all_final = fit_kmeans(data_all, all_features, opt_k_all)
    print(f"Silhouette: {metrics_all_final['silhouette']:.4f}")
    print(f"Cluster sizes: {metrics_all_final['sizes']}")
    
    # ========================================================================
    # VISUALIZATION: Comparison Plot
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Compute t-SNE embeddings
    tsne_sel = compute_tsne(data_sel, FEATURES)
    tsne_sel['cluster'] = kmeans_sel.labels_
    
    tsne_all = compute_tsne(data_all, all_features)
    tsne_all['cluster'] = kmeans_all.labels_
    
    # Detect outliers (for visualization - either from early removal or detect now)
    if REMOVE_OUTLIERS_BEFORE_NORMALIZATION and outlier_info_sel:
        outlier_mask = outlier_info_sel['outlier_mask']
        inner_mask = outlier_info_sel['inner_mask']
        outer_mask = outlier_info_sel['outer_mask']
    else:
        _, outlier_mask, inner_mask, outer_mask = remove_outliers(data, FEATURES)
    
    plot_clusters_comparison(tsne_sel, tsne_all, kmeans_sel, kmeans_all,
                            outlier_mask, inner_mask, outer_mask,
                            opt_k_sel, opt_k_all,
                            metrics_sel_final['silhouette'],
                            metrics_all_final['silhouette'])
    
    # ========================================================================
    # ANALYSIS 3: Cleaned Data (Selected Features Only)
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYZING CLEANED DATA (OUTLIERS REMOVED)")
    print("="*70)
    
    data_cleaned, outlier_info = prepare_data(data, FEATURES, remove_outliers_first=True)
    data_filled = prepare_data(data, FEATURES)
    # Compute empirical k for cleaned data
    m_clean = len(data_cleaned)
    empirical_k_clean = compute_empirical_k(m_clean)
    print(f"\nEmpirical k (sqrt(m/2)): {empirical_k_clean} (m={m_clean})")
    
    metrics_clean = compute_clustering_metrics(data_cleaned, FEATURES)
    elbow_k_clean = find_elbow_k(metrics_clean)
    sil_k_clean = int(metrics_clean.loc[metrics_clean['silhouette'].idxmax(), 'k'])
    opt_k_clean, reason_clean = choose_optimal_k(metrics_clean, elbow_k_clean, empirical_k_clean, sil_k_clean)
    
    print(f"Elbow k: {elbow_k_clean}")
    print(f"Silhouette best k: {sil_k_clean}")
    print(f"Optimal k: {opt_k_clean} ({reason_clean})")
    
    plot_metrics(metrics_clean, "Cleaned Data", elbow_k_clean, empirical_k_clean, "cleaned")
    
    # Compare: assign to original vs recompute
    X_clean = data_cleaned[FEATURES].values
    
    labels_pred = kmeans_sel.predict(X_clean)
    sil_pred = silhouette_score(X_clean, labels_pred)
    
    kmeans_recomp, metrics_recomp = fit_kmeans(data_cleaned, FEATURES, opt_k_clean)
    
    print(f"\nAssigned to original: Silhouette={sil_pred:.4f}")
    print(f"Recomputed: Silhouette={metrics_recomp['silhouette']:.4f}")
    print(f"Recomputed cluster sizes: {metrics_recomp['sizes']}")
    
    tsne_clean = compute_tsne(data_cleaned, FEATURES)
    plot_cleaned_comparison(tsne_clean, labels_pred, kmeans_recomp.labels_,
                           data_filled, FEATURES,
                           opt_k_clean, sil_pred, metrics_recomp['silhouette'])
    

    # Label distribution and statistics
    print("\nLabel distribution in clusters (original data):")
    plot_label_distribution_in_clusters(data_sel, kmeans_sel, "Original Data - Label Distribution")
    
    print("\nLabel distribution in cleaned data:")
    plot_label_distribution_in_clusters(data_cleaned, kmeans_recomp, "Cleaned Data - Label Distribution")
    
    # Save results
    data_sel['cluster'] = kmeans_sel.labels_
    data_sel.to_csv("clustered_selected_features.csv", index=False)
    data_cleaned['cluster'] = kmeans_recomp.labels_
    data_cleaned.to_csv("cleaned_selected_features.csv", index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
