import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances, pairwise_distances
import time
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)

FEATURE_COLUMNS = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
CLASS_LABELS = [0, 1, 2]
CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
OUTPUT_DIRECTORY = "mds_plots"
IQR_MULTIPLIER = 3.0

def load_data():
    dataframes = []
    for label in CLASS_LABELS:
        df = pd.read_csv(f"sampled_label_{label}.csv")
        df['label'] = label
        dataframes.append(df)
    
    return pd.concat(dataframes, ignore_index=True)


def fill_missing_values(df):
    df_filled = df.copy()
    
    for feature in FEATURE_COLUMNS:
        df_filled[feature] = df_filled.groupby("label")[feature].transform(
            lambda x: x.fillna(x.median())
        )
    
    return df_filled


def normalize_features(df):
    df_normalized = df.copy()
    
    for feature in FEATURE_COLUMNS:
        mean = df_normalized[feature].mean()
        std = df_normalized[feature].std()
        df_normalized[feature] = (df_normalized[feature] - mean) / std
    
    return df_normalized

def filter_six_features(df):
    available = [col for col in FEATURE_COLUMNS if col in df.columns]
    return df[available].copy()


def detect_outliers_iqr(X, multiplier=3.0):
    outlier_mask = np.zeros(len(X), dtype=bool)
    
    for feature in X.columns:
        q1 = X[feature].quantile(0.25)
        q3 = X[feature].quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask |= (X[feature] < lower_bound) | (X[feature] > upper_bound)
    
    return outlier_mask


def calculate_stress_metrics(distance_matrix, X_embedded):
    distances_low_dim = euclidean_distances(X_embedded)
    
    upper_triangular_indices = np.triu_indices_from(distance_matrix, k=1)
    
    high_dim_distances = distance_matrix[upper_triangular_indices]
    low_dim_distances = distances_low_dim[upper_triangular_indices]
    
    raw_stress = np.sum((high_dim_distances - low_dim_distances) ** 2)
    stress_1 = np.sqrt(raw_stress / np.sum(high_dim_distances ** 2))
    
    total_sum_squares = np.sum((high_dim_distances - np.mean(high_dim_distances)) ** 2)
    residual_sum_squares = np.sum((high_dim_distances - low_dim_distances) ** 2)
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    
    return stress_1, r_squared


def run_mds_default(X, y):
    start_time = time.time()
    
    mds = MDS(n_components=2, random_state=42)
    
    X_embedded = mds.fit_transform(X)
    
    distance_matrix = euclidean_distances(X)
    stress, r_squared = calculate_stress_metrics(distance_matrix, X_embedded)
    
    results = {
        'embedding': X_embedded,
        'labels': y,
        'stress': stress,
        'r2': r_squared,
        'n_iter': mds.n_iter_,
        'time': time.time() - start_time,
    }
    
    return results


def run_mds_with_metric(X, y, metric_name, max_iterations, num_initializations):
    start_time = time.time()
    
    # Compute precomputed dissimilarity matrix
    distance_matrix = pairwise_distances(X, metric=metric_name)
    
    mds = MDS(
        n_components=2,
        max_iter=max_iterations,
        n_init=num_initializations,
        random_state=42,
        dissimilarity='precomputed'
    )
    
    X_embedded = mds.fit_transform(distance_matrix)
    
    stress, r_squared = calculate_stress_metrics(distance_matrix, X_embedded)
    
    results = {
        'embedding': X_embedded,
        'labels': y,
        'stress': stress,
        'r2': r_squared,
        'n_iter': mds.n_iter_,
        'time': time.time() - start_time,
        'metric': metric_name,
        'max_iter': max_iterations,
        'n_init': num_initializations
    }
    
    return results


def plot_mds_results_no_outliers(results, labels):
    plot_df = pd.DataFrame(
        results['embedding'],
        columns=['Dimension_1', 'Dimension_2']
    )
    plot_df['label'] = labels.values
    
    plt.figure(figsize=(8, 6))
    
    for class_idx, class_label in enumerate(sorted(plot_df['label'].unique())):
        class_data = plot_df[plot_df['label'] == class_label]
        
        normal_points = class_data
        plt.scatter(
            normal_points.Dimension_1,
            normal_points.Dimension_2,
            color=CLASS_COLORS[class_idx],
            alpha=0.6,
            label=f"Label {class_label}",
            edgecolor='k',
            linewidth=0.3,
            s=40
        )
    
    title = (
        f"Stress={results['stress']:.4f} | "
        f"R²={results['r2']:.4f} | Time={results['time']:.1f}s"
    )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()


def plot_mds_results(results, labels, outlier_mask):
    plot_df = pd.DataFrame(
        results['embedding'],
        columns=['Dimension_1', 'Dimension_2']
    )
    plot_df['label'] = labels.values
    plot_df['is_outlier'] = outlier_mask
    
    plt.figure(figsize=(8, 6))
    
    for class_idx, class_label in enumerate(sorted(plot_df['label'].unique())):
        class_data = plot_df[plot_df['label'] == class_label]
        
        normal_points = class_data[~class_data['is_outlier']]
        plt.scatter(
            normal_points.Dimension_1,
            normal_points.Dimension_2,
            color=CLASS_COLORS[class_idx],
            alpha=0.6,
            label=f"Label {class_label}",
            edgecolor='k',
            linewidth=0.3,
            s=40
        )
        
        outlier_points = class_data[class_data['is_outlier']]
        if not outlier_points.empty:
            plt.scatter(
                outlier_points.Dimension_1,
                outlier_points.Dimension_2,
                color=CLASS_COLORS[class_idx],
                marker='^',
                edgecolor='k',
                s=70,
                label=f"Label {class_label} outlier"
            )
    
    metric_name = results['metric'].capitalize()
    title = (
        f"MDS with {metric_name} Distance (max_iter={results['max_iter']}, n_init={results['n_init']})\n"
        f"Stress={results['stress']:.4f} | "
        f"R²={results['r2']:.4f} | Time={results['time']:.1f}s"
    )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIRECTORY}/MDS_{results['metric']}_iter{results['max_iter']}_init{results['n_init']}.png"
    plt.savefig(filename, dpi=150)
    plt.show()


def main():
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    data = load_data()
    data = fill_missing_values(data)

    print("=" * 70)
    print("Running MDS without normalization")
    print("=" * 70)
    no_normalize = run_mds_default(data[FEATURE_COLUMNS], data['label'])
    plot_mds_results_no_outliers(no_normalize, data['label'])

    print("\n" + "=" * 70)
    print("Running MDS with normalization")
    print("=" * 70)
    data = normalize_features(data)
    normalize = run_mds_default(data[FEATURE_COLUMNS], data['label'])
    plot_mds_results_no_outliers(normalize, data['label'])

    data = filter_six_features(data);
    normalize = run_mds_default(data[FEATURE_COLUMNS], data['label'])
    plot_mds_results_no_outliers(normalize, data['label'])

    X = data[FEATURE_COLUMNS]
    y = data['label']
    
    outlier_mask = detect_outliers_iqr(X, multiplier=IQR_MULTIPLIER)
    
    # Different distance metrics to compare
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev']
    
    # Different configurations to test
    max_iter_values = [100, 300, 500]
    n_init_values = [4]
    
    print("\n" + "=" * 70)
    print("Comparing Distance Metrics and MDS Parameters (Normalized Data)")
    print("=" * 70)
    
    all_results = []
    
    for metric in distance_metrics:
        print(f"\n{'='*70}")
        print(f"Analyzing {metric.upper()} distance")
        print('='*70)
        
        for max_iter in max_iter_values:
            for n_init in n_init_values:
                result = run_mds_with_metric(X, y, metric, max_iter, n_init)
                
                print(
                    f"{metric:<12} | max_iter={max_iter:<3} | n_init={n_init:<2} | "
                    f"Stress={result['stress']:.4f} | "
                    f"R²={result['r2']:.4f} | "
                    f"Time={result['time']:.1f}s"
                )
                
                plot_mds_results(result, y, outlier_mask)
                all_results.append(result)
    
    print("\n" + "=" * 70)
    print("Summary: Best Configuration for Each Metric")
    print("=" * 70)
    
    for metric in distance_metrics:
        metric_results = [r for r in all_results if r['metric'] == metric]
        best_for_metric = min(metric_results, key=lambda r: r['stress'])
        print(
            f"{metric.capitalize():<12} | "
            f"max_iter={best_for_metric['max_iter']:<3} | n_init={best_for_metric['n_init']:<2} | "
            f"Stress={best_for_metric['stress']:.4f} | "
            f"R²={best_for_metric['r2']:.4f}"
        )
    
    best_overall = min(all_results, key=lambda r: r['stress'])
    
    print("\n" + "=" * 70)
    print("Best Overall Configuration (lowest stress):")
    print(
        f"{best_overall['metric'].capitalize()} distance | "
        f"max_iter={best_overall['max_iter']} | n_init={best_overall['n_init']} | "
        f"Stress={best_overall['stress']:.4f} | "
        f"R²={best_overall['r2']:.4f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()