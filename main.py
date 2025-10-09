import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]
LABELS = [0, 1, 2]

def load_data():
    dfs = []
    for label in LABELS:
        df = pd.read_csv(f"sampled_label_{label}.csv")
        df['label'] = label
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_stats(data, name=""):
    stats = data[FEATURES].describe().T
    stats["median"] = data[FEATURES].median()
    stats["dispersion"] = data[FEATURES].var()
    if name:
        print(f"\nDescriptive statistics for {name}:")
        print(stats)
    return stats

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
    return data[mask]

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

def plot_scatter_matrix(data):
    plt.figure(figsize=(18, 18))
    num_features = len(FEATURES)
    
    for i in range(num_features):
        for j in range(i + 1):
            plt.subplot(num_features, num_features, i * num_features + j + 1)

            if i != j:
                for label in LABELS:
                    subset = data[data['label'] == label]
                    plt.scatter(subset[FEATURES[j]], subset[FEATURES[i]], 
                    label=f"Label {label}" if (i == 1 and j == 0) else "", alpha=0.6)

            plt.ylabel(FEATURES[i] if j == 0 else "")
            plt.xlabel(FEATURES[j] if i == num_features - 1 else "")
    
    plt.suptitle("Scatter Plot Matrix (Lower Triangle, Outliers Removed)", y=1)
    plt.show()

def plot_boxplots(data, title_suffix=""):
    plt.figure(figsize=(15, 8))
    palette = sns.color_palette("Set2", n_colors=len(LABELS))
    
    for i, feature in enumerate(FEATURES):
        plt.subplot(2, 3, i+1)
        sns.boxplot(x="label", y=feature, data=data, palette=palette)
        plt.title(f"Boxplot of {feature} by Label")
    
    plt.suptitle(f"Box Plots {title_suffix}", y=1.02)
    plt.tight_layout()
    plt.show()

def plot_histograms(data):
    plt.figure(figsize=(15, 8))
    
    for i, feature in enumerate(FEATURES):
        plt.subplot(2, 3, i+1)
        for label in LABELS:
            subset = data[data['label'] == label]
            sns.histplot(subset[feature], label=f"Label {label}", 
                        kde=True, stat="density", element="step", fill=False)
        plt.title(f"Histogram of {feature}")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmaps(data):
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(7, 6 * len(LABELS))) 
    if len(LABELS) == 1:
        axes = [axes]
    for idx, label in enumerate(LABELS):
        corr_label = data[data['label'] == label][FEATURES].corr()
        sns.heatmap(corr_label, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[idx])
        axes[idx].set_title(f"Correlation Heatmap (Label {label})")

    
    plt.tight_layout()
    plt.show()

def plot_frequency_charts(data, bins=10):
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(FEATURES):
        plt.subplot(2, 3, i+1)
        
        for label in LABELS:
            subset = data[data['label'] == label][feature]
            counts, bin_edges = np.histogram(subset, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            plt.bar(bin_centers, counts, alpha=0.7, 
                   label=f'Label {label}', width=(bin_edges[1]-bin_edges[0])*0.8)
        
        plt.title(f'Frequency Chart: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle('Frequency Charts by Feature and Label', y=1.02)
    plt.tight_layout()
    plt.show()


def export_summary(stats_dict):
    import os
    os.makedirs("summary_tables", exist_ok=True)
    for sheet_name, stats in stats_dict.items():
        stats.to_csv(f"summary_tables/{sheet_name}.csv")

def main():
    print("Loading data...")
    data = load_data()
    
    stats_missing = get_stats(data, "data with missing values")
    
    data_mean = fill_missing_values(data, 'mean')
    data_median = fill_missing_values(data, 'median')
    
    stats_mean = get_stats(data_mean, "mean imputed data")
    stats_median = get_stats(data_median, "median imputed data")
    
    data_filled = data_median
    stats_filled = get_stats(data_filled, "filled data")
    
    data_clean = remove_outliers(data_filled)
    stats_clean = get_stats(data_clean, "data after outlier removal")

    data_minmax = normalize_data(data_clean, 'minmax')
    data_standard = normalize_data(data_clean, 'standard')
    
    stats_minmax = get_stats(data_minmax, "Min-Max normalized data")
    stats_standard = get_stats(data_standard, "Standard normalized data")
    
    plot_scatter_matrix(data_clean)
    plot_boxplots(data_clean, "(Outliers Removed)")
    plot_boxplots(data_filled, "(With Outliers)")
    plot_histograms(data_clean)
    plot_correlation_heatmaps(data_clean)
    plot_frequency_charts(data_clean)

    stats_dict = {
        "Missing_Values": stats_missing,
        "Mean_Imputed": stats_mean,
        "Median_Imputed": stats_median,
        "Filled_Data": stats_filled,
        "No_Outliers": stats_clean,
        "MinMax_Normalized": stats_minmax,
        "Standard_Normalized": stats_standard
    }
    export_summary(stats_dict)

if __name__ == "__main__":
    main()
