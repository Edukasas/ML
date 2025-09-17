import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load sampled datasets
dfs = []
for label in [0, 1, 2]:
    df = pd.read_csv(f"sampled_label_{label}.csv")
    df['label'] = label
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

features = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]

# 1. Descriptive statistics for data with missing values
print("Descriptive statistics for data with missing values:")
desc_missing = data[features].describe().T
desc_missing["median"] = data[features].median()
desc_missing["dispersion"] = data[features].var()
print(desc_missing)

# 2. Fill missing data using mean and median per label, and compare
# Fill with mean per label
data_filled_mean = data.copy()
for feature in features:
    data_filled_mean[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.mean()))

# Fill with median per label
data_filled_median = data.copy()
for feature in features:
    data_filled_median[feature] = data.groupby("label")[feature].transform(lambda x: x.fillna(x.median()))

print("\nDescriptive statistics after filling missing values with MEAN (per label):")
desc_mean = data_filled_mean[features].describe().T
desc_mean["median"] = data_filled_mean[features].median()
desc_mean["dispersion"] = data_filled_mean[features].var()
print(desc_mean)

print("\nDescriptive statistics after filling missing values with MEDIAN (per label):")
desc_median = data_filled_median[features].describe().T
desc_median["median"] = data_filled_median[features].median()
desc_median["dispersion"] = data_filled_median[features].var()
print(desc_median)

# --- Continue with the rest of your pipeline using mean-imputed data ---
data_filled = data_filled_median

# Calculate Q1, Q3, and IQR
Q1 = data_filled.quantile(0.25)
Q3 = data_filled.quantile(0.75)
IQR = Q3 - Q1

# Define the outer barriers
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

# Create a mask to filter out extreme outliers
mask = ~((data_filled < lower_bound) | (data_filled > upper_bound)).any(axis=1)

# Apply the mask to the data
data_no_outliers = data_filled[mask]

# Min-max normalization (on data without outliers)
data_min_max_normalized = data_no_outliers.copy()
for feature in features:
    x_min = data_no_outliers[feature].min()
    x_max = data_no_outliers[feature].max()
    data_min_max_normalized[feature] = (data_no_outliers[feature] - x_min) / (x_max - x_min)

# Normalization by mean and standard deviation (on data without outliers)
data_standard_normalized = data_no_outliers.copy()
for feature in features:
    x_mean = data_no_outliers[feature].mean()
    x_std = data_no_outliers[feature].std()
    data_standard_normalized[feature] = (data_no_outliers[feature] - x_mean) / x_std

# Descriptive statistics for the original, filled, and outlier-removed data
print("\nDescriptive statistics for original data (with missing values filled using median):")
desc = data_filled[features].describe().T
desc["median"] = data_filled[features].median()
desc["dispersion"] = data_filled[features].var()
print(desc)

print("\nDescriptive statistics after outlier removal:")
desc_no_out = data_no_outliers[features].describe().T
desc_no_out["median"] = data_no_outliers[features].median()
desc_no_out["dispersion"] = data_no_outliers[features].var()
print(desc_no_out)

print("\nDescriptive statistics for Min-Max normalized data:")
desc_minmax = data_min_max_normalized[features].describe().T
desc_minmax["median"] = data_min_max_normalized[features].median()
desc_minmax["dispersion"] = data_min_max_normalized[features].var()
print(desc_minmax)

print("\nDescriptive statistics for Standard normalized data:")
desc_std = data_standard_normalized[features].describe().T
desc_std["median"] = data_standard_normalized[features].median()
desc_std["dispersion"] = data_standard_normalized[features].var()
print(desc_std)

# Visual analysis

# 1. Scatter plot matrix (pairplot) for features, colored by label
sns.pairplot(data_no_outliers, vars=features, hue="label", diag_kind="hist")
plt.suptitle("Scatter Plot Matrix (Outliers Removed)", y=1.02)
plt.show()

# 2. Box plots (rectangular plots) for each feature by label
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x="label", y=feature, data=data_no_outliers)
    plt.title(f"Boxplot of {feature} by Label")
plt.tight_layout()
plt.show()

# 3. Histograms for each feature, separated by label
plt.figure(figsize=(15, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i+1)
    for label in sorted(data_no_outliers['label'].unique()):
        sns.histplot(
            data_no_outliers[data_no_outliers['label'] == label][feature],
            label=f"Label {label}", kde=True, stat="density", element="step", fill=False
        )
    plt.title(f"Histogram of {feature}")
    plt.legend()
plt.tight_layout()
plt.show()

labels = sorted(data_no_outliers['label'].unique())
fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 5))
if len(labels) == 1:
    axes = [axes]
for idx, label in enumerate(labels):
    corr_label = data_no_outliers[data_no_outliers['label'] == label][features].corr()
    sns.heatmap(corr_label, annot=True, cmap="coolwarm", fmt=".2f", ax=axes[idx])
    axes[idx].set_title(f"Correlation Heatmap (Label {label})")
plt.tight_layout()
plt.show()
