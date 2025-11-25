from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from pathlib import Path

# Input - use the FULL prepared dataset (not the 80% subset)
in_path = sys.argv[1] if len(sys.argv) > 1 else "sampled_tsne_prepared.csv"
df_prepared = pd.read_csv(in_path)

print(f"Total samples in dataset: {len(df_prepared)}")

# Split
X = df_prepared.drop(columns=["label"])
y = df_prepared["label"]

# First split: 80% for training, 20% for initial test (your current split)
X_learn, X_test_20, y_learn, y_test_20 = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)

# Now split the training data again to get additional 1000 samples
# Calculate what percentage 1000 is of the remaining data
remaining_samples = len(X_learn)
test_1000_ratio = min(1000 / remaining_samples, 0.5)  # Cap at 50% to ensure enough training data

X_train_final, X_test_1000, y_train_final, y_test_1000 = train_test_split(
    X_learn, y_learn, test_size=test_1000_ratio, random_state=123, shuffle=True, stratify=y_learn
)

print(f"\nData split summary:")
print(f"Training set: {len(X_train_final)} samples")
print(f"Test set (20%): {len(X_test_20)} samples")
print(f"Additional test (1000): {len(X_test_1000)} samples")
print(f"Total: {len(X_train_final) + len(X_test_20) + len(X_test_1000)} samples")

# Ensure output folders exist
out_dir = Path("tsne_data")
out_dir.mkdir(parents=True, exist_ok=True)

# Save all datasets
pd.concat([X_train_final, y_train_final], axis=1).to_csv(out_dir / "train_final.csv", index=False)
pd.concat([X_test_20, y_test_20], axis=1).to_csv(out_dir / "test_20.csv", index=False)
pd.concat([X_test_1000, y_test_1000], axis=1).to_csv(out_dir / "test_1000.csv", index=False)

# Also create the 80% "learn_valid" file for backward compatibility
pd.concat([X_learn, y_learn], axis=1).to_csv(out_dir / "learn_valid_80.csv", index=False)

print(f"\nWrote files to {out_dir}:")
print(f"  - train_final.csv ({len(X_train_final)} samples)")
print(f"  - test_20.csv ({len(X_test_20)} samples)")
print(f"  - test_1000.csv ({len(X_test_1000)} samples)")
print(f"  - learn_valid_80.csv ({len(X_learn)} samples) [for compatibility]")

# Print class distribution
print(f"\nClass distribution:")
print(f"Training: {y_train_final.value_counts().to_dict()}")
print(f"Test 20%: {y_test_20.value_counts().to_dict()}")
print(f"Test 1000: {y_test_1000.value_counts().to_dict()}")