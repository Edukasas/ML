from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from pathlib import Path

# input
in_path = sys.argv[1] if len(sys.argv) > 1 else "sampled_all_prepared.csv"  # accepts CLI arg [web:79]
df_prepared = pd.read_csv(in_path)

# split
X = df_prepared.drop(columns=["label"])
y = df_prepared["label"]
X_learn, X_test, y_learn, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, shuffle=True, stratify=y
)  # stratified 80/20 split keeps class balance [web:136]

# ensure output folder exists
out_dir = Path("original_all_features_data")
out_dir.mkdir(parents=True, exist_ok=True)

# save to folder
pd.concat([X_learn, y_learn], axis=1).to_csv(out_dir / "learn_valid_80.csv", index=False)  # write CSVs [web:87]
pd.concat([X_test, y_test], axis=1).to_csv(out_dir / "test_20.csv", index=False)           # write CSVs [web:87]

print(f"Wrote {out_dir/'learn_valid_80.csv'} and {out_dir/'test_20.csv'}")
