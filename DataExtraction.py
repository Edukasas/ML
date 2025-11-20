import pandas as pd

# Load the CSV file with ';' as delimiter
file_path = "EKG_pupsniu_analize.csv"
df = pd.read_csv(file_path, delimiter=';')

# Use all columns (no filtering)
df_selected = df.copy()

for label in [0, 1]:
    df_label = df_selected[df_selected['label'] == label].sample(n=1000, random_state=42)
    output_file = f"sampled_label_{label}.csv"
    df_label.to_csv(output_file, index=False)

# Data quality check for missing data
print("\nMissing values in each column:")
print(df_selected.isnull().sum())

# Optionally, show rows with any missing values
missing_rows = df_selected[df_selected.isnull().any(axis=1)]
print(f"\nNumber of rows with missing values: {len(missing_rows)}")
if not missing_rows.empty:
    print("\nRows with missing values:")
    print(missing_rows)
