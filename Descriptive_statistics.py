import pandas as pd
import numpy as np

# EITHER: load your real data
df = pd.read_csv("sampled_all_prepared.csv")  # must contain a 'label' column and the six features

FEATURES = ["RR_l_0", "RR_l_0/RR_l_1", "RR_r_0", "R_val", "P_val", "signal_std"]

# ensure numeric for the six features
for c in FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")  # coerce non-numeric to NaN [web:43][web:49][web:51]

def build_stats_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        s = frame[feat].dropna()
        rows.append(
            {
                "feature": feat,
                "count": s.count(),
                "mean": s.mean(),
                "std": s.std(ddof=1),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "dispersion": s.var(ddof=1),
            }
        )
    tbl = pd.DataFrame(rows).set_index("feature")
    return tbl[["count","mean","std","min","q25","median","q75","max","dispersion"]]

# compute table per label and write to separate sheets
out_path = "feature_summary_by_label.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:  # multi-sheet writer [web:31][web:52]
    for label, g in df.groupby("label", dropna=False):       # group rows by label [web:24]
        sheet = f"label_{label}" if pd.notna(label) else "label_missing"
        tbl = build_stats_table(g)
        tbl.to_excel(writer, sheet_name=sheet, index=True)    # one sheet per label [web:31][web:52]

print(f"Wrote per-label feature summary to {out_path}")
