
import os
import pandas as pd

source = "data/processed/metadata/hursat_ibtracs_match_report.csv"
output = "data/processed/metadata/hursat_deduplicated.csv"

df = pd.read_csv(source)

# Identify satellite type from each filename.
df["satellite"] = df["file"].str.extract(
    r"\.(MET-7|HIM-8|MSG-3)\."
)[0]

# Prefer HIM-8, then MET-7, then MSG-3 for each timestamp.
priority = {"HIM-8": 0, "MET-7": 1, "MSG-3": 2}
df["priority"] = df["satellite"].map(priority).fillna(99)

df = df.sort_values("priority")

# Keep one satellite image per storm ID and timestamp.
clean = df.drop_duplicates(
    subset=["SID", "timestamp"],
    keep="first",
).copy()

clean = clean.drop(columns=["priority"])
os.makedirs(os.path.dirname(output), exist_ok=True)
clean.to_csv(output, index=False)

print("Original matched files:", len(df))
print("Unique storm observations:", len(clean))
print("\nSelected satellite types:")
print(clean["satellite"].value_counts().to_string())
print("\nGrades represented:")
print(clean["grade"].value_counts().to_string())
print("\nSaved:", output)
