import pandas as pd

# --- 1. Load the diabetic data ---
diabetic_data = pd.read_csv("dataset/diabetic_data.csv")
print(f"Diabetic data shape: {diabetic_data.shape}")

# --- 2. Parse IDS_mapping.csv (3 tables stacked vertically, separated by blank rows) ---
ids_raw = pd.read_csv("dataset/IDS_mapping.csv", header=None, names=["id", "description"])

# Find blank-row indices to split the three sections
blank_rows = ids_raw[ids_raw["id"].isna() & ids_raw["description"].isna()].index.tolist()

# Split into three sections (skip each section's header row)
# Section 1: admission_type_id  (rows 0 .. blank_rows[0]-1)
# Section 2: discharge_disposition_id  (rows blank_rows[0]+1 .. blank_rows[1]-1)
# Section 3: admission_source_id  (rows blank_rows[1]+1 .. end)

section1 = ids_raw.iloc[1:blank_rows[0]].copy()          # skip header at row 0
section2 = ids_raw.iloc[blank_rows[0]+2:blank_rows[1]].copy()  # skip header row after first blank
section3 = ids_raw.iloc[blank_rows[1]+2:].copy()                # skip header row after second blank
section3 = section3.dropna(subset=["id"])                       # drop trailing blank

# Build lookup dictionaries  (id -> description)
admission_type_map = dict(zip(section1["id"].astype(int), section1["description"].str.strip()))
discharge_disposition_map = dict(zip(section2["id"].astype(int), section2["description"].str.strip()))
admission_source_map = dict(zip(section3["id"].astype(int), section3["description"].str.strip()))

print(f"\nAdmission Type mappings ({len(admission_type_map)}): {admission_type_map}")
print(f"\nDischarge Disposition mappings ({len(discharge_disposition_map)}): {discharge_disposition_map}")
print(f"\nAdmission Source mappings ({len(admission_source_map)}): {admission_source_map}")

# --- 3. Map IDs to descriptions in the diabetic data ---
diabetic_data["admission_type_id"] = diabetic_data["admission_type_id"].map(admission_type_map)
diabetic_data["discharge_disposition_id"] = diabetic_data["discharge_disposition_id"].map(discharge_disposition_map)
diabetic_data["admission_source_id"] = diabetic_data["admission_source_id"].map(admission_source_map)

# --- 4. Save ---
output_path = "dataset/mapped_Dataset.csv"
diabetic_data.to_csv(output_path, index=False)
print(f"\nMapped dataset saved to: {output_path}")
print(f"Output shape: {diabetic_data.shape}")
print(f"\n--- Sample of mapped columns ---")
print(diabetic_data[["admission_type_id", "discharge_disposition_id", "admission_source_id"]].head(10))
