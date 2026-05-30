import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

print("--- 1. Loading mapped dataset ---")
# Load the dataset
df = pd.read_csv("dataset/mapped_Dataset.csv")
print(f"Loaded dataset shape: {df.shape}")

# Keep track of patient_nbr for selection in the app
patient_nbrs = df['patient_nbr'].copy()
original_readmitted = df['readmitted'].copy()

# Preprocess age mapping (matching notebook)
age_bins = {
    '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
    '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
    '[80-90)': 85, '[90-100)': 95
}
df['age_numeric'] = df['age'].map(age_bins)

# Preprocess race
top_races = df['race'].value_counts().head(5).index
df['race'] = df['race'].apply(lambda x: x if x in top_races else 'Other')

# Generate medication features
df['insulin_user'] = (df['insulin'] != 'No').astype(int)

metformin_cols = ['metformin', 'glyburide-metformin', 'glipizide-metformin',
                  'metformin-rosiglitazone', 'metformin-pioglitazone']
df['metformin_user'] = (df[metformin_cols] != 'No').any(axis=1).astype(int)

sulfa_cols = ['glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
              'chlorpropamide', 'tolazamide']
df['sulfonamide_user'] = (df[sulfa_cols] != 'No').any(axis=1).astype(int)

tzd_cols = ['pioglitazone', 'rosiglitazone', 'metformin-rosiglitazone',
            'metformin-pioglitazone']
df['thiazolidinedione_user'] = (df[tzd_cols] != 'No').any(axis=1).astype(int)

med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin']
med_count = (df[med_cols] != 'No').sum(axis=1)
df['medication_intensity'] = (med_count / len(med_cols)) * 10
df['medication_changed'] = (df['change'] != 'No').astype(int)

# Generate condition flags
df['has_diabetes'] = (
    df['diag_1'].astype(str).str[:3].isin(['250']) |
    df['diag_2'].astype(str).str[:3].isin(['250']) |
    df['diag_3'].astype(str).str[:3].isin(['250'])
).astype(int)

df['has_hypertension'] = (
    df['diag_1'].astype(str).str[:3].isin(['401', '402', '403']) |
    df['diag_2'].astype(str).str[:3].isin(['401', '402', '403']) |
    df['diag_3'].astype(str).str[:3].isin(['401', '402', '403'])
).astype(int)

df['has_renal_disease'] = (
    df['diag_1'].astype(str).str[:3].isin(['580', '581', '582', '583', '584', '585']) |
    df['diag_2'].astype(str).str[:3].isin(['580', '581', '582', '583', '584', '585']) |
    df['diag_3'].astype(str).str[:3].isin(['580', '581', '582', '583', '584', '585'])
).astype(int)

df['has_heart_disease'] = (
    df['diag_1'].astype(str).str[:3].isin(['410', '411', '412', '413', '414']) |
    df['diag_2'].astype(str).str[:3].isin(['410', '411', '412', '413', '414']) |
    df['diag_3'].astype(str).str[:3].isin(['410', '411', '412', '413', '414'])
).astype(int)

df['comorbidity_count'] = df['has_hypertension'] + df['has_renal_disease'] + df['has_heart_disease']

# Generate patient complexity index
num_med = df['num_medications'].fillna(0)
num_proc = df['num_procedures'].fillna(0)
los = df['time_in_hospital'].fillna(0)
num_diag = df['number_diagnoses'].fillna(0)
med_intensity = df['medication_intensity'].fillna(0)

num_med_norm = num_med / (num_med.max() + 1)
num_proc_norm = num_proc / (num_proc.max() + 1)
los_norm = los / (los.max() + 1)
num_diag_norm = num_diag / (num_diag.max() + 1)
med_intensity_norm = med_intensity / 10.0

df['patient_complexity'] = (
    0.25 * num_med_norm +
    0.15 * num_proc_norm +
    0.20 * los_norm +
    0.20 * num_diag_norm +
    0.20 * med_intensity_norm
) * 100

# Total recent visits
df['total_recent_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']

# Handle missing values
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Map readmitted to 'Yes' / 'No' (as expected by app.py)
df['readmitted'] = original_readmitted.apply(lambda x: 'No' if x == 'NO' else 'Yes')

# Target features to scale & cluster (13 features matching app.py)
FEATURES_TO_SCALE = [
    'number_inpatient', 'total_recent_visits', 'number_diagnoses',
    'number_emergency', 'number_outpatient', 'time_in_hospital',
    'patient_complexity', 'num_medications', 'insulin_user',
    'medication_changed', 'num_procedures', 'has_renal_disease',
    'num_lab_procedures'
]

print("--- 2. Fitting StandardScaler ---")
scaler = StandardScaler()
X_scaled_values = scaler.fit_transform(df[FEATURES_TO_SCALE])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to: scaler.pkl")

print("--- 3. Fitting KMeans (k=3) ---")
# Fit KMeans on the 13 scaled features
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
initial_clusters = kmeans.fit_predict(X_scaled_values)

# Let's align clusters dynamically with CLUSTER_INFO in app.py:
# Cluster 2 has the highest number_inpatient
# Cluster 0 has the higher patient_complexity among the remaining two
# Cluster 1 has the lowest complexity/medications

cluster_means = pd.DataFrame(X_scaled_values, columns=FEATURES_TO_SCALE)
cluster_means['init_cluster'] = initial_clusters
means = cluster_means.groupby('init_cluster').mean()

# Sort clusters to find which one is which
# 1. Cluster 2 has highest inpatient visits
c2_init = means['number_inpatient'].idxmax()

# 2. Of the remaining two, Cluster 0 has higher complexity
other_clusters = [c for c in range(3) if c != c2_init]
if means.loc[other_clusters[0], 'patient_complexity'] > means.loc[other_clusters[1], 'patient_complexity']:
    c0_init = other_clusters[0]
    c1_init = other_clusters[1]
else:
    c0_init = other_clusters[1]
    c1_init = other_clusters[0]

# Mapping dictionary: init_cluster -> target_cluster
cluster_mapping = {c0_init: 0, c1_init: 1, c2_init: 2}

# Re-map KMeans cluster centers and labels
# We need to rearrange cluster_centers_ so kmeans.predict returns the mapped label
kmeans.cluster_centers_ = kmeans.cluster_centers_[[c0_init, c1_init, c2_init]]
# Also adjust other attributes if needed
if hasattr(kmeans, 'labels_'):
    kmeans.labels_ = np.array([cluster_mapping[label] for label in kmeans.labels_])

# Save the re-mapped KMeans model
joblib.dump(kmeans, 'kmeans_cluster_k3.pkl')
print("KMeans model saved to: kmeans_cluster_k3.pkl")

# Predict using the re-mapped KMeans
final_clusters = kmeans.predict(X_scaled_values)
df['cluster'] = final_clusters

# Keep only necessary columns for the Streamlit app to run efficiently
columns_to_keep = ['patient_nbr', 'readmitted', 'cluster', 'comorbidity_count'] + FEATURES_TO_SCALE
df_output = df[columns_to_keep]

df_output.to_csv("diabetic_data_with_clusters.csv", index=False)
print("Clustered dataset saved to: diabetic_data_with_clusters.csv")
print(f"Output dataset shape: {df_output.shape}")
print("Cluster distribution:")
print(df_output['cluster'].value_counts())
print("Success!")
