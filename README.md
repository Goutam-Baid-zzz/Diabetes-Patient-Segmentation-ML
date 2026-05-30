# 🫀 MedSegment: Patient Segmentation & Healthcare Intelligence

An interactive **Streamlit** dashboard powered by **Machine Learning (KMeans Clustering)** to segment diabetic patients from the **UCI Diabetes 130-US Hospitals dataset**. This platform categorizes patients into three distinct, clinically meaningful cohorts to help healthcare providers optimize clinical outcomes, reduce readmission rates, customize care plans, and allocate resources efficiently.

---

## 🚀 Key Features

*   **Interactive Patient Analysis**: Selects random patients from the database to predict their risk cohort, compare individual stats against cohort averages, and view clear clinical reasons for classification ("Why this patient is in this cluster").
*   **Actionable Clinical Care Recommendations**: Generates tailored care checklists and monitoring frequencies based on the patient's segment (e.g., renal monitoring, medication adherence).
*   **Interactive Visual Analytics**: Visualizes multidimensional patient features using custom radar charts, 3D scatter plots, donut charts, and trend line graphs built with **Plotly**.
*   **Executive Dashboard**: Key Performance Indicators (KPIs) for hospital administrators including population distribution, overall readmission rates, and targeted reduction estimates.
*   **Clinical Theme**: Customized premium dark-mode interface utilizing a modern color palette (`#00d4ff` Cyan, `#10b981` Emerald, `#f59e0b` Amber, and `#ef4444` Crimson) for immediate clinical readability.

---

## 📊 Patient Segments (Cohorts)

The machine learning pipeline clusters patients into three distinct risk tiers based on clinical and demographic features:

| Cluster | Cohort Name | Risk Level | Readmission Rate | Population % | Key Clinical Characteristics | Focus Area / Care Plan |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **0** | **Moderate Complexity — Renal Focus** | Moderate | 47.17% | 34.4% | High medication intensity (avg. 23 meds), 71.7% insulin users, 11.7% with renal disease. | Regular nephrology checkups, blood pressure control (<130/80 mmHg), medication adherence checks. |
| **1** | **Low Complexity — Stable & Controlled** | Low | 41.53% | 44.3% | Lowest complexity score (18.8), fewest medications (avg. 12), minimal hospital visits/stays. | Annual wellness visits, lifestyle/exercise coaching, preventive screening, telemedicine. |
| **2** | **High-Risk — Frequent Hospitalizations** | Critical | 72.61% | 20.7% | Extremely high inpatient (avg. 3.2) and emergency (avg. 1.2) visits, 15.8% renal disease. | Intensive transitional care, post-discharge follow-up within 48h, home health visits. |

---

## 🛠️ File Structure & Purpose

Below is the directory mapping for the project assets:

*   **[app.py](file:///g:/Healthcare%20Patient%20Segmentation/app.py)**: The main **Streamlit** dashboard application containing page layouts (Home, Patient Analysis, Cluster Statistics, Dashboard), styling, and real-time model predictions.
*   **[generate_assets.py](file:///g:/Healthcare%20Patient%20Segmentation/generate_assets.py)**: Python script used to clean the raw diabetic data, engineer custom features (e.g., medication intensity, comorbidity counts), train/save the KMeans model and standardizer.
*   **[mapped.py](file:///g:/Healthcare%20Patient%20Segmentation/mapped.py)**: Utility script that maps raw database integers (admission codes, discharge disposition IDs) to human-readable strings.
*   **[kmeans_cluster_k3.pkl](file:///g:/Healthcare%20Patient%20Segmentation/kmeans_cluster_k3.pkl)**: Serialized **KMeans model ($k=3$)** used to predict patient categories.
*   **[scaler.pkl](file:///g:/Healthcare%20Patient%20Segmentation/scaler.pkl)**: Serialized **StandardScaler** mapping to normalize inputs prior to model predictions.
*   **[diabetic_data_with_clusters.csv](file:///g:/Healthcare%20Patient%20Segmentation/diabetic_data_with_clusters.csv)**: Preprocessed patient database complete with features and assigned cluster labels.
*   **[main.ipynb](file:///g:/Healthcare%20Patient%20Segmentation/main.ipynb)**: Jupyter notebook used for initial exploratory data analysis (EDA), statistical validation, and clustering experimentation.
*   **`dataset/`**: Directory containing raw database files:
    *   **[dataset/diabetic_data.csv](file:///g:/Healthcare%20Patient%20Segmentation/dataset/diabetic_data.csv)**: Raw input CSV containing clinical and demographic information.
    *   **[dataset/IDS_mapping.csv](file:///g:/Healthcare%20Patient%20Segmentation/dataset/IDS_mapping.csv)**: Standard description mapping for patient metadata IDs.
    *   **[dataset/mapped_Dataset.csv](file:///g:/Healthcare%20Patient%20Segmentation/dataset/mapped_Dataset.csv)**: Intermediate preprocessed data containing readable descriptive fields.

---

## ⚙️ How to Setup and Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
Install all required libraries via pip:
```bash
pip install pandas numpy scikit-learn joblib streamlit plotly
```

### 3. Generate Machine Learning Assets
If you wish to re-train the models and rebuild the preprocessed dataset, run the asset generator:
```bash
python generate_assets.py
```
This will fit the standardizer and KMeans models and export the serialized binary assets (`scaler.pkl` and `kmeans_cluster_k3.pkl`) alongside `diabetic_data_with_clusters.csv`.

### 4. Run the Streamlit Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser (typically at `http://localhost:8501`).

---

## 🧠 Machine Learning Methodology

1. **Feature Engineering**: Features such as `medication_intensity`, `patient_complexity`, and specific comorbidity flags (`has_renal_disease`, `has_heart_disease`, `has_hypertension`) are computed from raw diagnosis and medication columns.
2. **Feature Scaling**: To prevent features with large scales (like lab procedures) from dominating the distance metric in clustering, a `StandardScaler` is applied to standardise values.
3. **Clustering Algorithm**: KMeans ($k=3$) partition algorithm is run.
4. **Dynamic Mapping**: Clusters are dynamically aligned after training so that the risk tiers consistently map to specific integers:
    *   `Cluster 1` -> Stable / Low Risk (lowest complexity and hospitalisation stats)
    *   `Cluster 0` -> Moderate Risk (high medication density and moderate complexity)
    *   `Cluster 2` -> High Risk (frequent hospital admissions and critical complexity)
