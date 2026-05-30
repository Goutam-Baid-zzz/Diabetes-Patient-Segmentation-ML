# 🫀 MedSegment: Healthcare Patient Segmentation Using Machine Learning

> 🚑 Transforming healthcare data into actionable clinical insights through Machine Learning.

🌐 **Live Application:**  
👉 https://diabetes-patient-segmentation-ml.streamlit.app/

MedSegment is a Machine Learning-powered healthcare analytics platform built with **Streamlit** that segments diabetic patients into clinically meaningful cohorts using **K-Means Clustering**. The project leverages the **Diabetes 130-US Hospitals Dataset** from the UCI Machine Learning Repository to identify patient groups based on hospitalization patterns, disease complexity, medication usage, and readmission behavior.

The platform helps healthcare providers:

* ✅ Identify high-risk patients
* ✅ Reduce hospital readmissions
* ✅ Personalize treatment plans
* ✅ Improve resource allocation
* ✅ Support data-driven clinical decision-making

---

# 📌 Dataset

This project utilizes the **Diabetes 130-US Hospitals for Years 1999–2008 Dataset** from the **UCI Machine Learning Repository**.

### 📊 Dataset Overview

| Attribute    | Value                                      |
| ------------ | ------------------------------------------ |
| 🏥 Hospitals | 130 US Hospitals                           |
| 👥 Records   | 100,000+ Patient Encounters                |
| 📅 Duration  | 1999 – 2008                                |
| 🎯 Domain    | Diabetes Management & Readmission Analysis |
| 📚 Source    | UCI Machine Learning Repository            |

### 🔗 Dataset Link

https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

# ✨ Key Features

## 👤 Interactive Patient Analysis

* Random patient selection from dataset
* Real-time cluster prediction
* Individual patient profiling
* Comparison against cohort averages
* Explainable AI insights
* Clinical reasoning behind classification

## 🩺 Clinical Care Recommendations

* Personalized care plans
* Follow-up recommendations
* Monitoring frequency suggestions
* Preventive intervention guidance
* Risk mitigation strategies

## 📈 Interactive Visual Analytics

* 📊 Radar Charts
* 🌐 3D Cluster Visualizations
* 🍩 Donut Charts
* 📉 Trend Analysis
* 📋 Cohort Comparison Dashboards
* 🎯 Population Distribution Insights

## 🏢 Executive Dashboard

* Healthcare KPIs
* Readmission Analytics
* Population Health Monitoring
* Resource Allocation Insights
* Estimated Intervention Impact

---

# 🧠 Patient Segments

The K-Means model groups patients into three clinically meaningful cohorts.

| Cluster      | Cohort                            | Risk Level | Description                                                                          |
| ------------ | --------------------------------- | ---------- | ------------------------------------------------------------------------------------ |
| 🟡 Cluster 0 | Moderate Complexity – Renal Focus | Moderate   | Patients with higher medication intensity and renal-related complications.           |
| 🟢 Cluster 1 | Stable & Controlled               | Low        | Patients with lower complexity, fewer medications, and minimal hospital utilization. |
| 🔴 Cluster 2 | Frequent Hospitalization          | High       | Patients with repeated admissions, emergency visits, and elevated readmission risk.  |

---

# 🔬 Machine Learning Pipeline

## 1️⃣ Data Preprocessing

* Missing value handling
* Feature selection
* Data cleaning
* Clinical data transformation

## 2️⃣ Feature Engineering

Custom healthcare indicators were developed, including:

* 💊 Medication Intensity Score
* 🏥 Hospital Utilization Score
* ❤️ Heart Disease Indicator
* 🩸 Hypertension Indicator
* 🩺 Renal Disease Indicator
* 📋 Patient Complexity Score

## 3️⃣ Feature Scaling

Features are standardized using **StandardScaler** to ensure fair clustering and prevent scale dominance.

## 4️⃣ Patient Segmentation

Patients are segmented using:

```python
KMeans(n_clusters=3)
```

This identifies naturally occurring patient groups based on clinical and demographic characteristics.

## 5️⃣ Cluster Interpretation

The discovered clusters are mapped into meaningful healthcare risk tiers:

🟢 Low Risk

🟡 Moderate Risk

🔴 High Risk

---

# 📂 Project Structure

```text
MedSegment/
│
├── app.py
├── generate_assets.py
├── mapped.py
├── main.ipynb
│
├── kmeans_cluster_k3.pkl
├── scaler.pkl
├── diabetic_data_with_clusters.csv
│
└── dataset/
    ├── diabetic_data.csv
    ├── IDS_mapping.csv
    └── mapped_Dataset.csv
```

---

# 📁 File Description

| File                               | Description                                                 |
| ---------------------------------- | ----------------------------------------------------------- |
| 🚀 app.py                          | Main Streamlit dashboard application                        |
| ⚙️ generate_assets.py              | Data preprocessing, feature engineering, and model training |
| 🔄 mapped.py                       | Utility functions for readable healthcare mappings          |
| 📓 main.ipynb                      | Exploratory Data Analysis and experimentation               |
| 🤖 kmeans_cluster_k3.pkl           | Trained K-Means clustering model                            |
| 📏 scaler.pkl                      | Trained StandardScaler                                      |
| 📊 diabetic_data_with_clusters.csv | Processed dataset with assigned cluster labels              |

---

# 🛠️ Technology Stack

### 💻 Programming & Analytics

* Python
* Pandas
* NumPy

### 🤖 Machine Learning

* Scikit-Learn
* K-Means Clustering
* StandardScaler

### 📊 Visualization

* Plotly
* Streamlit

### 📦 Utilities

* Joblib

---

# ⚙️ Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/MedSegment.git
cd MedSegment
```

## Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit plotly joblib
```

---

# ▶️ Running the Application

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 🔄 Rebuild Model Assets

To retrain the clustering model and regenerate processed files:

```bash
python generate_assets.py
```

Generated files:

```text
scaler.pkl
kmeans_cluster_k3.pkl
diabetic_data_with_clusters.csv
```

---

# 📈 Business & Healthcare Impact

### 🎯 Clinical Benefits

* Early identification of vulnerable patients
* Improved personalized care delivery
* Better patient monitoring
* Reduced readmission rates

### 🏥 Hospital Benefits

* Efficient resource utilization
* Improved operational planning
* Data-driven healthcare decisions
* Population health management

---

# 🚀 Future Enhancements

* 🔮 Readmission Risk Prediction Model
* 🤖 Advanced Explainable AI Module
* 📱 Mobile-Friendly Dashboard
* ☁️ Cloud Deployment Support
* 📊 Real-Time Hospital Analytics
* 🧬 Disease Progression Forecasting

---

# 👨‍💻 Author

Developed as a Healthcare Analytics and Machine Learning project focused on patient segmentation, risk stratification, and clinical decision support using real-world hospital data.

---

# 📜 License

This project is intended for educational, research, and healthcare analytics purposes.
