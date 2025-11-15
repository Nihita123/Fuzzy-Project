💧 Predicting Groundwater Fluoride Levels — Machine Learning & Fuzzy Logic

A data-driven, interpretable groundwater quality assessment project

💡 Overview

This project builds an integrated framework that combines Machine Learning, Regression models, and a Mamdani-type Fuzzy Inference System (FIS) to:

Predict continuous fluoride concentrations in groundwater, and

Classify sampling sites into Safe / Moderate / High-risk categories

Using a dataset of ~16,776 groundwater samples collected across Indian states and union territories, the system is designed for accuracy, interpretability, and scalability — helping policy makers and water managers prioritize interventions.

Repository: https://github.com/Nihita123/Fuzzy-Project

🧩 Key Features

📈 Classification: Random Forest, XGBoost, LightGBM, SVM, Logistic Regression, AdaBoost, ANN

🔬 Regression: Random Forest Regressor, Linear Regression, SVR (to predict fluoride mg/L)

🧠 Fuzzy Inference System: Mamdani FIS produces a 0–100 risk score and Low/Medium/High labels

🔁 Data Pipeline: Standardization, median imputation, Min–Max scaling, One-Hot encoding

⚖️ Class Balance: SMOTE to address class imbalance (final balanced dataset = 16,776 samples)

📊 State-level aggregation & visualization: Mean fluoride, mean fuzzy risk score, label distributions

♻️ Reproducible preprocessing objects saved for deployment (scaler, imputer, encoders)

📂 Project Structure
Fuzzy-Project/
├── data/
│   ├── raw/                     # raw CSVs (if available)
│   └── processed/               # processed model-ready dataset
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_classification_models.ipynb
│   ├── 03_regression_models.ipynb
│   └── 04_fuzzy_logic_and_aggregation.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models/
│   │   ├── train_classifiers.py
│   │   ├── train_regressors.py
│   │   └── evaluate.py
│   ├── fuzzy/
│   │   └── fuzzy_system.py
│   └── visualization.py
├── saved_objects/
│   ├── scaler.pkl
│   ├── imputer.pkl
│   └── classifier_best.pkl
├── reports/
│   └── FUZZY_REPORT.pdf
├── requirements.txt
└── README.md

🛠️ Libraries & Tools
Library	Purpose
Python	Core language
pandas, numpy	Data handling & numeric ops
scikit-learn	ML models, evaluation, SMOTE
xgboost, lightgbm	Gradient boosting models
scikit-fuzzy / skfuzzy	Fuzzy membership & Mamdani FIS
matplotlib / seaborn	Plots & visualizations
joblib / pickle	Save preprocessing & models
⚙️ Data Preprocessing (Concise)

Column standardization — unify names, remove units/symbols.

Invalid→NaN — convert placeholders (NA, “—”) to NaN.

Median imputation — fill numeric missings.

Create risk classes (WHO-based):

Class 0: F < 1.5 mg/L (Safe)

Class 1: 1.5 ≤ F ≤ 2.5 mg/L (Moderate)

Class 2: F > 2.5 mg/L (High Risk)

Min–Max scaling to [0,1].

One-Hot encode categorical fields (state, district, well-type).

SMOTE to balance classes → split 70% train / 30% test.

🤖 Models & Metrics (Highlights)
Classification (best results)

Best classifier: Random Forest — ~93% accuracy

Classification report (best model):

Class 0 (Low): Precision 0.94, Recall 0.93, F1 0.93

Class 1 (Med): Precision 0.91, Recall 0.91, F1 0.91

Class 2 (High): Precision 0.93, Recall 0.94, F1 0.94

Confusion matrix showed most misclassifications occur between adjacent classes (Low ↔ Medium).

Regression
Model	R²	RMSE
Linear Regression	0.218	0.709
Random Forest Regressor	0.273	0.684
SVR (RBF)	0.174	0.729

Chosen regressor: Random Forest Regressor (best R² / lowest RMSE)

🧭 Fuzzy Logic System

Input: Predicted fluoride concentration (range 0–4 mg/L)

Input fuzzy sets: Very Low, Low, Normal, High, Very High (triangular MFs)

Output fuzzy sets: Low Risk, Medium Risk, High Risk (0–100 score)

Rules: Example – If Fluoride is Very High → Risk is High

Defuzzification: Centroid method → crisp risk score (0–100)

Label thresholds:

Low Risk: score < 33

Medium Risk: 33 ≤ score < 66

High Risk: score ≥ 66

📊 Sample Results & Visuals

Physicochemical summary (excerpt):

Parameter	Min	Max	Mean
pH	2.54	9.85	7.69
EC (µS/cm)	12	84660	1298.10
F⁻ (mg/L)	0	22	0.74

Visual outputs included:

ROC curves for different classifiers

Confusion matrix heatmap

Fuzzy risk heatmap & state-wise risk distribution plots

(Plots available in notebooks/ and reports/FUZZY_REPORT.pdf)

📈 Decision Rules (Example)

If ≥ 0.66 predicted fluoride (mg/L) and fuzzy risk score ≥ 66 → High Risk

If majority of neighboring samples in same state show high fuzzy risk → flag state-level alert

These rules are configurable in src/fuzzy/fuzzy_system.py.

🚀 How to Run

Clone the repo

git clone https://github.com/Nihita123/Fuzzy-Project.git
cd Fuzzy-Project


Create virtual env & install

python -m venv venv
source venv/bin/activate     # on Linux/macOS
# .\venv\Scripts\activate    # on Windows
pip install -r requirements.txt


Preprocess & train (notebooks or scripts)

Option A — run notebooks:

Open Jupyter: jupyter notebook

Run 01_data_preprocessing.ipynb → 02_classification_models.ipynb → 03_regression_models.ipynb → 04_fuzzy_logic_and_aggregation.ipynb

Option B — run scripts:

python src/preprocessing.py
python src/models/train_classifiers.py
python src/models/train_regressors.py
python src/fuzzy/fuzzy_system.py


View results & visualizations

Outputs and plots are saved under reports/ and plots/.

🧪 Reproducibility & Deployment

Preprocessing objects (scaler, imputer, encoders) are saved as pickles under saved_objects/ for consistent inference.

The trained classifier and regressor pickles are also saved for deployment.

For production, wrap prediction + FIS in an API (FastAPI/Flask) and serve with the saved objects.

⚠️ Limitations

Original raw dataset imbalance (handled by SMOTE but still a caution).

No seasonal / temporal features included — cannot capture monsoon/seasonal shifts.

Not all contaminants present (heavy metals, perchlorates, etc.) — single-contaminant focus.

Spatial hydrogeological characteristics (aquifer depth, lithology) not explicitly modeled.

🔮 Future Work

Integrate GIS mapping and spatial interpolation (kriging) for continuous contamination maps.

Add temporal datasets to model seasonal trends.

Expand to multi-contaminant assessment (e.g., fluoride + nitrate + heavy metals).

Apply explainable AI (SHAP/LIME) for per-sample feature attribution.

Create a web dashboard (Dash / Streamlit) for interactive state-level monitoring.

👥 Contributors
Name	Role
Nihita Kolukula	Core modeling, fuzzy system, report
Aishwarya Para	Data preprocessing, visualizations, documentation
