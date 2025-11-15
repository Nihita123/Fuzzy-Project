Predicting Groundwater Fluoride Levels Using Machine Learning and Fuzzy Logic

A comprehensive framework combining Machine Learning, Regression models, and Fuzzy Logic to predict groundwater fluoride levels across India and classify regions into data-driven fluoride risk categories. This project supports large-scale groundwater quality assessment and assists decision-makers in identifying fluoride-vulnerable zones.

🚀 Project Overview

Groundwater is a primary drinking water source in India, but elevated fluoride concentrations pose severe health risks. Traditional testing methods are slow, costly, and region-limited.

This project builds an integrated analytics pipeline that:

Predicts fluoride levels using Regression models

Classifies water samples into Safe, Moderate, and High-risk using ML

Applies a Fuzzy Inference System (FIS) for human-interpretable risk assessment

Generates state-level safety summaries and risk visualizations

Using a dataset of 16,776 groundwater samples from multiple Indian states and districts, the system offers a scalable and intelligent tool for groundwater quality management.

📂 Dataset Description

Each record in the dataset corresponds to a groundwater sampling point and includes:

Physicochemical parameters:
pH, EC, TDS, Na⁺, Ca²⁺, Mg²⁺, K⁺, Cl⁻, SO₄²⁻, NO₃⁻, HCO₃⁻

Fluoride concentration (F⁻)

State and district identifiers

These hydrochemical variables influence fluoride mobility and are essential for predictive modeling.

🧹 Data Preprocessing Pipeline

The dataset undergoes a structured multi-stage preprocessing workflow:

✔ Standardization of Column Names

Removes units and symbols, detects key hydrochemical features (pH, EC, TDS, etc.) automatically.

✔ Handling Missing and Invalid Values

Converts placeholders (NA, “–”, empty) to NaN

Uses median imputation for numerical data

✔ Creation of Fluoride Risk Classes

Based on WHO standards:

Class 0: < 1.5 mg/L (Safe)

Class 1: 1.5–2.5 mg/L (Moderate Risk)

Class 2: > 2.5 mg/L (High Risk)

✔ Feature Scaling

Min–Max normalization to ensure uniform scale

✔ Encoding Categorical Features

One-Hot Encoding for state/district/location-type columns

✔ Handling Imbalanced Classes

Uses SMOTE to generate synthetic samples

Achieves balanced distribution across all 3 risk categories

🤖 Machine Learning Models Used

Seven classification models were trained and compared:

Logistic Regression

Support Vector Machine (SVM – RBF kernel)

Artificial Neural Network (ANN)

AdaBoost

XGBoost

LightGBM

Random Forest

🎯 Best performer:
Random Forest Classifier — 93% Accuracy

📈 Regression Models

To estimate continuous fluoride concentration, three regressors were tested:

Model	R² Score	RMSE
Linear Regression	0.218	0.709
Random Forest Regressor	0.273	0.684
SVR (RBF)	0.174	0.729

🏆 Best model: Random Forest Regressor

Predicted fluoride values are fed into the fuzzy logic system for further interpretation.

🧠 Fuzzy Logic Risk Classification

A Mamdani-type Fuzzy Inference System (FIS) was developed for interpretable risk scoring.

Fuzzy Input Categories (Fluoride):

Very Low

Low

Normal

High

Very High

Fuzzy Output Categories (Risk Score):

Low Risk

Medium Risk

High Risk

Sample Fuzzy Rules:

If Fluoride is Very High → Risk is High

If Fluoride is Normal → Risk is Low

If Fluoride is Low → Risk is Medium

Final risk labels:

Low Risk: score < 33

Medium Risk: 33–66

High Risk: > 66

📊 Results
✔ Machine Learning Classification

Best Accuracy: 93%

High precision, recall, and F1-score for all classes

Confusion matrix shows minimal cross-class error

✔ Regression

Random Forest Regressor chosen for final fluoride prediction

✔ Fuzzy Logic

Generates risk heatmaps

Computes state-wise mean risk score

Produces overall risk distribution

⚠️ Limitations

Dataset originally imbalanced

Does not include contaminants like heavy metals, perchlorates, etc.

No temporal or seasonal variation included

Spatial hydrogeology not explicitly modeled

🔮 Future Enhancements

Integration with GIS maps for spatial visualization

Multi-contaminant groundwater quality prediction

Use of Deep Learning and hybrid ML–geostatistical models

Explainability tools (SHAP, LIME)

Real-time automated monitoring pipeline

🏁 Conclusion

This project presents a powerful data-driven framework that combines Machine Learning and Fuzzy Logic to accurately classify groundwater fluoride levels and assess risk across India.
The system supports policymakers, researchers, and water authorities in identifying unsafe regions and improving groundwater safety.

📜 How to Run the Project
# Clone the repository
git clone https://github.com/USERNAME/REPOSITORY

# Navigate to project folder
cd REPOSITORY

# Install required packages
pip install -r requirements.txt

# Run the main script
python main.py
🤝 Contributors

Aishwarya Para (2023BMS-022)

Nihita Kolukula (2023BMS-015)
