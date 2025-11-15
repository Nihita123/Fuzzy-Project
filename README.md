🎯 Groundwater Fluoride Prediction Using Machine Learning & Fuzzy Logic

A data-driven, intelligent, and scalable framework to analyze groundwater fluoride contamination across India using Machine Learning, Regression Models, and a Fuzzy Inference System (FIS).
This system supports early detection of fluoride-vulnerable regions and helps government agencies & water-resource managers make informed decisions.

🌍 Project At a Glance

✔ Analyzes 16,776+ groundwater samples from Indian states & districts
✔ Predicts fluoride levels using Regression Models
✔ Classifies water into Safe / Moderate / High-risk categories using ML
✔ Uses Fuzzy Logic for human-interpretable risk scoring
✔ Generates state-level analysis & heatmaps
✔ Built for accuracy, interpretability, and large-scale deployment

🧠 Why This Project?

Fluoride contamination is a growing threat in Indian groundwater. Traditional testing is slow, costly, and region-limited.
This project solves that by combining:

🔹 Hydrogeochemical science
🔹 Machine Learning
🔹 Fuzzy Logic interpretation

→ delivering a fast, flexible, and reliable digital solution.

📂 Dataset Overview

Each record contains:

Feature Type	Parameters
Physicochemical	pH, EC, TDS, Na⁺, Ca²⁺, Mg²⁺, K⁺, Cl⁻, SO₄²⁻, NO₃⁻, HCO₃⁻
Target	Fluoride concentration (mg/L)
Location	State + District identifiers

These features significantly impact fluoride mobility inside aquifers.

⚙️ Data Preprocessing Pipeline
🔧 1. Standardization

Cleans and normalizes column names (e.g., “EC µS/cm” → “EC”).

🧹 2. Invalid & Missing Values

Converts “NA”, “–”, blanks to NaN

Uses Median Imputation for numerical stability

🧪 3. Fluoride Risk Label Creation

Based on WHO drinking water standards:

Class	Fluoride Level	Interpretation
0	< 1.5 mg/L	Safe
1	1.5–2.5 mg/L	Moderate Risk
2	> 2.5 mg/L	High Risk
📏 4. Scaling

All features normalized to 0–1 range (Min–Max).

🧩 5. Encode Categorical Features

Uses One-Hot Encoding for state/district/well-type.

⚖️ 6. Balancing the Dataset (SMOTE)

Generates synthetic minority samples → class distribution becomes perfectly balanced.

🤖 Machine Learning Models Implemented

Seven ML algorithms were trained:

Model	Type	Notes
Logistic Regression	Linear	Baseline clarity
SVM (RBF)	Kernel	Captures nonlinearity
ANN	Neural Network	Learns complex patterns
AdaBoost	Ensemble	Focuses on hard samples
XGBoost	Gradient Boosting	Fast + accurate
LightGBM	Boosting	Efficient, large-scale
Random Forest	Ensemble	⭐ Best classifier
🏆 Top Performer: Random Forest Classifier

🎯 Accuracy: 93%
🎯 Strong precision, recall, and F1 across all classes

📈 Regression Models for Continuous Prediction

Three regression models were tested:

Model	R² Score	RMSE
Linear Regression	0.218	0.709
Random Forest Regressor	0.273	0.684
SVR	0.174	0.729

🏅 Best Model: Random Forest Regressor
Used for predicting continuous fluoride values across the dataset.

🌡️ Fuzzy Logic Risk Classification

A Mamdani-type Fuzzy Inference System assigns human-friendly risk labels.

🏷 Input Memberships (Fluoride):

Very Low

Low

Normal

High

Very High

🟦 Output Memberships (Risk Score):

Low Risk

Medium Risk

High Risk

📜 Example Fuzzy Rules:

If Fluoride is Very High → Risk is High

If Fluoride is Normal → Risk is Low

If Fluoride is Low → Risk is Medium

🧮 Final Labels:
Risk Score	Category
< 33	Low
33–66	Medium
≥ 66	High
📊 Key Results
✔ ML Performance

93% accuracy

Low misclassification

Stable precision and recall

✔ Fuzzy Interpretation

Generates state-wise risk maps

Produces score distributions

Improves human understanding of risk levels

✔ Combined System

Machine Learning + Fuzzy Logic =
Accurate + Interpretable + Scalable groundwater risk assessment

⚠️ Limitations

🔸 Dataset originally imbalanced
🔸 Missing contaminants (e.g., heavy metals)
🔸 No temporal (seasonal) variations
🔸 Spatial hydrogeology not explicitly included

🔮 Future Directions

✨ Add GIS heatmaps
✨ Integrate deep learning
✨ Predict multiple contaminants
✨ Use explainable AI (SHAP/LIME)
✨ Build real-time dashboards

📥 Installation & Usage
# Clone the repository
git clone https://github.com/USERNAME/REPOSITORY

# Navigate into project folder
cd REPOSITORY

# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python main.py

👥 Contributors

👩‍💻 Aishwarya Para (2023BMS-022)
👩‍💻 Nihita Kolukula (2023BMS-015)
