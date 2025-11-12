"""
Preprocessing for Fluoride dataset (dataset1.csv).
Auto-detects chemical features, cleans data, normalizes, and creates fluoride risk classes.
Categorical features are one-hot encoded automatically.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define chemical features
FEATURE_KEYWORDS = ['pH', 'TDS', 'EC', 'Na', 'Mg', 'Ca', 'K', 'HCO3', 'SO4', 'ClO4', 'Cl', 'NO3']
TARGET_KEYWORDS = ['F', 'Fluoride']

def normalize_col(col):
    """Remove units and extra characters from column name"""
    col = re.sub(r'\(.*?\)', '', str(col))
    return col.strip().lower()

def detect_columns(df):
    """Auto-detect target and feature columns"""
    cols = {normalize_col(c): c for c in df.columns}
    target = None
    for key in TARGET_KEYWORDS:
        for k, v in cols.items():
            if key.lower() in k:
                target = v
                break
        if target:
            break

    features = []
    for key in FEATURE_KEYWORDS:
        for k, v in cols.items():
            if key.lower() in k and v != target:
                features.append(v)
                break
    return target, features

def prepare_dataset(path, verbose=True):
    """Load, clean, normalize dataset and create fluoride class labels"""
    df = pd.read_csv(path, encoding='latin1')
    target_col, features = detect_columns(df)

    if verbose:
        print(f"Detected Features: {features}")
        print(f"Target Column: {target_col}")

    # Replace common non-numeric placeholders with NaN
    df.replace(['-', '—', 'NA', 'na', ''], np.nan, inplace=True)

    # Detect numeric vs categorical features
    numeric_features = []
    categorical_features = []
    for col in features:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().any():
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        except:
            categorical_features.append(col)

    # Impute numeric features
    imputer = SimpleImputer(strategy='median')
    df[numeric_features + [target_col]] = imputer.fit_transform(df[numeric_features + [target_col]])

    # Store raw target for classification
    df['F_raw'] = df[target_col]

    # Risk classification (USEPA)
    df['fluoride_class'] = pd.cut(
        df['F_raw'], bins=[-1, 1.5, 2.5, df['F_raw'].max() + 1], labels=[0, 1, 2]
    ).astype(int)

    # Normalize numeric features
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

    # One-hot encode categorical features
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df_scaled[categorical_features])
        X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_features), index=df_scaled.index)
        X = pd.concat([df_scaled[numeric_features], X_cat_df], axis=1)
    else:
        X = df_scaled[numeric_features]

    y = df['fluoride_class'].astype(int)


    if verbose:
        print(f"Dataset Shape: {df.shape}")
        print(f"Numeric Features: {numeric_features}")
        print(f"Categorical Features: {categorical_features}")
        print(f"Number of NaNs after preprocessing: {X.isna().sum().sum()}")

    return X, y, df_scaled[target_col], df, features, target_col, scaler, encoder


