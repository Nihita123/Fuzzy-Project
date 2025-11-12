# src/model_utils.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

def get_model(name):
    """
    Returns an untrained model instance based on the given name.
    Used for retraining or evaluation after best model selection.
    """
    name = name.upper()

    if name == "LR":
        return LogisticRegression(max_iter=1000)
    elif name == "SVM":
        return SVC(probability=True, kernel='rbf')
    elif name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    elif name == "ADABOOST":
        return AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=2), n_estimators=100, random_state=42)
    elif name == "XGBOOST":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )
    elif name == "LIGHTGBM":
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    elif name == "DNN":
        model = Sequential([
            Input(shape=(10,)),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dense(3, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    else:
        raise ValueError(f"Unknown model name: {name}")
