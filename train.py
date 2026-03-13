import pandas as pd
from pandas import Series
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys
from typing import Any, Dict, Tuple, cast

# Standardize path to ensure imports from the same directory work
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_preprocessing import load_and_preprocess_data, save_preprocessors


def train_model(data_path: str, models_dir: str) -> None:
    """
    ULTIMATE TRAINING PIPELINE
    """
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        return

    print(f"🚀 Loading and Enriching records from: {data_path}")
    res_data = load_and_preprocess_data(data_path, is_training=True)
    X, y, encoders, scaler = cast(Tuple[pd.DataFrame, Series, Dict[str, LabelEncoder], StandardScaler], res_data)

    print("⚖️ Balancing dataset using SMOTE...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("🔎 Searching for absolute mathematical peak of SVM Accuracy...")
    param_grid = {
        'C': [1, 10, 20],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }

    grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=2, cv=3)
    grid.fit(X_train_res, y_train_res)

    best_model = grid.best_estimator_
    print(f"💡 Best Parameters Found: {grid.best_params_}")

    print("📊 Final evaluation...")
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Peak Model Accuracy: {accuracy:.4f}")

    print("\n[ Classification Report ]")
    target_le = encoders.get('Burnout_Risk')
    if target_le:
        print(classification_report(y_test, y_pred, target_names=target_le.classes_))

    # Handle Directory Creation
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Fixed path to avoid "models/models/" duplication
    model_save_path = os.path.join(models_dir, 'svm_model.joblib')
    joblib.dump(best_model, model_save_path)

    save_preprocessors(encoders, scaler, models_dir)
    print(f"✨ SUCCESS: Model saved to {model_save_path}")


if __name__ == "__main__":
    # Get the directory where THIS script (train.py) lives
    # This is C:\Users\Ella\PycharmProjects\PythonProject1
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # The CSV is in the same folder as this script
    data_file = os.path.join(current_dir, "Work Productivity.csv")

    # Define the models folder inside the current directory
    models_folder = os.path.join(current_dir, "models")

    train_model(data_file, models_folder)