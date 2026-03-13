import pandas as pd
import joblib
import os
import sys
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_preprocessing import load_and_preprocess_data


def analyze_importance():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    data_path = os.path.join(base_dir, "Work Productivity.csv")

    # Load assets
    model = joblib.load(os.path.join(models_dir, 'models/svm_model.joblib'))

    # Load and preprocess data
    print("Loading data for analysis...")
    X, y, encoders, scaler = load_and_preprocess_data(data_path, is_training=True)

    # Split to get a test set for importance calculation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Calculating permutation importance (this may take a minute)...")
    # Use a subset of X_test if it's too large to speed up calculation
    n_samples = min(1000, len(X_test))
    result = permutation_importance(model, X_test.iloc[:n_samples], y_test.iloc[:n_samples],
                                    n_repeats=5, random_state=42, n_jobs=-1)

    # Create a dataframe of results
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=False)

    print("\n[ Feature Importance Rankings ]")
    print(importance_df.to_string(index=False))

    top_feature = importance_df.iloc[0]['Feature']
    print(f"\n🏆 The highest impact column is: {top_feature}")


if __name__ == "__main__":
    analyze_importance()