import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
import numpy as np

from typing import Dict, Any, List, Optional, Tuple, Union, cast


def load_and_preprocess_data(
        file_path: Optional[str],
        is_training: bool = True,
        encoders: Optional[Dict[str, LabelEncoder]] = None,
        scaler: Optional[StandardScaler] = None,
        input_df: Optional[pd.DataFrame] = None
) -> Union[Tuple[pd.DataFrame, pd.Series, Dict[str, LabelEncoder], StandardScaler], pd.DataFrame]:
    """
    Advanced preprocessing for maximum prediction accuracy.
    Includes feature enrichment, outlier handling, and ordinal mapping.
    """
    if input_df is not None:
        df = input_df.copy()
    elif file_path is not None:
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Either file_path or input_df must be provided.")

    df = pd.DataFrame(df)
    df = df.dropna().reset_index(drop=True)

    # 1. Cleaning & Standardisation
    if 'Employee_ID' in df.columns:
        df = df.drop('Employee_ID', axis=1)

    if 'Job_Role' in df.columns:
        df['Job_Role'] = df['Job_Role'].replace({'Software Engineer': 'Software Developer'})

    # 2. Manual Ordinal Mapping (Mathematically significant)
    stress_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
    size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3, 'MNC': 4}

    df['Stress_Level'] = df['Stress_Level'].map(stress_mapping).fillna(2)
    df['Company_Size'] = df['Company_Size'].map(size_mapping).fillna(2)

    # 3. Numeric Clamping (Stability Anchor)
    numeric_cols = ['Work_Hours_Per_Day', 'Sleep_Hours', 'Meetings_Per_Day',
                    'Experience_Years', 'Internet_Speed_Mbps', 'Exercise_Hours_Per_Week',
                    'Screen_Time_Hours', 'Productivity_Score']

    for col in numeric_cols:
        if col in df.columns:
            if 'Hours' in col or 'Day' in col:
                df[col] = df[col].clip(0, 24)
            if 'Score' in col:
                df[col] = df[col].clip(0, 100)

    # 4. Advanced Feature Engineering (The "Accuracy Booster")
    # Ratio of Meetings to Work (Over-scheduled?)
    df['Interaction_Load'] = df['Meetings_Per_Day'].astype(float) / (df['Work_Hours_Per_Day'].astype(float) + 1.0)

    # Recovery Score (Biological health)
    df['Health_Buffer'] = df['Sleep_Hours'].astype(float) + (df['Exercise_Hours_Per_Week'].astype(float) / 7.0)

    # Stress-Work Magnitude
    df['Work_Stress_Magnitude'] = df['Work_Hours_Per_Day'].astype(float) * df['Stress_Level'].astype(float)

    # Digital Strain
    df['Digital_Strain'] = df['Screen_Time_Hours'].astype(float) * df['Stress_Level'].astype(float)

    # 5. Categorical Encoding
    remaining_cats = ['Gender', 'Country', 'Job_Role', 'Work_Environment']

    if is_training:
        new_encoders: Dict[str, LabelEncoder] = {}
        for col in remaining_cats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            new_encoders[col] = le

        # Target variable
        target_le = LabelEncoder()
        y_encoded = target_le.fit_transform(df['Burnout_Risk'].astype(str))
        new_encoders['Burnout_Risk'] = target_le

        X = df.drop('Burnout_Risk', axis=1)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Consistent Scaling
        new_scaler = StandardScaler()
        X_scaled = new_scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=X.columns)

        return X_df, pd.Series(y_encoded), new_encoders, new_scaler
    else:
        # Inference Mode
        if encoders is None or scaler is None:
            raise ValueError("Encoders and Scaler must be provided for inference mode.")

        curr_encoders = cast(Dict[str, LabelEncoder], encoders)
        curr_scaler = cast(StandardScaler, scaler)

        for col in remaining_cats:
            if col in curr_encoders:
                le = curr_encoders[col]
                # Default to 0 if label not seen in training; ensures str comparison
                df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0)

        if hasattr(curr_scaler, 'feature_names_in_'):
            expected_cols: List[str] = cast(List[str], getattr(curr_scaler, 'feature_names_in_'))
            df = df[expected_cols]

        X_scaled = curr_scaler.transform(df)
        X_df = pd.DataFrame(X_scaled, columns=df.columns)

        return X_df


# Updated version
def save_preprocessors(encoders, scaler, path):
    # Ensure the directory exists first
    if not os.path.exists(path):
        os.makedirs(path)

    # Save directly into the path provided (which is already the 'models' folder)
    joblib.dump(scaler, os.path.join(path, 'scaler.joblib'))

    # Do the same for your encoders
    for col, le in encoders.items():
        joblib.dump(le, os.path.join(path, f'encoder_{col}.joblib'))

def load_preprocessors(path: str) -> Tuple[Dict[str, LabelEncoder], StandardScaler]:
    encoders = joblib.load(os.path.join(path, 'encoders.joblib'))
    scaler = joblib.load(os.path.join(path, 'models/scaler.joblib'))
    return cast(Dict[str, LabelEncoder], encoders), cast(StandardScaler, scaler)