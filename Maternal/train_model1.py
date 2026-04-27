import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    for file in files:
        df = pd.read_csv(os.path.join(folder_path, file))

        state = file.split('-')[-1].split('.')[0].upper()
        df['State'] = state

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_data(df):
    df = df.copy()

    if 'Indicators' in df.columns:
        df.rename(columns={'Indicators': 'District'}, inplace=True)
    else:
        df['District'] = 'Unknown'

    for col in df.columns:
        if col not in ['State', 'District']:
            df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


# -------------------------------
# CREATE RISK LABEL
# -------------------------------
def create_risk(df):
    df = df.copy()

    num_cols = df.select_dtypes(include=np.number).columns
    selected_cols = num_cols[:min(10, len(num_cols))]

    df['risk_score'] = df[selected_cols].median(axis=1)

    low = df['risk_score'].quantile(0.33)
    high = df['risk_score'].quantile(0.66)

    def label(x):
        if x < low:
            return "Low"
        elif x < high:
            return "Medium"
        else:
            return "High"

    df['Risk_Level'] = df['risk_score'].apply(label)

    return df


# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_and_save():
    folder_path = r"C:\MCA\Project\AIML\Maternal\data"

    df = load_data(folder_path)
    df = clean_data(df)
    df = create_risk(df)

    X = df.select_dtypes(include=np.number).drop(['risk_score'], axis=1)
    y = df['Risk_Level']

    feature_columns = X.columns.tolist()

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save everything
    os.makedirs("model", exist_ok=True)

    pickle.dump(model, open("model/model.pkl", "wb"))
    pickle.dump(scaler, open("model/scaler.pkl", "wb"))
    pickle.dump(le, open("model/encoder.pkl", "wb"))
    pickle.dump(feature_columns, open("model/features.pkl", "wb"))
    pickle.dump(imputer, open("model/imputer.pkl", "wb"))

    print(" Model trained and saved successfully!")


if __name__ == "__main__":
    train_and_save()