import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# -------------------------------
# IMPORTANT FEATURES
# -------------------------------
selected_cols = [

    "% 1st Trimester registration to Total ANC Registrations - 2019-20",

    "% Pregnant Woman received 4 ANC check ups to Total ANC Registrations - 2019-20",

    "% Pregnant women given 180 IFA to Total ANC Registration - 2019-20",

    "% New cases of Pregnant Women detected at institution for hypertension to Total ANC Registrations - 2019-20",

    "% Institutional deliveries to Total Reported Deliveries - 2019-20",

    "% Safe deliveries to Total Reported Deliveries - 2019-20",

    "% Home deliveries to Total Reported Deliveries - 2019-20",

    "% C-section deliveries (Public + Pvt.) to reported institutional (Public + Pvt.) deliveries - 2019-20",

    "% cases of Pregnant women with Obstetric Complications and attended to reported deliveries - 2019-20",
 
    "% Complicated Pregnancies treated with Blood Transfusion to Total Women with Obstetric Complications attended - 2019-20",

    "% Pregnant women given 360 Calcium to Total ANC Registration - 2019-20",

    "% Pregnant women tested positive for GDM to Total ANC Registration - 2019-20",

    "% Pregnant women tested for Syphilis to Total ANC Registration - 2019-20",

    "% Pregnant women treatd for Syphilis to Total sero positive for Syphilis - 2019-20",

    "Number of Pregnant Women having severe anaemia (Hb<7) treated at institution - 2019-20",


    ]

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []

    for file in files:
        file_path = os.path.join(folder_path, file)

        df = pd.read_csv(file_path)

        state = file.split('-')[-1].split('.')[0].upper()

        df['State'] = state

        df = clean_data(df)

        df.to_csv(file_path, index=False)

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_data(df):
    df = df.copy()

    df = df.loc[:, ~df.columns.str.contains("2018-19")]
    
    if 'Indicators' in df.columns:
        df.rename(columns={'Indicators': 'District'}, inplace=True)
    else:
        df['District'] = 'Unknown'

    keep_cols = ['S.No', 'District', 'State'] + selected_cols;
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]
    
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    for col in df.columns:
        if col not in ['State', 'District']:
            df[col] = df[col].astype(str).str.replace(',', '',regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = df.fillna(0)

    return df


# -------------------------------
# CREATE RISK LABEL
# -------------------------------
def create_risk(df):
    df = df.copy()

    df['risk_score'] = df[selected_cols].median(axis=1)

    low = df['risk_score'].quantile(0.33)
    high = df['risk_score'].quantile(0.66)

    def label(x):
        if x < low:
            return "Low"
        elif x < high and x > low:
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
    df = create_risk(df)

    X = df[selected_cols]
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
