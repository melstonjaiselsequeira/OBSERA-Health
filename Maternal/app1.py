import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Maternal Health Dashboard",
    layout="wide"
)

# ---------------- CUSTOM UI STYLE ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    border-radius: 10px;
    height: 3em;
    background-color: #2E86C1;
    color: white;
    font-weight: bold;
}
.card {
    padding:20px;
    border-radius:15px;
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    text-align:center;
    color:black;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
        }
            
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
le = pickle.load(open("model/encoder.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))
imputer = pickle.load(open("model/imputer.pkl", "rb"))

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    folder_path = r"C:/MCA/Project/AIML/Maternal/data"

    dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder_path, file))
            state = file.split('-')[-1].split('.')[0].upper()
            df['State'] = state
            dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    if 'Indicators' in df.columns:
        df.rename(columns={'Indicators': 'District'}, inplace=True)
    else:
        df['District'] = 'Unknown'

    return df

df = load_data()

# ---------------- DASHBOARD ----------------
if st.session_state.page == "dashboard":

    st.markdown("## 🏥 Maternal Health Risk Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
        <h3>🔍 Risk Prediction</h3>
        <p>Predict maternal health risk</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Go to Prediction", use_container_width=True):
            st.session_state.page = "prediction"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="card">
        <h3>📊 State Comparison</h3>
        <p>Compare states visually</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Compare States", use_container_width=True):
            st.session_state.page = "comparison"
            st.rerun()

# ---------------- PREDICTION ----------------
elif st.session_state.page == "prediction":

    st.markdown("###  Risk Prediction")
    st.markdown("---")

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    df['District'] = df['District'].astype(str)
    df['is_state_row'] = df['District'].str.contains('_', na=False)

    states = sorted(df['State'].unique())
    state = st.selectbox("Select State", states)

    districts = sorted(
        df[df['State'] == state]['District']
        .dropna()
        .astype(str)
        .unique()
    )

    district = st.selectbox("Select District", districts)

    if st.button("Predict Risk"):

        if "_" in district:
            filtered = df[(df['State'] == state) & (df['is_state_row'])]
        else:
            filtered = df[
                (df['State'] == state) &
                (df['District'] == district) &
                (~df['is_state_row'])
            ]

        if filtered.empty:
            st.error("No data available")
        else:
            sample = filtered.head(1)

            X = sample.reindex(columns=features, fill_value=np.nan)
            X = X.astype(str).replace(',', '', regex=True)
            X = X.apply(pd.to_numeric, errors='coerce')

            X = imputer.transform(X)
            X = scaler.transform(X)

            pred = model.predict(X)
            result = le.inverse_transform(pred)[0]

            label = "State Risk" if "_" in district else "District Risk"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("State", state)
            with col2:
                st.metric("District", district)
            with col3:
                st.metric("Risk Level", result)
            # ---------------- RESULT CARD ----------------
            if result == "High":
                 color = "#E74C3C"
                 message = "Immediate medical attention required!"

            elif result == "Medium":
                 color = "#F39C12"
                 message = "Regular monitoring needed."

            else:
                 color = "#2ECC71"
                 message = "Healthy condition."

            st.markdown(f"""
                <div style="
                  background: linear-gradient(135deg, {color}, #111);
                  padding: 30px;
                  border-radius: 20px;
                  text-align: center;
                  color: white;
                  box-shadow: 0px 0px 20px rgba(0,0,0,0.5);
                  margin-top: 20px;
                ">


                  <h2>{label}: {result}</h2>

                  <p style="font-size:18px;">{message}</p>

                </div>
            """, unsafe_allow_html=True)
            

# ---------------- COMPARISON ----------------
elif st.session_state.page == "comparison":

    st.markdown("###  State Comparison")
    st.markdown("---")

    if st.button("⬅ Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    df_numeric = df.copy()

    for col in df_numeric.columns:
        if col not in ['State', 'District']:
            df_numeric[col] = pd.to_numeric(
                df_numeric[col].astype(str).str.replace(',', ''),
                errors='coerce'
            )

    df_numeric = df_numeric.fillna(df_numeric.median(numeric_only=True))
    df_compare = df_numeric.groupby('State').median(numeric_only=True)

    states = df_compare.index.tolist()

    state1 = st.selectbox("Select State 1", states)
    state2 = st.selectbox("Select State 2", states)

    if st.button("Compare"):

        if state1 == state2:
            st.error("Please select two different states")
            st.stop()

        data1 = df_compare.loc[state1]
        data2 = df_compare.loc[state2]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=data1.values,
            x=data1.index,
            mode='lines+markers',
            name=state1,
            line=dict(color='#1f77b4', width=3)
        ))

        fig.add_trace(go.Scatter(
            y=data2.values,
            x=data2.index,
            mode='lines+markers',
            name=state2,
            line=dict(color='#ff7f0e', width=3, )
        ))

        fig.update_layout(
            title=" State Comparison",
            template="plotly_dark",
            title_x=0.5,
            xaxis_title="Indicators",
            yaxis_title="Values"
        )

        st.plotly_chart(fig, use_container_width=True,config={'displayModeBar': False,'staticPlot': True })

