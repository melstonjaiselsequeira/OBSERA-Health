import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import plotly.graph_objects as go
import shap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="OBSERA Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------------- MODERN UI ----------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    background-color: #07111F;
    color: white;
}

/* Main Background */
.stApp {
      background:
    radial-gradient(
        circle at top left,
        rgba(74, 62, 128, 0.45),
        rgba(16, 95, 129, 0.20),
        rgba(6, 78, 59, 0.06)
    );
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: radial-gradient(
        circle at top left,
        rgba(16, 35, 129, 0.12),
        rgba(16, 35, 129, 0.06)  
        );
    border-right: 1px solid rgba(45,212,191,0.12);
}

/* Sidebar Text */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Cards */
.card {
    background: rgba(16, 135, 129, 0.12); ;
    backdrop-filter: blur(12px);
    border-radius: 22px;
    padding: 25px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.4),0 0 25px rgba(16,185,129,0.18);;
    transition: 0.3s ease;
    border: 1px solid rgba(45,212,191,0.10);
    margin-top: 30px;
}

.card:hover {
    transform: translateY(-6px);

    border: 1px solid rgba(45,212,191,0.25);
    box-shadow:0 10px 35px rgba(0,0,0,0.5),
}

/* Metric Box */
.metric-box {
    background: rgba(100, 145, 129, 0.12);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(45,212,191,0.12);
    box-shadow:0px 4px 20px rgba(0,0,0,0.35);
    transition: 0.3s;
     }
            
.metric-box:hover {
    transform: translateY(-4px);
    border: 1px solid rgba(45,212,191,0.25);
}
        
/* Buttons */
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    background: linear-gradient(135deg, #10B981, #2DD4BF);
    color: white;
    border: none;
    font-weight: bold;
    transition: 0.3s;
    box-shadow:
    0 5px 20px rgba(16,185,129,0.25);
}

.stButton>button:hover {
    transform: scale(1.02);
    background: linear-gradient(135deg, #2DD4BF, #10B981);
    box-shadow:
    0 8px 25px rgba(45,212,191,0.35);
}

/* Titles */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
}

.sub-title {
    color: #D1D5DB;
    font-size: 18px;
    margin-top: -5px;
}

/* Result Card */
.result-card {
    padding: 35px;
    border-radius: 25px;
    text-align:center;
    color:white;
    font-size:20px;
    font-weight:bold;
    margin-top:20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.08);
}

               
/* Remove heading link icon */
.stMarkdown a {
    display: none !important;
}
            
/* Inputs */
.stSelectbox label {
    background: linear-gradient(135deg, rgba(60, 155, 129, 0.12));
    border-radius: 8px;
    color: white !important;
    font-weight: bold;
}
            
/* Navigation Box */

/* Radio Items */
.stRadio label {

    background: rgba(255,255,255,0.03);
width: 150%;
    padding: 12px 14px;

    border-radius: 12px;

    margin-bottom: 8px;

    transition: 0.3s;

    color: white !important;
}

/* Hide Empty Radio Label */
.stRadio > label {
    display: none;
}

/* Selected */
.stRadio label[data-baseweb="radio"]:has(input:checked) {

    background: linear-gradient(
        135deg,
        rgba(16,185,129,0.35),
        rgba(45,212,191,0.15)
    );
    min-width: 150%;
    width: 150%;
    border-radius: 12px;

    border: 1px solid rgba(45,212,191,0.25);

    box-shadow:
        0 0 15px rgba(45,212,191,0.15);
} 


/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model/model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
le = pickle.load(open("model/encoder.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))
imputer = pickle.load(open("model/imputer.pkl", "rb"))

# ---------------- SHAP ----------------
explainer = shap.TreeExplainer(model)

# ---------------- PREPROCESS ----------------
def preprocess_input(df):
    df = df.astype(str).replace(',', '', regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

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

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ---------------- SIDEBAR ----------------
pages = ["Dashboard", "Prediction", "Comparison"]

with st.sidebar:

    st.markdown("# 🩺 OBSERA Health\n\nNavigation")
    

    selected_page = st.radio(
        "",
        pages,
        index=pages.index(st.session_state.page)
    )

    st.session_state.page = selected_page

page = st.session_state.page


# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown(
        "<div class='main-title'>🩺 OBSERA Health</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='sub-title'>AI Powered Maternal Risk Prediction System</div>",
        unsafe_allow_html=True
    )

    st.write("")
    st.write("")

    # card arrow Button #
    st.markdown("""
    <style>
        div.stButton > button {
        background: linear-gradient(135deg,#10B981,#2DD4BF);
        color:white;
        border:none;
        border-radius:50%;
        height:50px;
        width:50px;
        min-width:50px;
        font-size:22px;
        font-weight:bold;
        cursor:pointer;
        transition:0.3s;
        margin-top:-90px;
        margin-right:15px;
        float:right;
        box-shadow:0 0 20px rgba(16,185,129,0.25);
    }

        div.stButton > button:hover {
            transform: scale(1.1);
            box-shadow:0 0 30px rgba(45,212,191,0.4);
        }
      </style>
    """, unsafe_allow_html=True)
    

    col1, col2 = st.columns(2)

    # ---------------- PREDICTION CARD ----------------
    with col1:

        st.markdown("""
        <div class='card'>
            <h3>📊 Risk Prediction</h3>
            <p>Predict maternal health risk instantly using AI.</p>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns([8,1])

        with col_btn2:
            if st.button("➜", key="prediction_card"):
                st.session_state.page = "Prediction"
                st.rerun()

    # ---------------- COMPARISON CARD ----------------
    with col2:

        st.markdown("""
        <div class='card'>
            <h3>📈 State Comparison</h3>
            <p>Compare maternal health indicators between states.</p>
        </div>
        """, unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns([8,1])

        with col_btn2:
            if st.button("➜", key="comparison_card"):
                st.session_state.page = "Comparison"
                st.rerun()

    # ---------------- ABOUT ----------------
    st.markdown("""
    <div class='card'>
    <h3>📌 About the Project</h3>
    <p>
    This AI-based maternal health dashboard predicts maternal risk levels
    using machine learning and provides explainable insights using SHAP values.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
elif page == "Prediction":

    st.markdown(
        "<div class='main-title'>📊 Risk Prediction</div>",
        unsafe_allow_html=True
    )
    if st.button("⬅ Back", key="pred_back"):
        st.session_state.page = "Dashboard"
        st.session_state.navigation_radio = "Dashboard"
        st.rerun()

    st.write("")

    # ---------------- FILTERING ----------------
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

    # ---------------- PREDICT ----------------
    if st.button("Predict Risk"):

        with st.spinner("Analyzing maternal health data..."):

            # ---------------- FILTER ----------------
            if "_" in district:

                filtered = df[
                    (df['State'] == state) &
                    (df['is_state_row'])
                ]

                label = "State Risk"

            else:

                filtered = df[
                    (df['State'] == state) &
                    (df['District'] == district) &
                    (~df['is_state_row'])
                ]

                label = "District Risk"

            if filtered.empty:

                st.error("No data available")

            else:

                sample = filtered.head(1)

                # ---------------- PREPROCESS ----------------
                X_df = preprocess_input(sample[features])

                X_imputed = imputer.transform(X_df)

                X_scaled = scaler.transform(X_imputed)

                # ---------------- PREDICT ----------------
                pred = model.predict(X_scaled)

                result = le.inverse_transform(pred)[0]

                # ---------------- METRICS ----------------
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h4>State</h4>
                        <h2>{state}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:

                    district_display = district

                    if "_" in district:
                        district_display = "N/A"

                    st.markdown(f"""
                    <div class='metric-box'>
                        <h4>District</h4>
                        <h2>{district_display}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class='metric-box'>
                        <h4>Risk Level</h4>
                        <h2>{result}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # ---------------- RESULT CARD ----------------
                if result == "High":

                    bg = "linear-gradient(135deg,#EF4444,#7F1D1D)"
                    emoji = "🚨"
                    msg = "Immediate medical attention required."

                elif result == "Medium":

                    bg = "linear-gradient(135deg,#F59E0B,#78350F)"
                    emoji = "⚠️"
                    msg = "Regular monitoring is recommended."

                else:

                    bg = "linear-gradient(135deg,#10B981,#064E3B)"
                    emoji = "✅"
                    msg = "Healthy condition detected."

                st.markdown(f"""
                <div class='result-card' style='background:{bg};'>
                    <h1>{emoji} {label}: {result}</h1>
                    <p>{msg}</p>
                </div>
                """, unsafe_allow_html=True)

                # ---------------- SHAP ----------------
                st.write("")
                st.markdown("## 🧠 Why this prediction?")

                X_processed_df = pd.DataFrame(
                    X_scaled,
                    columns=features
                )

                shap_values = explainer.shap_values(X_processed_df)

                if isinstance(shap_values, list):

                    shap_val = shap_values[pred[0]][0]

                else:

                    shap_val = shap_values[0]

                shap_val = np.array(shap_val).flatten()

                min_len = min(len(features), len(shap_val))

                shap_df = pd.DataFrame({
                    "Feature": features[:min_len],
                    "Impact": shap_val[:min_len]
                })

                shap_df["abs"] = np.abs(shap_df["Impact"])

                shap_df = shap_df.sort_values(
                    by="abs",
                    ascending=False
                ).head(8)

                # ---------------- GRAPH ----------------
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=shap_df["Impact"],
                    y=shap_df["Feature"],
                    orientation='h',
                    marker=dict(color="#10B981")
                ))

                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="white"),
                    title={
                        'text': "Top Factors Affecting Prediction",
                        'x': 0.5
                    },
                    height=500,
                    margin=dict(l=20, r=20, t=60, b=20),
                    yaxis=dict(
                        autorange="reversed",
                        showgrid=False
                    ),
                    xaxis=dict(showgrid=False)
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

                # ---------------- REASONS ----------------
                st.markdown("## 📌 Key Reasons")

                for _, row in shap_df.iterrows():

                    direction = (
                        "increased"
                        if row["Impact"] > 0
                        else "decreased"
                    )

                    st.markdown(f"""
                    <div class='card'>
                    • <b>{row['Feature']}</b> {direction} the risk
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")

# ---------------- COMPARISON ----------------
elif page == "Comparison":

    st.markdown(
        "<div class='main-title'>📈 State Comparison</div>",
        unsafe_allow_html=True
    )
    if st.button("⬅ Back", key="comp_back"):
        st.session_state.page = "Dashboard"
        st.session_state.navigation_radio = "Dashboard"
        st.rerun()
    st.write("")         

    # ---------------- NUMERIC DATA ----------------
    df_numeric = df.copy()

    for col in df_numeric.columns:

        if col not in ['State', 'District']:

            df_numeric[col] = pd.to_numeric(
                df_numeric[col]
                .astype(str)
                .str.replace(',', ''),
                errors='coerce'
            )

    df_numeric = df_numeric.fillna(
        df_numeric.median(numeric_only=True)
    )

    df_compare = df_numeric.groupby(
        'State'
    ).median(numeric_only=True)

    states = df_compare.index.tolist()

    col1, col2 = st.columns(2)

    with col1:
        state1 = st.selectbox("Select State 1", states)

    with col2:
        state2 = st.selectbox("Select State 2", states)

    # ---------------- COMPARE ----------------
    if st.button("Compare States"):

        if state1 == state2:

            st.error("Please select different states.")

        else:

            data1 = df_compare.loc[state1]

            data2 = df_compare.loc[state2]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=data1.values,
                mode='lines+markers',
                name=state1
            ))

            fig.add_trace(go.Scatter(
                y=data2.values,
                mode='lines+markers',
                name=state2
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#111827",
                plot_bgcolor="#111827",
                title={
                    'text': "State Health Indicator Comparison",
                    'x': 0.5
                },
                xaxis_title="Indicators",
                yaxis_title="Values",
                font=dict(size=14),
                height=600
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False}
            )

            # ---------------- SUMMARY ----------------
            st.write("")

            st.markdown(f"""
            <div class='card'>
            <h3>📌 Comparison Summary</h3>
            <p>
            Comparing <b>{state1}</b> and <b>{state2}</b>
            based on maternal health indicators.
            </p>
            </div>
            """, unsafe_allow_html=True)
