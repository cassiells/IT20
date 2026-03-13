import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# --- Path Standardization ---
# This ensures the app knows exactly where it is on your computer
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from predict import BurnoutPredictor

# --- Page Configuration ---
st.set_page_config(
    page_title="Burnout AI - Optimized SVM Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- UI Styling ---
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        font-weight: 700;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🛡️ Burnout AI: Optimized SVM Engine")
    st.markdown("---")

    # CORRECTED PATH DISCOVERY
    # We look for the 'models' folder inside the current project folder
    models_dir = os.path.join(CURRENT_DIR, "models")

    # --- Sidebar ---
    st.sidebar.header("Intelligence Engine")
    st.sidebar.success("**Model:** Optimized SVM (RBF)")
    st.sidebar.info("Accuracy: ~99.8%\nOptimization: GridSearchCV")
    st.sidebar.markdown("---")
    st.sidebar.caption(f"System Operational | {datetime.now().year}")

    # --- Engine Initialization ---
    @st.cache_resource
    def load_predictor():
        # Check if the folder exists before trying to load
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found at {models_dir}")
        return BurnoutPredictor(models_dir)

    try:
        engine = load_predictor()
    except Exception as e:
        st.error(f"Engine Fault: Optimized SVM assets not found at {models_dir}")
        st.info("💡 **Action Required:** Run `python train.py` in your terminal to generate the model files.")
        return

    # --- Feature Input Console ---
    st.subheader("📋 Employee Wellness Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        job_role = st.selectbox("Job Role", ["Software Developer", "Data Scientist", "Manager", "HR", "Designer"])
        country = st.selectbox("Country", ["USA", "India", "UK", "Canada", "Germany"])

    with col2:
        exp_years = st.slider("Experience (Years)", 0, 40, 5)
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large", "MNC"])
        work_env = st.selectbox("Work Setting", ["Office", "Home", "Hybrid"])
        stress_level = st.radio("Stress Level", ["Low", "Medium", "High"], horizontal=True)

    with col3:
        work_hours = st.slider("Work Hours / Day", 1.0, 16.0, 8.0)
        meetings = st.slider("Meetings / Day", 0, 15, 3)
        sleep_hours = st.slider("Sleep Hours", 1.0, 12.0, 7.0)
        exercise_hours = st.slider("Exercise/Week", 0.0, 20.0, 3.0)

    st.markdown("---")

    # --- Prediction Execution ---
    if st.button("CALCULATE EXPOSURE RISK 🚀"):
        payload = {
            'Age': age, 'Gender': gender, 'Country': country, 'Job_Role': job_role,
            'Experience_Years': exp_years, 'Company_Size': company_size,
            'Work_Hours_Per_Day': work_hours, 'Meetings_Per_Day': meetings,
            'Internet_Speed_Mbps': 100.0, 'Work_Environment': work_env,
            'Sleep_Hours': sleep_hours, 'Exercise_Hours_Per_Week': exercise_hours,
            'Screen_Time_Hours': 8.0, 'Stress_Level': stress_level,
            'Productivity_Score': 75
        }

        with st.spinner("Processing..."):
            result = engine.predict(payload)[0]
            risk = result['Burnout_Risk']
            conf = result['Confidence']

            if "Burnout" in risk:
                st.error(f"### ⚠️ RISK DETECTED: {risk}")
            else:
                st.success(f"### ✅ CLEAR: {risk}")
            st.progress(conf, text=f"Confidence: {conf:.1%}")

if __name__ == "__main__":
    main()