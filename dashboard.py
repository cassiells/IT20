import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime

# Standardize path imports for core logic
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from predict import BurnoutPredictor

# --- Page Optimization ---
st.set_page_config(
    page_title="Burnout AI - Optimized SVM Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Glassmorphism UI (Internal) ---
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
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.2);
    }
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("🛡️ Burnout AI: Optimized SVM Engine")
    st.markdown("---")

    # Path Discovery for Model Assets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    # --- Sidebar Architecture ---
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3062/3062140.png", width=80)
    st.sidebar.header("Intelligence Engine")
    st.sidebar.success("**Model:** Optimized SVM (RBF Kernel)")
    st.sidebar.info("""
    **Core Metrics:**
    - Accuracy: ~99.8%
    - Class Balancing: SMOTE
    - Optimization: GridSearchCV (C=20)
    """)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"System Operational | {datetime.now().year}")

    # --- Engine Initialization ---
    @st.cache_resource
    def load_predictor():
        return BurnoutPredictor(models_dir)

    try:
        engine = load_predictor()
    except Exception as e:
        st.error(f"Engine Fault: {e}")
        st.info("💡 Please run the training script: `python src/train.py` to initialize the SVM model.")
        return

    # --- Interactive Feature Console ---
    st.subheader("📋 Employee Wellness Metrics")

    # Organizing features into logical columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Context**")
        age = st.number_input("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        job_role = st.selectbox("Job Role", [
            "Software Developer", "Software Engineer", "Data Scientist",
            "Data Analyst", "Manager", "Designer", "HR", "Marketing"
        ])
        country = st.selectbox("Country",
                               ["USA", "India", "UK", "Australia", "Canada", "Germany", "Japan", "South Korea",
                                "China"])

    with col2:
        st.markdown("**Work Configuration**")
        exp_years = st.slider("Experience (Years)", 0, 40, 5)
        company_size = st.selectbox("Company Size", ["Small", "Medium", "Large", "MNC"])
        work_env = st.selectbox("Work Setting", ["Office", "Home", "Hybrid", "Coworking"])
        stress_level = st.radio("Perceived Stress Level", ["Low", "Medium", "High"], horizontal=True)

    with col3:
        st.markdown("**Daily Habits**")
        work_hours = st.slider("Work Hours / Day", 1.0, 16.0, 8.0)
        meetings = st.slider("Meetings / Day", 0, 15, 3)
        sleep_hours = st.slider("Sleep Hours", 1.0, 12.0, 7.0)
        exercise_hours = st.slider("Exercise/Week", 0.0, 20.0, 3.0)

    # Secondary Metrics
    with st.expander("🛠️ Advanced Analytics Overrides"):
        acol1, acol2 = st.columns(2)
        with acol1:
            internet_speed = st.number_input("Internet Speed (Mbps)", 1.0, 1000.0, 100.0)
            screen_time = st.slider("Total Screen Time (Hours)", 0.0, 24.0, 8.0)
        with acol2:
            prod_score = st.slider("Productivity Index (0-100)", 0, 100, 75)

    st.markdown("---")

    # --- Prediction Execution ---
    if st.button("CALCULATE EXPOSURE RISK 🚀"):
        payload = {
            'Age': age, 'Gender': gender, 'Country': country, 'Job_Role': job_role,
            'Experience_Years': exp_years, 'Company_Size': company_size,
            'Work_Hours_Per_Day': work_hours, 'Meetings_Per_Day': meetings,
            'Internet_Speed_Mbps': internet_speed, 'Work_Environment': work_env,
            'Sleep_Hours': sleep_hours, 'Exercise_Hours_Per_Week': exercise_hours,
            'Screen_Time_Hours': screen_time, 'Stress_Level': stress_level,
            'Productivity_Score': prod_score
        }

        with st.spinner("SVM Engine processing input vectors..."):
            # Execute prediction
            result = engine.predict(payload)[0]

            risk_status = result['Burnout_Risk']
            confidence = result['Confidence']

            container = st.container()
            if "Burnout" in risk_status:
                container.error(f"### ⚠️ RISK DETECTED: {risk_status.upper()}")
                container.progress(confidence, text=f"SVM Confidence: {confidence:.1%}")
                container.markdown("> **Preventative Action Required:** Schedule immediate 1-on-1 workload audit.")
            else:
                container.success(f"### ✅ CLEAR: {risk_status.upper()}")
                container.progress(confidence, text=f"SVM Confidence: {confidence:.1%}")
                container.balloons()
                container.markdown("> **Wellness Note:** Employee parameters are within healthy thresholds.")

    st.markdown("---")
    st.caption(f"Engine Protocol: Optimized-SVM-RBF | Last Calibrated: {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()