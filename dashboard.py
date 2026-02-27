import streamlit as st
import pandas as pd
from deployment.api import run_search

st.set_page_config(page_title="Automated Model Architect", layout="wide")

st.title("🏆 Automated Model Architect")
st.markdown("Neural Architecture Search Research Dashboard")

# ---- SESSION STATE INIT ----
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ---- SIDEBAR ----
st.sidebar.header("⚙️ Experiment Settings")
trials = st.sidebar.slider("Search Trials", 3, 30, 5)

if st.sidebar.button("🚀 Run Architecture Search"):

    with st.spinner("Running Neural Architecture Search..."):
        result = run_search(trials)

    trials_df = pd.DataFrame(result["all_trials"])

    # Append to session history
    st.session_state.history = pd.concat(
        [st.session_state.history, trials_df],
        ignore_index=True
    )

    st.success("Search Complete!")

# ---- DISPLAY SECTION ----
if not st.session_state.history.empty:

    df = st.session_state.history.copy()

    # Leaderboard
    st.subheader("🏆 Leaderboard (All Trials)")
    leaderboard = df.sort_values("accuracy", ascending=False)
    st.dataframe(leaderboard, use_container_width=True)

    # Best model summary
    best_model = leaderboard.iloc[0]
    st.subheader("🥇 Best Configuration So Far")
    st.json(best_model.to_dict())

    # ---- ACCURACY TREND ----
    st.subheader("📈 Accuracy Over Trials")

    df["cumulative_best"] = df["accuracy"].cummax()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Trial Accuracy**")
        st.line_chart(df["accuracy"])

    with col2:
        st.markdown("**Best-So-Far Curve (Research View)**")
        st.line_chart(df["cumulative_best"])