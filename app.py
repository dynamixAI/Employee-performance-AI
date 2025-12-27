# app.py
import streamlit as st
import pandas as pd

from engine import (
    standard_df,
    route_request,
    manager_team_insights,
    manager_private_ranking,
    manager_ai_summary,
    build_llm_payload,
    llm_rewrite
)

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Employee Performance Intelligence",
    layout="wide"
)

st.title("Employee Performance Intelligence System")
st.caption(
    "Demo loads with sample data. Upload your own CSV to test with your data."
)

# =========================
# SIDEBAR — DATA SOURCE
# =========================
st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (optional)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using uploaded data")
else:
    df = standard_df.copy()
    st.sidebar.info("Using demo data")

# =========================
# BASIC VALIDATION
# =========================
REQUIRED_COLUMNS = [
    "employee_id",
    "output_score",
    "quality_score",
    "development_score",
    "performance_status"
]

missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

# =========================
# AUTHENTICATION MOCK
# =========================
st.sidebar.header("Login")

role = st.sidebar.radio(
    "Select role",
    ["Employee", "Manager"]
)

# =========================
# EMPLOYEE VIEW
# =========================
if role == "Employee":
    st.header("Employee Portal")

    emp_id = st.number_input(
        "Enter your Employee ID",
        min_value=1,
        step=1
    )

    emp_row = df[df["employee_id"] == emp_id]

    if emp_id and emp_row.empty:
        st.error("Employee not found.")

    if not emp_row.empty:
        emp = emp_row.iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Status", emp["performance_status"])
        col2.metric("Output Score", round(emp["output_score"], 2))
        col3.metric("Quality Score", round(emp["quality_score"], 2))

        # -------------------------
        # PERFORMANCE BREAKDOWN
        # -------------------------
        st.subheader("Performance Breakdown")

        breakdown_df = pd.DataFrame({
            "Metric": ["Output", "Quality", "Development"],
            "Score": [
                emp["output_score"],
                emp["quality_score"],
                emp["development_score"]
            ]
        })

        st.bar_chart(breakdown_df.set_index("Metric"))

        # -------------------------
        # TIME TREND (IF AVAILABLE)
        # -------------------------
        if "date" in df.columns:
            st.subheader("Performance Over Time")

            trend_df = df[df["employee_id"] == emp_id]
            trend_df["date"] = pd.to_datetime(trend_df["date"])

            trend_df = trend_df.sort_values("date")

            st.line_chart(
                trend_df.set_index("date")["output_score"]
            )

        # -------------------------
        # AI CHAT
        # -------------------------
        st.subheader("Ask about your performance")

        question = st.text_input(
            "Ask a question (e.g. How can I improve next month?)"
        )

        if question:
            response = route_request(
                user_id=f"EMP-{emp_id}",
                question=question
            )
            st.write(response)

# =========================
# MANAGER VIEW
# =========================
if role == "Manager":
    st.header("Manager Portal")

    # -------------------------
    # TEAM INSIGHTS
    # -------------------------
    st.subheader("Team Overview")

    insights = manager_team_insights(df)

    col1, col2 = st.columns(2)
    col1.metric("Team Size", insights["team_size"])
    col2.metric("Average Output Score", insights["avg_output_score"])

    st.bar_chart(
        pd.Series(insights["status_distribution"])
    )

    # -------------------------
    # TIME TREND (IF AVAILABLE)
    # -------------------------
    if "date" in df.columns:
        st.subheader("Team Output Trend")

        df["date"] = pd.to_datetime(df["date"])
        trend_df = (
            df.groupby("date")["output_score"]
            .mean()
            .sort_index()
        )

        st.line_chart(trend_df)

    # -------------------------
    # PRIVATE RANKING
    # -------------------------
    st.subheader("Private Performance Ranking")
    st.dataframe(
        manager_private_ranking(df),
        use_container_width=True
    )

    # -------------------------
    # EMPLOYEE DRILL-DOWN
    # -------------------------
    st.subheader("Review Individual Employee")

    selected_emp = st.selectbox(
        "Select employee ID",
        df["employee_id"].unique()
    )

    emp = df[df["employee_id"] == selected_emp].iloc[0]

    st.metric("Status", emp["performance_status"])
    st.write("Performance Drivers:", emp.get("performance_drivers", []))

    if st.button("Generate Coaching Guidance"):
        payload = build_llm_payload(
            emp,
            user_question="How should I support this employee?"
        )
        st.write(llm_rewrite(payload))

    # -------------------------
    # AI TEAM SUMMARY
    # -------------------------
    st.subheader("AI Team Summary")

    if st.button("Generate AI Summary"):
        st.write(manager_ai_summary(df))
