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

st.set_page_config(page_title="Employee Performance Intelligence", layout="wide")
st.title("Employee Performance Intelligence System")

role = st.radio("Select your role:", ["Employee", "Manager"])

# =========================
# EMPLOYEE VIEW
# =========================
if role == "Employee":
    st.header("Employee Portal")

    emp_id = st.number_input("Enter your Employee ID", min_value=1, step=1)

    if emp_id:
        emp_row = standard_df[standard_df["employee_id"] == emp_id]

        if emp_row.empty:
            st.error("Employee not found.")
        else:
            emp = emp_row.iloc[0]

            st.metric("Status", emp["performance_status"])

            chart_df = pd.DataFrame({
                "Metric": ["Output", "Quality", "Development"],
                "Score": [
                    emp["output_score"],
                    emp["quality_score"],
                    emp["development_score"]
                ]
            })
            st.bar_chart(chart_df.set_index("Metric"))

            question = st.text_input("Ask about your performance")
            if question:
                st.write(route_request(f"EMP-{emp_id}", question))

# =========================
# MANAGER VIEW
# =========================
if role == "Manager":
    st.header("Manager Portal")

    insights = manager_team_insights(standard_df)
    st.subheader("Team Insights")
    st.json(insights)

    st.subheader("Performance Distribution")
    st.bar_chart(pd.Series(insights["status_distribution"]))

    st.subheader("Private Ranking")
    st.dataframe(manager_private_ranking(standard_df))

    st.subheader("Review Individual Employee")
    selected_emp = st.selectbox(
        "Select employee",
        standard_df["employee_id"].tolist()
    )

    emp = standard_df[standard_df["employee_id"] == selected_emp].iloc[0]
    st.metric("Status", emp["performance_status"])
    st.write("Drivers:", emp["performance_drivers"])

    if st.button("Generate Coaching Guidance"):
        payload = build_llm_payload(emp, "How should I support this employee?")
        st.write(llm_rewrite(payload))

    st.subheader("AI Team Summary")
    if st.button("Generate AI Summary"):
        st.write(manager_ai_summary(standard_df))
