# app.py
import streamlit as st
from engine import authenticate_user, get_user_context

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Employee Performance AI",
    layout="centered"
)

# --------------------------------------------------
# SESSION STATE INIT
# --------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.role = None

# --------------------------------------------------
# APP TITLE
# --------------------------------------------------
st.title("Employee Performance AI")
st.caption("Secure, explainable performance insights powered by data and AI")

st.divider()

# --------------------------------------------------
# LOGIN VIEW
# --------------------------------------------------
if not st.session_state.authenticated:
    st.subheader("Login")

    user_id = st.text_input("User ID", placeholder="EMP-1, MGR-001, ADM-01")
    password = st.text_input("Password", type="password")

    login_btn = st.button("Login")

    if login_btn:
        role = authenticate_user(user_id.strip().upper(), password)

        if role is None:
            st.error("Invalid user ID or password.")
        else:
            ctx = get_user_context(user_id)
            st.session_state.authenticated = True
            st.session_state.user_id = user_id.strip().upper()
            st.session_state.role = ctx["role"]
            st.success(f"Login successful ({ctx['role'].capitalize()})")
            st.rerun()

    st.stop()

# --------------------------------------------------
# ROLE ROUTING (NO ROLE SELECTION)
# --------------------------------------------------
role = st.session_state.role
user_id = st.session_state.user_id

st.success(f"Logged in as: {user_id} ({role})")
st.divider()

# --------------------------------------------------
# EMPLOYEE VIEW
# --------------------------------------------------
# --------------------------------------------------
# EMPLOYEE VIEW
# --------------------------------------------------
if role == "employee":
    from engine import (
        standard_df,
        build_llm_payload,
        llm_rewrite,
        generate_employee_explanation,
        answer_employee_question
    )

    st.header("Your Performance Overview")

    # Extract employee row safely
    emp_id = int(user_id.replace("EMP-", ""))
    emp_row = standard_df[standard_df["employee_id"] == emp_id]

    if emp_row.empty:
        st.error("Your performance record could not be found.")
        st.stop()

    emp = emp_row.iloc[0]
    explanation = generate_employee_explanation(emp)

    # --- Summary ---
    st.subheader("Status")
    st.write(f"**{explanation['status']}**")

    st.subheader("What happened")
    st.write(explanation["summary"])

    # --- Drivers ---
    st.subheader("Why this happened")
    if explanation["drivers"]:
        for d in explanation["drivers"]:
            st.write(f"- {d}")
    else:
        st.write("No major issues identified.")

    # --- Strengths ---
    st.subheader("Your strengths")
    if explanation["strengths"]:
        for s in explanation["strengths"]:
            st.write(f"- {s}")
    else:
        st.write("No specific strengths identified in this period.")

    # --- Actions ---
    st.subheader("What to focus on next")
    if explanation["recommended_actions"]:
        for i, a in enumerate(explanation["recommended_actions"], 1):
            st.write(f"{i}. {a}")
    else:
        st.write("Keep maintaining your current performance.")

    st.divider()

    # --------------------------------------------------
    # EMPLOYEE AI COACH
    # --------------------------------------------------
    st.subheader("Ask your AI performance coach")

    st.caption(
        "You can ask things like:\n"
        "- Why was my performance low?\n"
        "- What should I focus on next month?\n"
        "- What am I doing well?"
    )

    question = st.text_input("Your question")

    if st.button("Ask AI"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            payload = build_llm_payload(emp, user_question=question)

            with st.spinner("Thinking..."):
                response = llm_rewrite(payload)

            st.markdown("### AI Feedback")
            st.write(response)

# --------------------------------------------------
# MANAGER VIEW (placeholder)
# --------------------------------------------------
elif role == "manager":
    st.header("Manager Portal")
    st.info("This area shows team-level insights and coaching tools.")
    st.write("✅ Secure access confirmed.")
    st.write("➡️ Next: team summary, rankings, coaching AI")

# --------------------------------------------------
# ADMIN VIEW (placeholder)
# --------------------------------------------------
elif role == "admin":
    st.header("Admin Portal")
    st.info("This area is for system and user management.")
    st.write("✅ Secure access confirmed.")
    st.write("➡️ Next: add/remove users, reset passwords")

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------
st.divider()
if st.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.role = None
    st.rerun()
