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
# EMPLOYEE VIEW (placeholder)
# --------------------------------------------------
if role == "employee":
    st.header("Employee Portal")
    st.info("This area shows your personal performance insights only.")
    st.write("✅ Secure access confirmed.")
    st.write("➡️ Next: performance summary, AI coach, charts")

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
