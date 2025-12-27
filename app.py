import streamlit as st
import pandas as pd

from engine import (
    build_llm_payload,
    llm_rewrite,
    authenticate_user,
    admin_create_user,
    admin_remove_user,
    admin_reset_password,
    admin_list_users,
    resolve_user_data_view,
    standard_df,
    manager_ai_summary,
    answer_employee_question,
)


# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Employee Performance AI",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------
if "auth" not in st.session_state:
    st.session_state.auth = None

def logout():
    st.session_state.auth = None
    st.rerun()

# -----------------------------------------------------
# LOGIN PAGE
# -----------------------------------------------------
def login_page():
    st.title("📊 Employee Performance AI")
    st.subheader("Secure Login")

    user_id = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        auth = authenticate_user(user_id, password)
        if auth is None:
            st.error("Invalid credentials")
        else:
            st.session_state.auth = auth
            st.success(f"Logged in as {auth['role'].capitalize()}")
            st.rerun()

# -----------------------------------------------------
# ADMIN DASHBOARD
# -----------------------------------------------------
def admin_dashboard(auth):
    st.sidebar.title("Admin")
    if st.sidebar.button("Logout"):
        logout()

    st.title("🛠️ Admin Dashboard")

    tabs = st.tabs(["Create User", "User Registry", "Remove User", "Reset Password", "View Dataset"])

    # -------------------------------
    # CREATE USER
    # -------------------------------
    with tabs[0]:
        st.subheader("Create User")

        role = st.selectbox("Role", ["employee", "manager"])
        new_user_id = st.text_input("User ID (e.g. EMP-JOHN, MGR-SARAH)")
        password = st.text_input("Password", type="password")

        data_employee_id = None
        scope_type = None
        scope_value = None

        if role == "employee":
            st.markdown("**Link employee to dataset**")
            data_employee_id = st.selectbox(
                "Employee ID (from data)",
                sorted(standard_df["employee_id"].astype(str).unique())
            )

        if role == "manager":
            st.markdown("**Manager Scope**")
            scope_type = st.selectbox("Scope Type", ["department", "employee_ids"])

            if scope_type == "department":
                scope_value = st.selectbox(
                    "Department",
                    sorted(standard_df["sector"].dropna().unique())
                )
            else:
                scope_value = st.multiselect(
                    "Employee IDs",
                    sorted(standard_df["employee_id"].astype(str).unique())
                )

        if st.button("Create User"):
            result = admin_create_user(
                user_id=new_user_id,
                role=role,
                password=password,
                data_employee_id=data_employee_id,
                data_scope_type=scope_type,
                data_scope_value=scope_value,
            )

            if result["success"]:
                st.success("User created successfully")
            else:
                st.error(result["error"])
    # -------------------------------
    # USER REGISTRY
    # -------------------------------
    with tabs[1]:
        st.subheader("User Registry")

        users_view = admin_list_users()

        if users_view.empty:
            st.info("No users in the system.")
        else:
            st.dataframe(users_view, use_container_width=True)

    # -------------------------------
    # REMOVE USER
    # -------------------------------
    with tabs[1]:
        st.subheader("Remove User")
        uid = st.text_input("User ID to remove")
        if st.button("Remove"):
            res = admin_remove_user(uid)
            if res["success"]:
                st.success("User removed")
            else:
                st.error(res["error"])

    # -------------------------------
    # RESET PASSWORD
    # -------------------------------
    with tabs[2]:
        st.subheader("Reset Password")
        uid = st.text_input("User ID")
        new_pw = st.text_input("New Password", type="password")
        if st.button("Reset"):
            res = admin_reset_password(uid, new_pw)
            if res["success"]:
                st.success("Password reset")
            else:
                st.error(res["error"])

    # -------------------------------
    # VIEW DATA
    # -------------------------------
    with tabs[3]:
        st.subheader("Dataset Preview")
        st.dataframe(standard_df.head(50))


# -----------------------------------------------------
# EMPLOYEE DASHBOARD
# -----------------------------------------------------
def employee_dashboard(auth):
    st.sidebar.title("Employee")
    if st.sidebar.button("Logout"):
        logout()

    st.title("👤 My Performance")

    df_view = resolve_user_data_view(auth, standard_df)

    if df_view.empty:
        st.warning("You are not yet assigned to any performance data.")
        return

    row = df_view.iloc[0]

    st.metric("Performance Status", row["performance_status"])
    st.metric("Output Score", round(row["output_score"], 2))
    st.metric("Quality Score", row["quality_score"])

    st.subheader("Ask your AI coach")
    q = st.text_input("Ask a question (e.g. How can I improve?)")

    if q:
        payload = build_llm_payload(row, user_question=q)

    with st.spinner("Generating personalised feedback..."):
        ai_text = llm_rewrite(payload)

    st.markdown(ai_text)


# -----------------------------------------------------
# MANAGER DASHBOARD
# -----------------------------------------------------
def manager_dashboard(auth):
    st.sidebar.title("Manager")
    if st.sidebar.button("Logout"):
        logout()

    st.title("👥 Manager Overview")

    df_view = resolve_user_data_view(auth, standard_df)

    if df_view.empty:
        st.warning("No team data assigned.")
        return

    st.subheader("Team Snapshot")
    st.dataframe(df_view[["employee_id", "performance_status", "output_score"]].head(50))

    st.subheader("AI Team Summary")
    with st.spinner("Generating summary..."):
        summary = manager_ai_summary(df_view)
    st.write(summary)


# -----------------------------------------------------
# ROUTER
# -----------------------------------------------------
def main():
    if st.session_state.auth is None:
        login_page()
        return

    role = st.session_state.auth["role"]

    if role == "admin":
        admin_dashboard(st.session_state.auth)
    elif role == "employee":
        employee_dashboard(st.session_state.auth)
    elif role == "manager":
        manager_dashboard(st.session_state.auth)
    else:
        st.error("Unknown role")
        logout()

main()
