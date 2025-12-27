# engine.py
import os
import json
import re
import hashlib
import pandas as pd
from typing import Any, Dict, List, Optional

# =========================
# 1) LOAD DEMO DATASET
# =========================
DATA_PATH = "standard_df.csv"

if os.path.exists(DATA_PATH):
    standard_df = pd.read_csv(DATA_PATH)
else:
    standard_df = pd.DataFrame()

# =========================
# 2) AUTH & PASSWORD SECURITY
# =========================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash


PASSWORD_POLICY = {
    "min_length": 8,
    "require_number": True,
    "require_special": True,
}

def validate_password(password: str) -> List[str]:
    errors = []

    if len(password) < PASSWORD_POLICY["min_length"]:
        errors.append(f"Password must be at least {PASSWORD_POLICY['min_length']} characters.")

    if PASSWORD_POLICY["require_number"] and not re.search(r"\d", password):
        errors.append("Password must contain at least one number.")

    if PASSWORD_POLICY["require_special"] and not re.search(r"[!@#$%^&*]", password):
        errors.append("Password must contain at least one special character (!@#$%^&*).")

    return errors


# =========================
# 3) USER REGISTRY (DEMO BACKEND)
# =========================
users_df = pd.DataFrame(columns=["user_id", "role", "password_hash"])

def _seed_users():
    global users_df
    if users_df.empty:
        users_df = pd.DataFrame([
            {"user_id": "ADM-01", "role": "admin", "password_hash": hash_password("Admin@123")},
            {"user_id": "MGR-001", "role": "manager", "password_hash": hash_password("Manager@123")},
            {"user_id": "EMP-1", "role": "employee", "password_hash": hash_password("Employee@123")},
        ])

_seed_users()


def authenticate_user(user_id: str, password: str) -> Optional[str]:
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        return None

    if not verify_password(password, row.iloc[0]["password_hash"]):
        return None

    return row.iloc[0]["role"]


def create_user(user_id: str, role: str, password: str) -> Dict[str, Any]:
    global users_df

    if role not in ["employee", "manager", "admin"]:
        return {"success": False, "error": "Invalid role."}

    if user_id in users_df["user_id"].values:
        return {"success": False, "error": "User already exists."}

    password_errors = validate_password(password)
    if password_errors:
        return {"success": False, "error": password_errors}

    users_df = pd.concat(
        [users_df, pd.DataFrame([{
            "user_id": user_id,
            "role": role,
            "password_hash": hash_password(password)
        }])],
        ignore_index=True
    )

    return {"success": True}


def remove_user(user_id: str) -> Dict[str, Any]:
    global users_df

    if user_id not in users_df["user_id"].values:
        return {"success": False, "error": "User not found."}

    users_df = users_df[users_df["user_id"] != user_id]
    return {"success": True}


def reset_password(user_id: str, new_password: str) -> Dict[str, Any]:
    password_errors = validate_password(new_password)
    if password_errors:
        return {"success": False, "error": password_errors}

    idx = users_df.index[users_df["user_id"] == user_id]
    if len(idx) == 0:
        return {"success": False, "error": "User not found."}

    users_df.loc[idx, "password_hash"] = hash_password(new_password)
    return {"success": True}


# =========================
# 4) OPENROUTER CLIENT (SAFE)
# =========================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client = None

if OpenAI is not None and OPENROUTER_API_KEY:
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


# =========================
# 5) UTILITIES
# =========================
REQUIRED_COLUMNS = [
    "employee_id",
    "output_score",
    "quality_score",
    "development_score",
    "performance_status",
]

def require_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def clean_ai_text(text: str) -> str:
    for token in ["[B_INST]", "[/B_INST]", "<s>", "</s>", "[OUT]", "[/OUT]"]:
        text = text.replace(token, "")
    return text.strip()


# =========================
# 6) PERFORMANCE LOGIC (UNCHANGED)
# =========================
DRIVER_ACTIONS = {
    "Lower productivity relative to workload":
        "Review workload planning and focus on prioritising high-impact tasks.",
    "Quality score below expected level":
        "Focus on accuracy and seek feedback or refresher training.",
    "High number of sick days affecting consistency":
        "Discuss workload balance or wellbeing support with your manager.",
    "Limited recent training or development activity":
        "Complete relevant training to strengthen performance.",
    "No major performance risk factors identified":
        "Maintain current working practices.",
}

def identify_strengths(row: pd.Series) -> List[str]:
    strengths = []
    if float(row.get("output_score", 0)) >= 50:
        strengths.append("Good productivity level")
    if float(row.get("quality_score", 0)) >= 4:
        strengths.append("Strong quality of work")
    if float(row.get("Sick_Days", 0)) <= 5:
        strengths.append("Good attendance consistency")
    return strengths or ["Maintaining baseline performance"]

def identify_performance_drivers(row: pd.Series) -> List[str]:
    drivers = []
    if float(row.get("output_score", 0)) < 50:
        drivers.append("Lower productivity relative to workload")
    if float(row.get("quality_score", 0)) < 3:
        drivers.append("Quality score below expected level")
    if float(row.get("Sick_Days", 0)) > 10:
        drivers.append("High number of sick days affecting consistency")
    if float(row.get("development_score", 0)) < 5:
        drivers.append("Limited recent training or development activity")
    return drivers or ["No major performance risk factors identified"]


# =========================
# 7) AI EXPLANATION
# =========================
def build_llm_payload(row: pd.Series, user_question: Optional[str] = None) -> Dict[str, Any]:
    drivers = identify_performance_drivers(row)
    strengths = identify_strengths(row)

    return {
        "employee_id": int(row.get("employee_id")),
        "performance_status": row.get("performance_status"),
        "metrics": {
            "output_score": float(row.get("output_score", 0)),
            "quality_score": float(row.get("quality_score", 0)),
            "development_score": float(row.get("development_score", 0)),
        },
        "drivers": drivers,
        "strengths": strengths,
        "recommended_actions": [DRIVER_ACTIONS[d] for d in drivers if d in DRIVER_ACTIONS],
        "question": user_question,
    }

def llm_rewrite(payload: Dict[str, Any]) -> str:
    if client is None:
        return "\n".join(payload.get("recommended_actions", []))

    resp = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": json.dumps(payload, indent=2)}],
        temperature=0.3,
        max_tokens=300,
    )

    text = resp.choices[0].message.content if resp.choices else ""
    return clean_ai_text(text) if text else "[AI returned empty response]"


# =========================
# 8) ROUTING
# =========================
def route_request(user_id: str, question: str, df: Optional[pd.DataFrame] = None):
    if df is None:
        df = standard_df

    if user_id.startswith("EMP-"):
        emp_id = int(user_id.replace("EMP-", ""))
        row = df[df["employee_id"] == emp_id]
        if row.empty:
            return "Employee not found."
        payload = build_llm_payload(row.iloc[0], question)
        return llm_rewrite(payload)

    if user_id.startswith("MGR-"):
        return "Manager portal active."

    return "Unknown user role."


# =========================
# 9) MANAGER ANALYTICS
# =========================
def manager_team_insights(df: pd.DataFrame) -> Dict[str, Any]:
    require_columns(df)
    return {
        "team_size": len(df),
        "status_distribution": df["performance_status"].value_counts().to_dict(),
        "avg_output_score": round(df["output_score"].mean(), 2),
    }

def manager_private_ranking(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df)
    return df.sort_values("output_score", ascending=False)[
        ["employee_id", "performance_status", "output_score"]
    ]

def manager_ai_summary(df: pd.DataFrame) -> str:
    insights = manager_team_insights(df)
    return json.dumps(insights, indent=2)
