# engine.py

import os
import json
import re
import hashlib
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# =====================================================
# 0) DATA LOAD (DEMO DATASET)
# =====================================================
DATA_PATH = "standard_df.csv"
if os.path.exists(DATA_PATH):
    standard_df = pd.read_csv(DATA_PATH)
else:
    standard_df = pd.DataFrame()

# =====================================================
# 1) AUTH + PASSWORDS (Streamlit-safe)
# =====================================================
#  ADDED: Missing password helpers that your file already calls
PASSWORD_POLICY = {
    "min_length": 8,
    "require_number": True,
    "require_special": True,
}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def verify_password(password: str, password_hash: str) -> bool:
    return hash_password(password) == password_hash

def validate_password(password: str) -> List[str]:
    errors = []
    if len(password) < PASSWORD_POLICY["min_length"]:
        errors.append(f"Password must be at least {PASSWORD_POLICY['min_length']} characters.")
    if PASSWORD_POLICY["require_number"] and not re.search(r"\d", password):
        errors.append("Password must contain at least one number.")
    if PASSWORD_POLICY["require_special"] and not re.search(r"[!@#$%^&*]", password):
        errors.append("Password must contain at least one special character (!@#$%^&*).")
    return errors


# =====================================================
# USER REGISTRY — ADMIN IS SOURCE OF TRUTH
# =====================================================
# user_id: login identifier
# role: admin | manager | employee
# data_employee_id: links user to dataset (employees)
# data_scope_type / value: manager access control

users_df = pd.DataFrame(
    columns=[
        "user_id",
        "role",
        "password_hash",
        "data_employee_id",
        "data_scope_type",
        "data_scope_value",
    ]
)

def _seed_users():
    global users_df
    if not users_df.empty:
        return

    #  CHANGE: Admin is the ONLY default account.
    # Managers and employees MUST be created by admin via admin_create_user().
    users_df = pd.DataFrame([
        {
            "user_id": "ADM-01",
            "role": "admin",
            "password_hash": hash_password("Admin@123"),
            "data_employee_id": None,
            "data_scope_type": None,
            "data_scope_value": None,
        },
    ])

_seed_users()

def authenticate_user(user_id: str, password: str) -> Optional[Dict[str, Any]]:
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        return None

    row = row.iloc[0]
    if not verify_password(password, row["password_hash"]):
        return None

    return {
        "user_id": row["user_id"],
        "role": row["role"],
        "data_employee_id": row["data_employee_id"],
        "data_scope_type": row["data_scope_type"],
        "data_scope_value": row["data_scope_value"],
    }

def admin_create_user(
    user_id: str,
    role: str,
    password: str,
    data_employee_id: Optional[str] = None,
    data_scope_type: Optional[str] = None,
    data_scope_value: Optional[Any] = None,
) -> Dict[str, Any]:
    global users_df

    if role not in ["admin", "manager", "employee"]:
        return {"success": False, "error": "Invalid role."}

    if user_id in users_df["user_id"].values:
        return {"success": False, "error": "User already exists."}

    pw_errors = validate_password(password)
    if pw_errors:
        return {"success": False, "error": pw_errors}

    users_df = pd.concat([users_df, pd.DataFrame([{
        "user_id": user_id,
        "role": role,
        "password_hash": hash_password(password),
        "data_employee_id": data_employee_id,
        "data_scope_type": data_scope_type,
        "data_scope_value": json.dumps(data_scope_value) if isinstance(data_scope_value, (list, dict)) else data_scope_value,
    }])], ignore_index=True)

    return {"success": True}

def admin_remove_user(user_id: str) -> Dict[str, Any]:
    global users_df
    if user_id not in users_df["user_id"].values:
        return {"success": False, "error": "User not found."}
    users_df = users_df[users_df["user_id"] != user_id]
    return {"success": True}

def admin_reset_password(user_id: str, new_password: str) -> Dict[str, Any]:
    pw_errors = validate_password(new_password)
    if pw_errors:
        return {"success": False, "error": pw_errors}

    idx = users_df.index[users_df["user_id"] == user_id]
    if len(idx) == 0:
        return {"success": False, "error": "User not found."}

    users_df.loc[idx, "password_hash"] = hash_password(new_password)
    return {"success": True}

# =====================================================
# 2) CANONICAL SCHEMA + AUTO-MAPPING (FROM YOUR COLAB)
# =====================================================
CANONICAL_SCHEMA: Dict[str, Dict[str, Any]] = {
    "employee_id": {"required": True},
    "sector": {"required": False},
    "role": {"required": False},
    "date": {"required": True},              # you used record_date -> date
    "quality_score": {"required": True},     # Performance_Score
    "development_score": {"required": True}, # Training_Hours
    "output_score": {"required": True},      # computed
    "performance_status": {"required": True} # derived
}

# Keyword rules for mapping
MAPPING_KEYWORDS: Dict[str, List[str]] = {
    "employee_id": ["employee_id", "employee id", "id", "staff_id", "staff id", "person_id", "person id"],
    "sector": ["department", "dept", "team", "function", "unit", "sector"],
    "role": ["job_title", "job title", "role", "position", "designation"],
    "date": ["record_date", "date", "period", "month", "week", "as_of", "as of"],
    "quality_score": ["performance_score", "quality", "rating", "score", "review_score"],
    "development_score": ["training_hours", "training", "learning_hours", "development", "upskilling"],
    # For computing output_score we need these raw columns (not canonical)
    "projects_handled": ["projects_handled", "projects", "tickets", "cases", "tasks_completed", "tasks", "handled"],
    "work_hours_per_week": ["work_hours_per_week", "hours", "weekly_hours", "work_hours"],
    "overtime_hours": ["overtime_hours", "overtime", "ot_hours", "extra_hours"],
    "sick_days": ["sick_days", "absence_days", "absences", "sickness", "sick"]
}

def _normalize(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")

def suggest_column_mapping(columns: List[str], keyword_rules: Dict[str, List[str]] = MAPPING_KEYWORDS) -> Dict[str, str]:
    """
    Suggest mapping from raw dataframe columns -> canonical or needed raw fields.
    Returns dict {raw_column_name: canonical_name}
    """
    norm_map = {c: _normalize(c) for c in columns}
    suggested: Dict[str, str] = {}

    # Inverse lookup: for each canonical, find best matching raw col
    for canonical_field, keywords in keyword_rules.items():
        best = None
        for raw_col, normed in norm_map.items():
            for kw in keywords:
                if _normalize(kw) in normed:
                    best = raw_col
                    break
            if best:
                break
        if best:
            suggested[best] = canonical_field

    return suggested

def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns based on mapping and return a new df.
    mapping is {raw_col: new_name}
    """
    out = df.copy()
    out = out.rename(columns=mapping)
    return out

def validate_required_fields(df: pd.DataFrame, canonical_schema: Dict[str, Dict[str, Any]] = CANONICAL_SCHEMA) -> List[str]:
    required = [f for f, props in canonical_schema.items() if props.get("required", False)]
    missing = [f for f in required if f not in df.columns]
    return missing

# =====================================================
# 3) COMPUTE OUTPUT_SCORE (YOUR PRODUCTIVITY + MINMAX)
# =====================================================
def _minmax_scale(series: pd.Series, feature_range: Tuple[float, float] = (0.0, 100.0)) -> pd.Series:
    """
    Min-max scaling without sklearn to keep Streamlit deps smaller.
    """
    a, b = feature_range
    s = pd.to_numeric(series, errors="coerce")
    minv = s.min()
    maxv = s.max()
    if pd.isna(minv) or pd.isna(maxv) or maxv == minv:
        # avoid divide-by-zero; all same -> mid-point
        return pd.Series([ (a + b) / 2 ] * len(series), index=series.index)
    return (s - minv) / (maxv - minv) * (b - a) + a

def compute_output_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Implements your notebook logic:
    raw_productivity = Projects_Handled / (Work_Hours_Per_Week + 0.5*Overtime_Hours)
    output_score = MinMaxScaler(0..100)(raw_productivity)
    """
    out = df.copy()

    # Accept both canonicalized raw fields and typical original names
    # (so it still works after mapping or if user uploads original)
    proj_col = "projects_handled" if "projects_handled" in out.columns else "Projects_Handled" if "Projects_Handled" in out.columns else None
    wh_col = "work_hours_per_week" if "work_hours_per_week" in out.columns else "Work_Hours_Per_Week" if "Work_Hours_Per_Week" in out.columns else None
    ot_col = "overtime_hours" if "overtime_hours" in out.columns else "Overtime_Hours" if "Overtime_Hours" in out.columns else None

    if not proj_col or not wh_col:
        # Can't compute; leave as-is if already there
        if "output_score" not in out.columns:
            out["output_score"] = 0.0
        return out

    # If overtime missing, treat as 0
    if not ot_col:
        out["_overtime_tmp"] = 0.0
        ot_col = "_overtime_tmp"

    projects = pd.to_numeric(out[proj_col], errors="coerce").fillna(0.0)
    work_hours = pd.to_numeric(out[wh_col], errors="coerce").fillna(0.0)
    overtime = pd.to_numeric(out[ot_col], errors="coerce").fillna(0.0)

    raw_productivity = projects / (work_hours + 0.5 * overtime + 1e-9)
    out["output_score"] = _minmax_scale(raw_productivity, (0.0, 100.0))

    if "_overtime_tmp" in out.columns:
        out.drop(columns=["_overtime_tmp"], inplace=True)

    return out

# =====================================================
# 4) STATUS + DRIVERS + STRENGTHS (MATCHING YOUR NOTEBOOK)
# =====================================================
def determine_performance_status(row: pd.Series) -> str:
    """
    Colab rule:
    On Track if output_score>=50 AND quality_score>=3 AND Sick_Days<=10 else Needs Improvement
    """
    sick = float(row.get("Sick_Days", row.get("sick_days", 0)) or 0)
    if float(row.get("output_score", 0)) >= 50 and float(row.get("quality_score", 0)) >= 3 and sick <= 10:
        return "On Track"
    return "Needs Improvement"

def identify_performance_drivers(row: pd.Series) -> List[str]:
    drivers = []
    if float(row.get("output_score", 0)) < 50:
        drivers.append("Lower productivity relative to workload")
    if float(row.get("quality_score", 0)) < 3:
        drivers.append("Quality score below expected level")
    sick = float(row.get("Sick_Days", row.get("sick_days", 0)) or 0)
    if sick > 10:
        drivers.append("High number of sick days affecting consistency")
    if float(row.get("development_score", 0)) < 5:
        drivers.append("Limited recent training or development activity")
    return drivers

def identify_strengths(row: pd.Series) -> List[str]:
    strengths = []
    if float(row.get("quality_score", 0)) >= 4:
        strengths.append("Strong quality of work")
    sick = float(row.get("Sick_Days", row.get("sick_days", 0)) or 0)
    if sick <= 5:
        strengths.append("Good attendance consistency")
    if float(row.get("output_score", 0)) >= 60:
        strengths.append("Good productivity level")
    return strengths

DRIVER_ACTIONS = {
    "Lower productivity relative to workload":
        "Review workload planning and focus on prioritising high-impact tasks.",
    "Quality score below expected level":
        "Seek feedback or refresher training to improve quality.",
    "High number of sick days affecting consistency":
        "Discuss wellbeing support and workload balance to improve consistency.",
    "Limited recent training or development activity":
        "Complete relevant training to strengthen skills and performance.",
}

def generate_employee_explanation(row: pd.Series) -> Dict[str, Any]:
    drivers = identify_performance_drivers(row)
    strengths = identify_strengths(row)
    actions = [DRIVER_ACTIONS[d] for d in drivers if d in DRIVER_ACTIONS]

    return {
        "status": row.get("performance_status", "Needs Improvement"),
        "summary": (
            f"Your performance for the selected period is classified as "
            f"'{row.get('performance_status', 'Needs Improvement')}'."
        ),
        "drivers": drivers,
        "strengths": strengths,
        "recommended_actions": actions,
    }

# =====================================================
# 5) ALLOWED QUESTIONS (EXPLICIT INTENT TABLE FROM COLAB)
# =====================================================
ALLOWED_QUESTIONS = {
    "why was my performance low": "drivers",
    "why is my performance low": "drivers",
    "why": "drivers",
    "what should i focus on": "recommended_actions",
    "how can i improve": "recommended_actions",
    "improve": "recommended_actions",
    "what am i doing well": "strengths",
    "strengths": "strengths",
    "status": "status",
}

def _detect_intent(question: str) -> str:
    q = (question or "").strip().lower()
    # Find best match: exact keys first, then substring heuristics
    for k, intent in ALLOWED_QUESTIONS.items():
        if q == k:
            return intent
    for k, intent in ALLOWED_QUESTIONS.items():
        if k in q:
            return intent
    # fallback: still guard-railed
    if "why" in q:
        return "drivers"
    if "improv" in q or "focus" in q:
        return "recommended_actions"
    if "well" in q or "strength" in q:
        return "strengths"
    if "status" in q:
        return "status"
    return "unsupported"

def answer_employee_question(employee_row: pd.Series, question: str) -> Any:
    expl = employee_row.get("employee_explanation")
    if not isinstance(expl, dict):
        expl = generate_employee_explanation(employee_row)

    intent = _detect_intent(question)
    if intent == "drivers":
        return expl["drivers"]
    if intent == "recommended_actions":
        return expl["recommended_actions"]
    if intent == "strengths":
        return expl["strengths"]
    if intent == "status":
        return expl["status"]

    return (
        "I can answer questions about: why your performance is low, "
        "what to focus on next, what you are doing well, or your status."
    )

# =====================================================
# 6) DETERMINISTIC FALLBACK FORMATTER (FROM COLAB)
# =====================================================
def deterministic_format(payload: Dict[str, Any]) -> str:
    q = (payload.get("user_question") or "").lower()

    strengths = payload.get("strengths", [])
    drivers = payload.get("drivers", [])
    actions = payload.get("recommended_actions", [])
    status = payload.get("performance_status", "Unknown")

    if any(k in q for k in ["well", "strength"]):
        return "Here’s what you’re doing well:\n\n" + "\n".join(f"- {s}" for s in strengths)

    if any(k in q for k in ["why", "low", "problem"]):
        return "Here’s why your performance is rated this way:\n\n" + "\n".join(f"- {d}" for d in drivers)

    if any(k in q for k in ["improve", "focus", "next"]):
        return "Here’s what you should focus on next:\n\n" + "\n".join(f"- {a}" for a in actions)

    return (
        f"Status: {status}\n\n"
        "Recommended next steps:\n"
        + "\n".join(f"- {a}" for a in actions)
    )
# =====================================================
# 7) AI GATEWAY (EMPLOYEE + MANAGER)
# =====================================================
def build_manager_payload(
    df: pd.DataFrame,
    question: str
) -> Dict[str, Any]:
    """
    Build manager-level AI payload using aggregated data only.
    REQUIRED by manager_dashboard in app.py
    """

    if df is None or df.empty:
        return {
            "level": "manager",
            "user_question": question,
            "summary": {},
            "risk_groups": {},
        }

    return {
        "level": "manager",
        "user_question": question,
        "summary": {
            "total_employees": int(len(df)),
            "avg_output_score": float(
                pd.to_numeric(df["output_score"], errors="coerce").mean()
            ),
            "avg_quality_score": float(
                pd.to_numeric(df["quality_score"], errors="coerce").mean()
            ),
            "status_breakdown": (
                df["performance_status"]
                .value_counts()
                .to_dict()
            ),
        },
        "risk_groups": {
            "low_output": int(
                (df["output_score"] < df["output_score"].quantile(0.25)).sum()
            ),
            "low_quality": int(
                (df["quality_score"] < df["quality_score"].quantile(0.25)).sum()
            ),
            "high_sick_days": int(
                (df["Sick_Days"] > df["Sick_Days"].quantile(0.75)).sum()
            ),
        },
    }

def build_llm_payload(
    employee_row: pd.Series,
    user_question: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build employee-level payload for AI.
    Mirrors original Colab logic.
    """

    expl = employee_row.get("employee_explanation")
    if not isinstance(expl, dict):
        expl = generate_employee_explanation(employee_row)

    return {
        "level": "employee",
        "employee_id": str(employee_row.get("employee_id")),
        "sector": str(employee_row.get("sector", "")),
        "role": str(employee_row.get("role", "")),
        "date": str(employee_row.get("date", "")),
        "performance_status": expl.get(
            "status",
            employee_row.get("performance_status", "Needs Improvement")
        ),
        "metrics": {
            "output_score": float(employee_row.get("output_score", 0)),
            "quality_score": float(employee_row.get("quality_score", 0)),
            "development_score": float(employee_row.get("development_score", 0)),
            "sick_days": float(
                employee_row.get("Sick_Days", employee_row.get("sick_days", 0)) or 0
            ),
        },
        "drivers": expl.get("drivers", []),
        "strengths": expl.get("strengths", []),
        "recommended_actions": expl.get("recommended_actions", []),
        "user_question": user_question,
    }


def compress_manager_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce manager payload to AI-safe summary.
    CRITICAL: keep user_question key.
    """

    return {
        "level": "manager",
        "user_question": payload.get("user_question", ""),
        "summary": payload.get("summary", {}),
        "risk_groups": payload.get("risk_groups", {}),
    }


def llm_rewrite(payload: Dict[str, Any]) -> str:
    """
    Shared AI gateway for employee and manager.
    Multi-model fallback, never crashes the app.
    """

    # ---- deterministic fallback ----
    if client is None:
        return deterministic_format(payload)

    is_manager = payload.get("level") == "manager"

    # ---- system prompt routing ----
    system_prompt = (
        MANAGER_SYSTEM_PROMPT
        if is_manager
        else EMPLOYEE_SYSTEM_PROMPT
    )

    # ---- compress manager payload BEFORE prompting ----
    safe_payload = (
        compress_manager_payload(payload)
        if is_manager
        else payload
    )

    # ---- user prompt ----
    if is_manager:
        user_prompt = f"""
The manager asked the following question:
\"{safe_payload.get('user_question', '').strip()}\"

Use ONLY the aggregated team data below.
Focus on patterns, risks, and priorities.
Do NOT invent individual employee details.

Team data:
{json.dumps(safe_payload, indent=2)}
"""
    else:
        user_prompt = f"""
The employee asked the following question:
\"{safe_payload.get('user_question', '').strip()}\"

Answer THIS QUESTION directly.
Use ONLY the data below.
Do NOT invent metrics or assumptions.

Employee data:
{json.dumps(safe_payload, indent=2)}
"""

    # ---- multi-model fallback ----
    for model in OPENROUTER_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=450,
            )

            if not resp.choices:
                continue

            text = resp.choices[0].message.content or ""
            text = clean_ai_text(text)

            if text.strip():
                print(f"✅ AI response from {model}")
                return text

        except Exception as e:
            print(f"⚠️ Model failed → {model} | {type(e).__name__}")
            continue

    print("❌ All AI models failed — using deterministic fallback")
    return deterministic_format(payload)



# =====================================================
# 8) DATA PIPELINE HELPERS – BUILD STANDARD_DF (FROM RAW)
# =====================================================
def prepare_standard_df(raw_df: pd.DataFrame,
                        mapping: Optional[Dict[str, str]] = None,
                        record_date_value: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Rebuilds your notebook pipeline:
    - (optional) apply mapping
    - create/ensure record_date then map to date
    - compute output_score
    - create performance_status
    - create performance_drivers, strengths, employee_explanation
    Returns: (standard_df, report)
    """
    df_work = raw_df.copy()

    # Apply mapping if provided
    if mapping:
        df_work = apply_mapping(df_work, mapping)

    # Ensure record_date then alias to canonical date
    if "record_date" not in df_work.columns:
        if record_date_value is None:
            # safe default: today-ish string; UI can override
            record_date_value = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        df_work["record_date"] = record_date_value

    # If user mapped record_date -> date already, keep date; else map record_date to date
    if "date" not in df_work.columns:
        df_work["date"] = df_work["record_date"]

    # Map likely original cols to canonical names if still present
    rename_map = {}
    if "Employee_ID" in df_work.columns and "employee_id" not in df_work.columns:
        rename_map["Employee_ID"] = "employee_id"
    if "Department" in df_work.columns and "sector" not in df_work.columns:
        rename_map["Department"] = "sector"
    if "Job_Title" in df_work.columns and "role" not in df_work.columns:
        rename_map["Job_Title"] = "role"
    if "Performance_Score" in df_work.columns and "quality_score" not in df_work.columns:
        rename_map["Performance_Score"] = "quality_score"
    if "Training_Hours" in df_work.columns and "development_score" not in df_work.columns:
        rename_map["Training_Hours"] = "development_score"
    if rename_map:
        df_work = df_work.rename(columns=rename_map)

    # Compute output_score
    df_work = compute_output_score(df_work)

    # Ensure required canonical columns exist
    missing = validate_required_fields(df_work, CANONICAL_SCHEMA)

    # If performance_status missing, compute it (depends on output_score, quality_score, sick days)
    if "performance_status" not in df_work.columns and "output_score" in df_work.columns and "quality_score" in df_work.columns:
        df_work["performance_status"] = df_work.apply(determine_performance_status, axis=1)

    # If now missing required fields still (because performance_status was required)
    missing = validate_required_fields(df_work, CANONICAL_SCHEMA)

    # Enrich with drivers/strengths/explanation (matches Colab approach)
    df_work["performance_drivers"] = df_work.apply(identify_performance_drivers, axis=1)
    df_work["strengths"] = df_work.apply(identify_strengths, axis=1)
    df_work["employee_explanation"] = df_work.apply(generate_employee_explanation, axis=1)

    # Final report
    report = {
        "missing_required_fields": missing,
        "rows": int(len(df_work)),
        "columns": list(df_work.columns),
    }

    # Select canonical + useful extra columns (keep others too, but canonical guaranteed)
    return df_work, report

# =====================================================
# 9) ROLE ROUTING (EMPLOYEE vs MANAGER) – Streamlit calls this
# =====================================================
# stop guessing role from prefix. Use admin assignments in users_df.

def get_user_context(user_id: str) -> Dict[str, Any]:
    user_id = (user_id or "").strip()
    row = users_df[users_df["user_id"] == user_id]
    if row.empty:
        return {"role": "unknown"}
    row = row.iloc[0]
    return {
        "role": row["role"],
        "data_employee_id": row["data_employee_id"],
        "data_scope_type": row["data_scope_type"],
        "data_scope_value": row["data_scope_value"],
        "user_id": row["user_id"],
    }

#  Identity → Data resolvers (Admin-controlled access)
def get_employee_row(df: pd.DataFrame, data_employee_id: Any) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "employee_id" not in df.columns:
        return pd.DataFrame()
    return df[df["employee_id"].astype(str) == str(data_employee_id)]

def get_manager_scope_df(df: pd.DataFrame, scope_type: Any, scope_value: Any) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    if scope_type == "department":
        if "sector" not in df.columns:
            return pd.DataFrame()
        return df[df["sector"].astype(str) == str(scope_value)]

    if scope_type == "employee_ids":
        if isinstance(scope_value, str):
            try:
                scope_value = json.loads(scope_value)
            except Exception:
                scope_value = []
        if "employee_id" not in df.columns:
            return pd.DataFrame()
        return df[df["employee_id"].astype(str).isin([str(x) for x in scope_value])]

    return pd.DataFrame()

def resolve_user_data_view(ctx: Dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    role = ctx.get("role")
    if role == "employee":
        return get_employee_row(df, ctx.get("data_employee_id"))
    if role == "manager":
        return get_manager_scope_df(df, ctx.get("data_scope_type"), ctx.get("data_scope_value"))
    if role == "admin":
        return df.copy()
    return pd.DataFrame()

def route_request(user_id: str, question: str, df: Optional[pd.DataFrame] = None) -> Any:
    """
    Employee: guard-railed answers based on deterministic explanation.
    Manager/Admin: basic message (UI provides dashboards).
    """
    if df is None:
        df = standard_df

    ctx = get_user_context(user_id)

    if ctx["role"] == "employee":
        emp_df = resolve_user_data_view(ctx, df)
        if emp_df.empty:
            return "Employee not found (not assigned to data)."
        return answer_employee_question(emp_df.iloc[0], question)

    if ctx["role"] in ["manager", "admin"]:
        return "Manager/Admin portal active."

    return "Unknown user role."

# =====================================================
# 10) MANAGER: TOKEN-SAFE INSIGHTS + COACHING + AI SUMMARY
# =====================================================
def manager_team_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Aggregated insights only; no raw table passed to LLM.
    """
    if df.empty:
        return {"team_size": 0, "status_distribution": {}, "avg_output_score": 0.0, "common_issues": {}}

    status_dist = df["performance_status"].value_counts().to_dict()
    avg_output = float(pd.to_numeric(df["output_score"], errors="coerce").mean())

    # most common issues (explode drivers)
    common_issues = {}
    if "performance_drivers" in df.columns:
        common_issues = (
            df.explode("performance_drivers")["performance_drivers"]
              .value_counts()
              .head(5)
              .to_dict()
        )

    return {
        "team_size": int(len(df)),
        "status_distribution": status_dist,
        "avg_output_score": round(avg_output, 2) if avg_output == avg_output else 0.0,
        "common_issues": common_issues,
    }

def manager_private_ranking(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Private ranking table for manager UI (not for AI).
    """
    if df.empty:
        return df
    cols = [c for c in ["employee_id", "performance_status", "output_score", "quality_score", "development_score"] if c in df.columns]
    ranked = df.sort_values("output_score", ascending=False)[cols]
    return ranked.head(top_n)

def manager_coaching_actions(df: pd.DataFrame, top_n_needing_support: int = 20) -> List[Dict[str, Any]]:
    """
    Builds coaching list (token-safe): only top N employees needing support.
    """
    if df.empty:
        return []

    needs = df[df["performance_status"] == "Needs Improvement"].copy()
    if needs.empty:
        return []

    needs = needs.sort_values("output_score", ascending=True).head(top_n_needing_support)

    actions = []
    for _, r in needs.iterrows():
        expl = r.get("employee_explanation")
        if not isinstance(expl, dict):
            expl = generate_employee_explanation(r)
        actions.append({
            # ✅ CHANGED: do NOT force int
            "employee_id": str(r.get("employee_id")),
            "status": r.get("performance_status"),
            "drivers": expl.get("drivers", []),
            "recommended_actions": expl.get("recommended_actions", []),
        })
    return actions

def format_manager_numbers(insights: Dict[str, Any]) -> Dict[str, Any]:
    team_size = max(int(insights.get("team_size", 0)), 1)
    dist = insights.get("status_distribution", {})
    return {
        "team_size": insights.get("team_size", 0),
        "needs_improvement_pct": round(100 * dist.get("Needs Improvement", 0) / team_size, 2),
        "on_track_pct": round(100 * dist.get("On Track", 0) / team_size, 2),
        "avg_output_score": insights.get("avg_output_score", 0),
        "common_issues": insights.get("common_issues", {}),
    }

def build_manager_payload(df: pd.DataFrame, question: str) -> dict:
    return {
        "level": "manager",
        "question": question,
        "summary": {
            "total_employees": int(len(df)),
            "avg_output_score": float(df["output_score"].mean()),
            "avg_quality_score": float(df["quality_score"].mean()),
            "status_breakdown": df["performance_status"]
            .value_counts()
            .to_dict(),
        },
        "risk_groups": {
            "low_output": int((df["output_score"] < df["output_score"].quantile(0.25)).sum()),
            "low_quality": int((df["quality_score"] < df["quality_score"].quantile(0.25)).sum()),
            "high_sick_days": int((df["Sick_Days"] > df["Sick_Days"].quantile(0.75)).sum()),
        },
    }


MANAGER_SYSTEM_PROMPT = """
You are an AI People Analytics advisor for managers.

Your task:
- Answer the manager's QUESTION using the aggregated data provided.
- Identify patterns, risks, and priorities.
- Focus on support and improvement, not blame.

Rules:
- Use ONLY the data in the payload.
- Do NOT invent employee details.
- Provide actionable recommendations.

Tone:
Professional, strategic, concise.
""".strip()

def manager_ai_summary(df: pd.DataFrame) -> str:
    """
    Token-safe: only aggregated data + top 20 coaching priorities.
    """
    payload = build_manager_ai_payload(df)

    if client is None:
        # deterministic management summary when AI not configured
        return (
            "AI unavailable — API key not configured.\n\n"
            + json.dumps(payload, indent=2)
        )

    user_prompt = "TEAM SUMMARY DATA:\n" + json.dumps(payload, indent=2) + "\n\nNow write the summary."
    resp = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {"role": "system", "content": MANAGER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=450
    )
    text = resp.choices[0].message.content if resp.choices else ""
    text = clean_ai_text(text or "")
    return text if text else "[AI returned empty response]"

# =====================================================
# ADMIN: VIEW USER REGISTRY (SAFE)
# =====================================================
def admin_list_users() -> pd.DataFrame:
    """
    Returns a safe view of users (no passwords).
    """
    if users_df.empty:
        return pd.DataFrame()

    cols = [
        "user_id",
        "role",
        "data_employee_id",
        "data_scope_type",
        "data_scope_value",
    ]
    return users_df[cols].copy()



# =====================================================
# 11) AUTO-ENRICH LOADED CSV (SAFE)
# =====================================================
#  ADDED: If your CSV is "raw" (missing computed columns), enrich it at startup.
if standard_df is not None and not standard_df.empty:
    required = ["employee_id", "sector", "quality_score", "development_score", "date"]
    has_minimum = all(col in standard_df.columns for col in required)

    # If minimum canonical columns exist but derived columns do not, prepare/enrich.
    if has_minimum and ("employee_explanation" not in standard_df.columns or "performance_status" not in standard_df.columns):
        try:
            standard_df, _ = prepare_standard_df(standard_df, mapping=None, record_date_value=None)
        except Exception:
            # Keep app alive even if enrichment fails; UI can show error.
            pass
