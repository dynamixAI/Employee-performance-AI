# engine.py
import os
import json
import pandas as pd
from typing import Any, Dict, List, Optional

from openai import OpenAI

# =========================================================
# 1) Load DEMO dataset from repo (Streamlit Cloud friendly)
# =========================================================
DATA_PATH = "standard_df.csv"

if os.path.exists(DATA_PATH):
    standard_df = pd.read_csv(DATA_PATH)
else:
    # Import-safe fallback (app can still load and show a clear error)
    standard_df = pd.DataFrame()

# =========================================================
# 2) OpenRouter client (import-safe)
# =========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
client: Optional[OpenAI] = None

if OPENROUTER_API_KEY:
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )

# =========================================================
# 3) Utilities / Validation
# =========================================================
REQUIRED_COLUMNS = [
    "employee_id",
    "output_score",
    "quality_score",
    "development_score",
    "performance_status",
]

def require_columns(df: pd.DataFrame, required: List[str] = REQUIRED_COLUMNS) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your dataset must contain at least: {required}"
        )

def clean_ai_text(text: str) -> str:
    # Remove common instruct-tokens from some open models
    for token in ["[B_INST]", "[/B_INST]", "<s>", "</s>", "[/OUT]", "[OUT]", "[/s]"]:
        text = text.replace(token, "")
    return text.strip()

def deterministic_format(payload: Dict[str, Any]) -> str:
    """
    Safe fallback when AI is unavailable.
    """
    status = payload.get("performance_status", "Unknown")
    drivers = payload.get("drivers", [])
    strengths = payload.get("strengths", [])
    actions = payload.get("recommended_actions", [])

    out = []
    out.append(f"Status: {status}\n")

    out.append("Why:")
    if drivers:
        for d in drivers:
            out.append(f"- {d}")
    else:
        out.append("- Not enough data to determine drivers")

    out.append("\nStrengths:")
    if strengths:
        for s in strengths:
            out.append(f"- {s}")
    else:
        out.append("- Not enough data to determine strengths")

    out.append("\nWhat to do next:")
    if actions:
        for i, a in enumerate(actions, start=1):
            out.append(f"{i}. {a}")
    else:
        out.append("1. Track key metrics and request feedback to identify improvement areas.")

    out.append("\nNote: This feedback is based only on the provided data.")
    return "\n".join(out)

# =========================================================
# 4) Business logic (drivers, explanations, questions)
# =========================================================
DRIVER_ACTIONS = {
    "Lower productivity relative to workload":
        "Review workload planning and focus on prioritising high-impact tasks.",
    "Quality score below expected level":
        "Focus on accuracy and seek feedback or refresher training to improve quality.",
    "High number of sick days affecting consistency":
        "Discuss workload balance or wellbeing support with your manager to improve consistency.",
    "Limited recent training or development activity":
        "Consider completing relevant training to strengthen skills and performance.",
    "No major performance risk factors identified":
        "Maintain current working practices and continue consistent performance.",
}

def identify_strengths(row: pd.Series) -> List[str]:
    strengths = []
    if float(row.get("output_score", 0)) >= 50:
        strengths.append("Good productivity level")
    if float(row.get("quality_score", 0)) >= 4:
        strengths.append("Strong quality of work")
    if float(row.get("Sick_Days", 0)) <= 5:
        strengths.append("Good attendance consistency")
    if not strengths:
        strengths.append("Maintaining baseline performance")
    return strengths

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
    if not drivers:
        drivers.append("No major performance risk factors identified")
    return drivers

def generate_employee_explanation(row: pd.Series) -> Dict[str, Any]:
    drivers = row.get("performance_drivers")
    if not isinstance(drivers, list):
        drivers = identify_performance_drivers(row)

    strengths = row.get("strengths")
    if not isinstance(strengths, list):
        strengths = identify_strengths(row)

    actions = [DRIVER_ACTIONS[d] for d in drivers if d in DRIVER_ACTIONS]

    return {
        "status": str(row.get("performance_status", "Unknown")),
        "summary": f"Your performance for the selected period is classified as '{row.get('performance_status', 'Unknown')}'.",
        "drivers": drivers,
        "strengths": strengths,
        "recommended_actions": actions,
    }

def answer_employee_question(employee_row: pd.Series, question: str) -> Any:
    q = (question or "").lower()
    explanation = employee_row.get("employee_explanation")

    if not isinstance(explanation, dict):
        explanation = generate_employee_explanation(employee_row)

    if "improve" in q or "focus" in q:
        return explanation["recommended_actions"]
    if "why" in q or "affected" in q:
        return explanation["drivers"]
    if "status" in q:
        return explanation["status"]
    if "doing well" in q or "strength" in q:
        return explanation["strengths"]

    return "Sorry, I can answer questions about your status, reasons (drivers), strengths, or improvement actions."

# =========================================================
# 5) LLM payload + rewrite (AI layer)
# =========================================================
SYSTEM_PROMPT = """
You are a supportive performance coach writing feedback for an employee.
Use ONLY the information provided in the JSON payload.
Do NOT invent metrics, reasons, comparisons, or outcomes.
Do NOT compare the employee to other employees.
Do NOT mention firing, discipline, or punishment.
Write in a constructive, non-judgemental tone.

Output format:
- Status (one short line)
- What happened (2–3 lines)
- Why (bullet list)
- Strengths (bullet list)
- What to do next (numbered list)
- Note (one short line about data limits)
""".strip()

def build_llm_payload(employee_row: pd.Series, user_question: Optional[str] = None) -> Dict[str, Any]:
    # Safely pull fields that may/may not exist
    payload = {
        "employee_id": int(employee_row.get("employee_id")),
        "sector": str(employee_row.get("sector", "")),
        "role": str(employee_row.get("role", "")),
        "date": str(employee_row.get("date", "")),
        "performance_status": str(employee_row.get("performance_status", "Unknown")),
        "metrics": {
            "output_score": float(employee_row.get("output_score", 0)),
            "quality_score": float(employee_row.get("quality_score", 0)),
            "development_score": float(employee_row.get("development_score", 0)),
            "sick_days": float(employee_row.get("Sick_Days", 0)),
        },
        "drivers": employee_row.get("performance_drivers", identify_performance_drivers(employee_row)),
        "strengths": employee_row.get("strengths", identify_strengths(employee_row)),
        "recommended_actions": [],
        "question": user_question,
    }

    # Ensure recommended actions exist
    actions = []
    for d in payload["drivers"]:
        if d in DRIVER_ACTIONS:
            actions.append(DRIVER_ACTIONS[d])
    payload["recommended_actions"] = actions

    return payload

def llm_rewrite(payload: Dict[str, Any]) -> str:
    # AI optional: if no client or key, return safe deterministic output
    if client is None:
        return deterministic_format(payload)

    user_prompt = (
        "Employee performance payload:\n"
        + json.dumps(payload, indent=2)
    )

    resp = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=400,
    )

    text = resp.choices[0].message.content if resp.choices else ""
    text = clean_ai_text(text or "")
    return text if text else "[AI returned empty response]"

# =========================================================
# 6) Role routing (Employee vs Manager)
# =========================================================
def get_user_context(user_id: str) -> Dict[str, Any]:
    if user_id.startswith("EMP-"):
        return {"role": "employee", "employee_id": int(user_id.replace("EMP-", ""))}
    if user_id.startswith("MGR-"):
        return {"role": "manager"}
    raise ValueError("Unknown user role. Use EMP-<id> or MGR-<id>.")

def handle_employee_request(user_context: Dict[str, Any], question: str, df: pd.DataFrame) -> Any:
    emp_id = user_context["employee_id"]
    row = df[df["employee_id"] == emp_id]
    if row.empty:
        return "Employee not found."
    emp_row = row.iloc[0]
    return answer_employee_question(emp_row, question)

def handle_manager_request(user_context: Dict[str, Any], question: Optional[str] = None) -> str:
    return (
        "Manager portal active.\n"
        "Use the dashboard to view team summaries, rankings, and coaching insights."
    )

def route_request(user_id: str, question: Optional[str] = None, df: Optional[pd.DataFrame] = None) -> Any:
    """
    NOTE: df is optional. If not supplied, defaults to standard_df.
    This matters for Streamlit when user uploads their own CSV.
    """
    if df is None:
        df = standard_df

    user_context = get_user_context(user_id)

    if user_context["role"] == "employee":
        return handle_employee_request(user_context, question or "", df)

    if user_context["role"] == "manager":
        return handle_manager_request(user_context, question)

    return "Unknown route."

# =========================================================
# 7) Manager analytics
# =========================================================
def manager_team_insights(df: pd.DataFrame) -> Dict[str, Any]:
    require_columns(df)

    return {
        "team_size": int(len(df)),
        "status_distribution": df["performance_status"].value_counts().to_dict(),
        "avg_output_score": round(float(df["output_score"].mean()), 2) if len(df) else 0.0,
        "common_issues": (
            df.explode("performance_drivers")["performance_drivers"]
              .value_counts()
              .head(5)
              .to_dict()
            if "performance_drivers" in df.columns else {}
        ),
    }

def manager_private_ranking(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df)
    cols = [c for c in ["employee_id", "performance_status", "output_score", "quality_score", "development_score"] if c in df.columns]
    return df[cols].sort_values(by="output_score", ascending=False)

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

def build_manager_ai_payload(df: pd.DataFrame) -> Dict[str, Any]:
    insights = manager_team_insights(df)
    formatted = format_manager_numbers(insights)
    return {
        "team_size": formatted["team_size"],
        "needs_improvement_pct": formatted["needs_improvement_pct"],
        "on_track_pct": formatted["on_track_pct"],
        "average_output_score": formatted["avg_output_score"],
        "most_common_issues": formatted["common_issues"],
    }

def manager_ai_summary(df: pd.DataFrame) -> str:
    """
    Token-safe manager summary:
    - Sends only aggregated numbers (no raw tables)
    """
    require_columns(df)

    payload = build_manager_ai_payload(df)

    # If AI not configured, return deterministic summary
    if client is None:
        return (
            "AI unavailable — API key not configured.\n\n"
            f"Team size: {payload['team_size']}\n"
            f"Needs improvement (%): {payload['needs_improvement_pct']}\n"
            f"On track (%): {payload['on_track_pct']}\n"
            f"Average output score: {payload['average_output_score']}\n"
            f"Top issues: {payload['most_common_issues']}\n"
        )

    prompt = (
        "You are an HR analytics advisor.\n\n"
        "TASK:\n"
        "Write a concise private management summary based on the team data.\n\n"
        "RULES:\n"
        "- Use the provided numbers exactly as written\n"
        "- Do NOT estimate, infer, or calculate percentages\n"
        "- Do NOT invent team size or scale\n"
        "- Do NOT use percentages unless explicitly provided\n"
        "- Focus on interpretation and coaching actions only\n"
        "- Do NOT shame individuals\n"
        "- Do NOT list every employee\n\n"
        "FORMAT:\n"
        "Team Overview:\n"
        "Key Issues:\n"
        "Who Needs Support (high level):\n"
        "Recommended Coaching Actions:\n\n"
        "TEAM SUMMARY DATA:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Now write the summary."
    )

    resp = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400,
    )

    text = resp.choices[0].message.content if resp.choices else ""
    text = clean_ai_text(text or "")
    return text if text else "[AI returned empty response]"
