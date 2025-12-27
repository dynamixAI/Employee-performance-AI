import pandas as pd
import numpy as np
from datetime import datetime

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 120)


# Canonical performance schema the system understands
CANONICAL_SCHEMA = {
    "employee_id": {
        "required": True,
        "description": "Unique identifier for an employee"
    },
    "date": {
        "required": True,
        "description": "Date of the performance record"
    },
    "role": {
        "required": True,
        "description": "Job role of the employee"
    },
    "sector": {
        "required": True,
        "description": "Business sector or function (Sales, HR, Ops, etc.)"
    },
    "output_score": {
        "required": True,
        "description": "Core productivity or output metric"
    },

    # Optional but valuable fields
    "quality_score": {
        "required": False,
        "description": "Quality or accuracy of work"
    },
    "reliability_score": {
        "required": False,
        "description": "Attendance or reliability indicator"
    },
    "development_score": {
        "required": False,
        "description": "Training, learning, or development metric"
    },
    "target_score": {
        "required": False,
        "description": "Expected or target performance value"
    }
}

CANONICAL_SCHEMA


from google.colab import files

uploaded = files.upload()


import zipfile
import os

# Unzip the uploaded archive
zip_file_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    zip_ref.extractall('.')

# Find the CSV file (assuming there's one relevant CSV after unzipping)
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

if csv_files:
    # Assuming the first CSV file is the one to load
    csv_file_to_load = csv_files[0]
    df = pd.read_csv(csv_file_to_load)
    print(f"Successfully loaded '{csv_file_to_load}' into a DataFrame named `df`.")
    print("Here's the head of the DataFrame:")
    print(df.head())
else:
    print("No CSV files found after unzipping.")

# Mapping columns

# Keywords the system uses to guess column meaning
MAPPING_KEYWORDS = {
    "employee_id": [
        "employee", "emp", "staff", "staff_no", "person", "worker", "id"
    ],
    "date": [
        "date", "time", "period", "record", "month", "day"
    ],
    "role": [
        "role", "job", "position", "title"
    ],
    "sector": [
        "sector", "department", "dept", "function", "business", "unit", "division"
    ],
    "output_score": [
        "revenue", "sales", "output", "productivity", "kpi", "turnover", "deals", "units"
    ],
    "quality_score": [
        "quality", "qa", "accuracy", "error", "score"
    ],
    "reliability_score": [
        "attendance", "absence", "absent", "lateness", "reliability"
    ],
    "development_score": [
        "training", "learning", "development", "coaching", "course"
    ],
    "target_score": [
        "target", "quota", "goal", "expected"
    ]
}


def suggest_column_mapping(columns, keyword_rules):
    suggestions = {}           # source_column -> canonical_field
    used_canonical = set()     # prevent duplicates where possible

    for col in columns:
        col_lower = col.lower().strip()

        for canonical_field, keywords in keyword_rules.items():
            if canonical_field in used_canonical:
                continue

            if any(keyword in col_lower for keyword in keywords):
                suggestions[col] = canonical_field
                used_canonical.add(canonical_field)
                break

    return suggestions


suggested_mapping = suggest_column_mapping(
    df.columns,
    MAPPING_KEYWORDS
)

suggested_mapping

# Make a copy so we never touch the raw data
df_working = df.copy()

# Create a performance record date (snapshot assumption)
df_working["record_date"] = pd.to_datetime("2024-12-31")

df_working.head()


confirmed_mapping = {
    "Employee_ID": "employee_id",
    "Department": "sector",
    "Job_Title": "role",
    "Performance_Score": "quality_score",
    "Training_Hours": "development_score",
    "record_date": "date"
}

confirmed_mapping


def apply_mapping(df, mapping):
    return df.rename(columns=mapping).copy()

standard_df = apply_mapping(df_working, confirmed_mapping)
standard_df.head()


def validate_required_fields(df, canonical_schema):
    required_fields = [
        field for field, props in canonical_schema.items()
        if props["required"]
    ]

    missing_fields = [
        field for field in required_fields
        if field not in df.columns
    ]

    return missing_fields


missing_required_fields = validate_required_fields(standard_df, CANONICAL_SCHEMA)
missing_required_fields


from sklearn.preprocessing import MinMaxScaler

df_calc = standard_df.copy()

df_calc["raw_productivity"] = (
    df_calc["Projects_Handled"] /
    (df_calc["Work_Hours_Per_Week"] + 0.5 * df_calc["Overtime_Hours"])
)


scaler = MinMaxScaler(feature_range=(0, 100))

df_calc["output_score"] = scaler.fit_transform(
    df_calc[["raw_productivity"]]
)


df_calc[[
    "employee_id",
    "Projects_Handled",
    "Work_Hours_Per_Week",
    "Overtime_Hours",
    "raw_productivity",
    "output_score"
]].head()


standard_df = df_calc.drop(columns=["raw_productivity"])
standard_df.head()


#Create Performance Status Function

def determine_performance_status(row):
    if (
        row["output_score"] >= 50 and
        row["quality_score"] >= 3 and
        row["Sick_Days"] <= 10
    ):
        return "On Track"
    else:
        return "Needs Improvement"


standard_df["performance_status"] = standard_df.apply(
    determine_performance_status,
    axis=1
)

standard_df[
    ["employee_id", "output_score", "quality_score", "Sick_Days", "performance_status"]
].head()


# Driver logic

def identify_performance_drivers(row):
    drivers = []

    if row["output_score"] < 50:
        drivers.append("Lower productivity relative to workload")

    if row["quality_score"] < 3:
        drivers.append("Quality score below expected level")

    if row["Sick_Days"] > 10:
        drivers.append("High number of sick days affecting consistency")

    if row["development_score"] < 5:
        drivers.append("Limited recent training or development activity")

    if not drivers:
        drivers.append("No major performance risk factors identified")

    return drivers

standard_df["performance_drivers"] = standard_df.apply(
    identify_performance_drivers,
    axis=1
)

standard_df[
    ["employee_id", "performance_status", "performance_drivers"]
].head()


# Action Recommendation Logic

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
        "Maintain current working practices and continue consistent performance."
}


def generate_employee_explanation(row):
    explanation = {}

    # 1. Status
    explanation["status"] = row["performance_status"]

    # 2. What happened (facts)
    explanation["summary"] = (
        f"Your performance for the selected period is classified as "
        f"'{row['performance_status']}'."
    )

    # 3. Why it happened (drivers)
    explanation["drivers"] = row["performance_drivers"]

    # 4. Recommended actions
    actions = []
    for driver in row["performance_drivers"]:
        action = DRIVER_ACTIONS.get(driver)
        if action:
            actions.append(action)

    explanation["recommended_actions"] = actions

    return explanation


standard_df["employee_explanation"] = standard_df.apply(
    generate_employee_explanation,
    axis=1
)

standard_df[
    ["employee_id", "employee_explanation"]
].head()


# Defining allowed questions

ALLOWED_QUESTIONS = {
    "improve": "recommended_actions",
    "affected": "drivers",
    "why": "drivers",
    "status": "status",
    "summary": "summary",
    "doing well": "strengths",
    "focus": "recommended_actions"
}

def identify_strengths(row):
    strengths = []

    if row["output_score"] >= 50:
        strengths.append("Good productivity level")

    if row["quality_score"] >= 4:
        strengths.append("Strong quality of work")

    if row["Sick_Days"] <= 5:
        strengths.append("Good attendance consistency")

    if not strengths:
        strengths.append("Maintaining baseline performance")

    return strengths


standard_df["strengths"] = standard_df.apply(
    identify_strengths,
    axis=1
)




#####################

def update_explanation_with_strengths(row):
    explanation = row["employee_explanation"].copy()
    explanation["strengths"] = row["strengths"]
    return explanation


standard_df["employee_explanation"] = standard_df.apply(
    update_explanation_with_strengths,
    axis=1
)



######################

def answer_employee_question(employee_row, question):
    question_lower = question.lower()

    explanation = employee_row["employee_explanation"]

    if "improve" in question_lower or "focus" in question_lower:
        return explanation["recommended_actions"]

    if "why" in question_lower or "affected" in question_lower:
        return explanation["drivers"]

    if "status" in question_lower:
        return explanation["status"]

    if "doing well" in question_lower or "strength" in question_lower:
        return explanation["strengths"]

    return "Sorry, I can only answer questions about your performance, reasons, strengths, or improvement actions."



# Pick one employee to simulate the portal
emp = standard_df.iloc[0]

answer_employee_question(emp, "What am I doing well?")




## AI Layer (LLM rewrites the deterministic explanation)
import json

def build_llm_payload(employee_row, user_question=None):
    """
    Create a structured payload that the LLM can safely rewrite.
    The LLM must not invent numbers or new facts.
    """
    payload = {
        "employee_id": int(employee_row["employee_id"]),
        "sector": str(employee_row["sector"]),
        "role": str(employee_row["role"]),
        "date": str(employee_row["date"]),
        "performance_status": str(employee_row["performance_status"]),
        "metrics": {
            "output_score": float(employee_row["output_score"]),
            "quality_score": float(employee_row["quality_score"]),
            "development_score": float(employee_row["development_score"]),
            "sick_days": float(employee_row["Sick_Days"]),
        },
        "drivers": employee_row["performance_drivers"],
        "strengths": employee_row["employee_explanation"].get("strengths", []),
        "recommended_actions": employee_row["employee_explanation"].get("recommended_actions", []),
        "question": user_question
    }
    return payload



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
"""


##### Testing llm

from google.colab import userdata
import os

# Retrieve the API key from Colab Secrets.
# To add the secret:
# 1. Click on the 'key' icon (Secrets) in the left sidebar of Colab.
# 2. Click 'Add new secret'.
# 3. For 'Name', enter OPENROUTER_API_KEY (case-sensitive).
# 4. For 'Value', paste your actual OpenRouter API key.
# 5. Make sure 'Notebook access' is checked for this notebook.
token = userdata.get("llm")

if token is None:
    raise ValueError("OPENROUTER_API_KEY not found in Colab Secrets. Please add it as instructed in the comments.")

os.environ["OPENROUTER_API_KEY"] = token

print("OpenRouter key loaded:", token[:8], "...")

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://colab.research.google.com",
        "X-Title": "Employee Performance AI"
    }
)


import requests

headers = {
    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"
}

resp = requests.get(
    "https://openrouter.ai/api/v1/models",
    headers=headers
)

models = resp.json()["data"]
len(models), [m["id"] for m in models[:15]]


try:
    resp = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=20
    )
    print("MODEL WORKS ✅")
    print(resp.choices[0].message.content)
except Exception as e:
    print("MODEL FAILED ❌", e)


###AI Integration

def llm_rewrite(payload: dict) -> str:
    prompt = (
        "You are an expert performance coach.\n\n"
        "TASK:\n"
        "Write a clear, supportive performance feedback message for the employee.\n\n"
        "RULES:\n"
        "- Use ONLY the data provided below\n"
        "- Do NOT invent metrics, comparisons, or outcomes\n"
        "- Be constructive and encouraging\n"
        "- Do NOT mention other employees\n\n"
        "FORMAT YOUR RESPONSE EXACTLY AS:\n"
        "Status:\n"
        "What happened:\n"
        "Why:\n"
        "Strengths:\n"
        "What to do next:\n\n"
        "EMPLOYEE DATA:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Now write the response."
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=400
    )

    text = response.choices[0].message.content.strip()
    text = text.replace("[/s]", "").strip()
    return text if text else "[AI returned empty response]"


#Testing
emp = standard_df.iloc[0]
payload = build_llm_payload(emp, user_question="How can I improve?")

try:
    print("=== AI OUTPUT ===")
    print(llm_rewrite(payload))
except Exception as e:
    print("AI not used:", e)
    print("\n--- Fallback output ---\n")
    print(deterministic_format(payload))


# Defining users


def get_user_context(user_id: str):
    """
    Determines whether a user is an employee or manager.
    """

    if user_id.startswith("EMP"):
        return {
            "role": "employee",
            "employee_id": int(user_id.replace("EMP-", ""))
        }

    if user_id.startswith("MGR"):
        return {
            "role": "manager"
        }

    raise ValueError("Unknown user role")


##########################################

##############employee requests ###################
def handle_employee_request(user_context, question):
    emp_id = user_context["employee_id"]

    emp_row = standard_df[
        standard_df["employee_id"] == emp_id
    ].iloc[0]

    return answer_employee_question(emp_row, question)



######### employers requests
def handle_manager_request(user_context, question=None):
    return (
        "Manager portal active.\n"
        "You will be able to see team summaries, rankings, "
        "and coaching insights here."
    )



################################### Central router#######################

def route_request(user_id: str, question: str = None):
    user_context = get_user_context(user_id)

    if user_context["role"] == "employee":
        return handle_employee_request(user_context, question)

    if user_context["role"] == "manager":
        return handle_manager_request(user_context, question)


# Simulate employee login
print("=== EMPLOYEE VIEW ===")
print(
    route_request(
        user_id="EMP-1",
        question="What should I focus on next month?"
    )
)

print("\n=== MANAGER VIEW ===")
print(
    route_request(
        user_id="MGR-01",
        question="Who needs support?"
    )
)


######### Manager analytics summary

def manager_team_insights(df):
    return {
        "team_size": len(df),
        "status_distribution": df["performance_status"].value_counts().to_dict(),
        "avg_output_score": round(df["output_score"].mean(), 2),
        "common_issues": (
            df.explode("performance_drivers")["performance_drivers"]
              .value_counts()
              .head(3)
              .to_dict()
        )
    }

def manager_coaching_actions(df):
    actions = []

    for _, row in df.iterrows():
        if row["performance_status"] == "Needs Improvement":
            actions.append({
                "employee_id": row["employee_id"],
                "focus_areas": row["performance_drivers"]
            })

    return actions

def manager_ai_summary(df):
    insights = manager_team_insights(df)

    # Limit coaching priorities to avoid exceeding context length
    # Filter for 'Needs Improvement' and sort by output_score (lowest first)
    # Take only the top N (e.g., 20) for the LLM payload
    needs_improvement_subset = df[df["performance_status"] == "Needs Improvement"] \
                                 .sort_values(by="output_score", ascending=True) \
                                 .head(20) # Limiting to 20 employees

    coaching_priorities_for_llm = []
    for _, row in needs_improvement_subset.iterrows():
        coaching_priorities_for_llm.append({
            "employee_id": row["employee_id"],
            "focus_areas": row["performance_drivers"]
        })

    payload = {
        "team_insights": insights,
        "coaching_priorities": coaching_priorities_for_llm
    }

    prompt = (
        "You are an HR analytics advisor.\n\n"
        "TASK:\n"
        "Write a management summary for a private manager dashboard.\n\n"
        "RULES:\n"
        "- Do NOT shame individuals\n"
        "- Focus on coaching and support\n"
        "- Be concise and actionable\n\n"
        "FORMAT:\n"
        "Team Overview:\n"
        "Key Issues:\n"
        "Who Needs Support:\n"
        "Recommended Coaching Actions:\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}\n\n"
        "Now write the summary."
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )

    text = response.choices[0].message.content.strip()
    return text.replace("[/s]", "") if text else "[AI returned empty response]"


##### Test manager view#######

insights = manager_team_insights(standard_df)
insights

{
 'team_size': 100,
 'status_distribution': {
     'Needs Improvement': 42,
     'On Track': 38,
     'High Performer': 20
 },
 'avg_output_score': 46.7,
 'common_issues': {
     'Lower productivity relative to workload': 18,
     'High sick days affecting consistency': 11
 }
}



def manager_private_ranking(df):
    return df[[
        "employee_id",
        "performance_status",
        "output_score",
        "quality_score",
        "development_score"
    ]].sort_values(by="output_score", ascending=False)



manager_private_ranking(standard_df).head(10)

manager_coaching_actions(standard_df)[:5]


def format_manager_numbers(insights):
    return {
        "team_size": insights["team_size"],
        "needs_improvement_pct": round(
            100 * insights["status_distribution"].get("Needs Improvement", 0) / insights["team_size"], 2
        ),
        "on_track_pct": round(
            100 * insights["status_distribution"].get("On Track", 0) / insights["team_size"], 2
        ),
        "avg_output_score": insights["avg_output_score"],
        "common_issues": insights["common_issues"]
    }


def build_manager_ai_payload(df):
    insights = manager_team_insights(df)
    formatted = format_manager_numbers(insights)

    return {
        "team_size": formatted["team_size"],
        "needs_improvement_pct": formatted["needs_improvement_pct"],
        "on_track_pct": formatted["on_track_pct"],
        "average_output_score": formatted["avg_output_score"],
        "most_common_issues": formatted["common_issues"]
    }


def manager_ai_summary(df):
    payload = build_manager_ai_payload(df)


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
        "- Do NOT list every employee\n"
        "- Base insights on patterns, not raw data\n\n"
        "FORMAT:\n"
        "Team Overview:\n"
        "Key Issues:\n"
        "Who Needs Support (high level):\n"
        "Recommended Coaching Actions:\n\n"
        "TEAM SUMMARY DATA:\n"
        f"{json.dumps(payload, indent=2)}\n\n"
        "Now write the summary."
    )

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=400
    )

    text = response.choices[0].message.content
    return clean_ai_text(text) if text else "[AI returned empty response]"

def clean_ai_text(text: str) -> str:
    for token in ["[B_INST]", "[/B_INST]", "<s>", "</s>", "[/OUT]", "[OUT]"]:
        text = text.replace(token, "")
    return text.strip()

print("=== MANAGER AI SUMMARY ===")
print(manager_ai_summary(standard_df))

