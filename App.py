"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine_AP.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated track only) -> execute_query
    -> generate_response  (or escalate to a human analyst)

Required files next to this app.py:
    credit_risk_portfolio.db   (SQLite database, opened READ-ONLY)
    test_queries.csv           (ground-truth test cases, optional but recommended)

Credentials (first one found wins):
    1. Streamlit secrets  (.streamlit/secrets.toml or the Streamlit Cloud "Secrets" box)
         OPENAI_API_KEY  = "sk-..."
         OPENAI_API_BASE = "https://..."      # optional
    2. Environment variables OPENAI_API_KEY / OPENAI_API_BASE (or OPENAI_BASE_URL)
    3. config.json next to app.py  {"OPENAI_API_KEY": "...", "OPENAI_API_BASE": "..."}
    4. API key typed into the sidebar (kept in memory for the session only)
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent
DB_PATH = os.environ.get("DB_PATH", str(APP_DIR / "credit_risk_portfolio.db"))
TEST_QUERIES_PATH = os.environ.get("TEST_QUERIES_PATH", str(APP_DIR / "test_queries.csv"))
CONFIG_JSON_PATH = os.environ.get("CONFIG_JSON_PATH", str(APP_DIR / "config.json"))
AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", str(APP_DIR / "audit_log.jsonl"))  # set to "" to disable


# ---------------------------------------------------------------------------
# Database schema provided to the LLM in the prompts (unchanged from notebook)
# ---------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ---------------------------------------------------------------------------
# Verified Query Template Library (SQL unchanged from notebook)
# ---------------------------------------------------------------------------
# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_millions,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_millions
FROM loan_master l
LEFT JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_millions DESC
"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """
SELECT
    loan_category,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC;
"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage;
"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS average_provision_coverage_ratio
FROM provisioning p
LEFT JOIN loan_master l ON p.loan_account_number = l.loan_account_number
LEFT JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY average_provision_coverage_ratio DESC;
"""

# VQ5: Top 10 Loan Exposures
sql_5 = """
SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    l.asset_classification
FROM loan_master l
LEFT JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY total_outstanding DESC
LIMIT 10;
"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """
SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL AND group_name <> ''
GROUP BY group_name
ORDER BY total_outstanding_millions DESC
LIMIT 5;
"""

# VQ7: All Overdue Loan Accounts
sql_7 = """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_millions,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
LEFT JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;
"""

# VQ8: DPD Bucket Distribution
sql_8 = """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        WHEN days_past_due > 90 THEN '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END
"""

# VQ9: Latest Rating Downgrades
sql_9 = """
SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30' AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;
"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;
"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },
    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },
    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },
    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },
    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },
    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },
    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },
    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },
    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },
    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# ---------------------------------------------------------------------------
# Configuration / credentials helpers
# ---------------------------------------------------------------------------
def get_setting(name, default=None):
    """Look up a setting in Streamlit secrets, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, default)


def load_credentials():
    """Return (api_key, api_base) from secrets / env / config.json (in that order)."""
    api_key = get_setting("OPENAI_API_KEY")
    api_base = get_setting("OPENAI_API_BASE") or get_setting("OPENAI_BASE_URL")

    if (not api_key or not api_base) and os.path.exists(CONFIG_JSON_PATH):
        try:
            with open(CONFIG_JSON_PATH, "r") as file:
                config = json.load(file)
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass
    return api_key, api_base


@st.cache_resource(show_spinner=False)
def build_llms(api_key, api_base):
    """Create the generation LLM and the evaluator LLM (same models as the notebook)."""
    kwargs = {"api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    _llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, **kwargs)
    _evaluator_llm = ChatOpenAI(model='gpt-4o', temperature=0, **kwargs)
    return _llm, _evaluator_llm


@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    """Read-only SQLite connection (mode=ro prevents any write operation)."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_data(show_spinner=False)
def load_ground_truth(path):
    """Load test_queries.csv (ground truth) if it exists."""
    if path and os.path.exists(path):
        return pd.read_csv(path)
    return None


# Module-level LLM handles used by the tool functions below (assigned in main()).
llm = None
evaluator_llm = None


# ---------------------------------------------------------------------------
# Tool 1: Intent Classification
# ---------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are a query router for a financial operations analytics system. Your job is to decide whether a business user's question can be answered by one of the pre-approved query templates, or whether it needs fresh SQL generation.

### INPUT
User Question:
{user_question}

Available Verified Query Templates:
{library_descriptions}

### INSTRUCTIONS
1. Read the user question carefully and identify the analytical intent.
2. Compare the intent against each template description.
3. Match on semantic meaning, not exact wording.
4. If a template genuinely answers the question, return that template ID.
5. If no template covers the question, return null for the query_id and set the route to generated.
6. Be careful about shape of answer

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ---------------------------------------------------------------------------
# Tool 2: Query Generation
# ---------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

### ROLE
You are a senior SQL developer specializing in Financial operations analytics on a SQLite database containing a commercial lending portfolio.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Write a single SQL query that answers the user question using only the provided schema.
2. The query must be read-only. Use SELECT (or WITH ... SELECT). Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns listed in the schema. Do not invent columns.
4. Resolve asset classification and delinquency terms correctly (for example, Non-Performing Assets (NPAs) are defined as: asset_classification IN ('Substandard', 'Doubtful', 'Loss')).
5. Ensure the query is SQLite compatible.
6. In SQLite, never subtract DATE() or date columns directly (e.g. DATE(a)-DATE(b)) — it silently returns 0; always use julianday(a)-julianday(b) for day differences.
7. Alias numeric columns clearly with suffixes indicating units (e.g., _millions for millions of dollars, _count for counts, _percent or _pct for percentages) so the results are business-readable.

### OUTPUT
Return ONLY the SQL query, with no markdown code blocks, no comments, and no explanation.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ---------------------------------------------------------------------------
# Tool 3: Query Validation (five checks)
# ---------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    # (strip surrounding whitespace first so a trailing ';' + newline, as in VQ2-VQ7 and VQ9, is not
    #  mistaken for a second statement - the notebook's version missed this case)
    if ';' in candidate_sql.strip().rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select', 'from', 'where', 'and', 'or', 'group', 'by', 'order', 'having', 'limit', 'join', 'on', 'as', 'case',
                    'when', 'then', 'else', 'end', 'sum', 'count', 'avg', 'min', 'max', 'round', 'desc', 'asc', 'left', 'right',
                    'inner', 'outer', 'distinct', 'null', 'is', 'not', 'in', 'like', 'with', 'union', 'all', 'between', 'coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

### ROLE
You are a senior data validator. Your job is to check whether a SQL query
correctly answers a business user's question about Financial operations.

### CONTEXT
{track_context}

### INPUT
User Question: {user_question}

Candidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns?
2. Does it apply the right aggregations and groupings?
3. Does it handle the requested business definitions correctly?
4. Does it resolve named entities correctly?
5. Does it return the right shape of answer?
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            # Wrap in a subquery so templates that already end in ';' or contain their own LIMIT
            # (e.g. VQ5, VQ6) can still be probed for their output columns.
            expected_body = expected_sql.strip().rstrip(';').rstrip()
            actual_body = candidate_sql.strip().rstrip(';').rstrip()
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_body}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({actual_body}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ---------------------------------------------------------------------------
# Tool 4: Retry Generation
# ---------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""
### ROLE
You are a senior SQL developer fixing a query that failed validation.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Fix only the specific issue identified by the validation error.
2. Preserve the original intent of the query.
3. The revised SQL must be read-only SELECT (or WITH ... SELECT).
4. Use only tables and columns from the schema.
5. Ensure the query is SQLite compatible.

### OUTPUT
Return ONLY the corrected SQL, with no markdown code blocks, no comments, and no explanation.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ---------------------------------------------------------------------------
# Tool 5: Query Execution (deterministic, read-only connection)
# ---------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ---------------------------------------------------------------------------
# Tool 6: Response Generation
# ---------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

### ROLE
You are a Credit Risk Analytics specialist writing a concise, professional risk narrative response for a commercial lending portfolio query.

### INPUT
User Question: {user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Answer the user's specific risk question directly. Do not dump the entire table.
2. If the user asked about a specific sector, borrower, rating grade, or reporting quarter, highlight only those relevant values.
3. Provide context from other rows only when it adds value (for example, comparing exposure or ranking top borrowers).
4. State exact numbers from the data. Do not round beyond what is shown.
5. Flag anything notable, such as high concentration in sensitive sectors, rapid increases in Stage 3 exposure, or notable rating downgrades.
6. Use clear, professional risk-management language suitable for a Chief Risk Officer (CRO) or Credit Committee memo.
7. Keep the response focused. Two to four sentences for simple questions, up to a short paragraph for complex ones.
8. State the unit for every number, inferred from its column name or context (e.g., millions as '$X million', percentages or rates as 'X%', counts as 'X accounts' or 'X loans'). Never state a bare number without its logical business unit.

### OUTPUT
Return ONLY the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records intermediate pipeline stages in log['trace']
                      (the notebook printed these; the app displays them in the UI).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'warnings': [],
        'trace': []
    }

    def emit(message):
        if verbose:
            log['trace'].append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route'] == 'verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)
    log['warnings'] = exec_result['warnings']

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[5] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ---------------------------------------------------------------------------
# Audit trail helpers
# ---------------------------------------------------------------------------
def build_audit_record(result):
    """Flatten a pipeline result into an audit record (no raw result data is stored)."""
    gate = result.get('gate_result') or {}
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'user_question': result.get('user_question'),
        'route': result.get('route'),
        'query_id': result.get('query_id'),
        'match_reason': result.get('match_reason'),
        'retry_used': result.get('retry_used'),
        'escalated': result.get('escalated'),
        'validation_passed': gate.get('passed'),
        'validation_failed_check': gate.get('failed_check'),
        'validation_details': gate.get('details'),
        'confidence': result.get('confidence'),
        'row_count': result.get('row_count'),
        'candidate_sql': result.get('candidate_sql'),
        'executed_sql': result.get('executed_sql'),
        'narrative': result.get('narrative'),
    }


def record_audit(result):
    """Store the audit record in the session and append it to the JSONL audit file (best effort)."""
    record = build_audit_record(result)
    st.session_state.setdefault('audit_history', []).append(record)
    if AUDIT_LOG_PATH:
        try:
            with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # file system may be read-only / ephemeral; session history still holds the record


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def format_confidence(confidence):
    if isinstance(confidence, (int, float)):
        return f"{confidence:.0%}"
    return str(confidence) if confidence is not None else "n/a"


def render_result(result, key_prefix="result"):
    """Render narrative, confidence, SQL, raw data and pipeline details for one pipeline result."""
    if result['escalated']:
        st.warning("⚠️ This question could not be reliably answered by the engine and has been "
                   "**escalated to a human analyst**.")
        st.write(result['narrative'])
    else:
        st.subheader("Answer")
        st.write(result['narrative'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", format_confidence(result['confidence']))
    col2.metric("Route", (result['route'] or "n/a").capitalize())
    col3.metric("Query ID", result['query_id'] or "—")
    col4.metric("Rows returned", result['row_count'] if result['row_count'] is not None else "—")

    if result['match_reason']:
        st.caption(f"Routing reason: {result['match_reason']}")
    if result['retry_used']:
        st.info("The first generated SQL failed validation; the engine retried once with a corrected query.")
    for warning in result.get('warnings', []):
        st.warning(warning)

    sql_to_show = result['executed_sql'] or result['candidate_sql']
    st.markdown("**SQL query used**" if not result['escalated'] else "**Candidate SQL (failed validation, for analyst review)**")
    st.code(sql_to_show or "-- no SQL available", language="sql")

    if result['dataframe'] is not None:
        st.markdown("**Raw data returned**")
        st.dataframe(result['dataframe'], hide_index=True)
        st.download_button(
            "Download result as CSV",
            data=result['dataframe'].to_csv(index=False).encode('utf-8'),
            file_name="query_result.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
        )

    with st.expander("Validation gate & pipeline trace"):
        gate = result.get('gate_result') or {}
        st.json({
            'passed': gate.get('passed'),
            'failed_check': gate.get('failed_check'),
            'details': gate.get('details'),
            'relevance_confidence': gate.get('relevance_confidence'),
        })
        if result['trace']:
            st.code("\n".join(result['trace']), language="text")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    global llm, evaluator_llm

    st.title("🏦 Credit Risk Query Engine")
    st.caption(
        "Northbridge Bank · Ask routine commercial lending portfolio questions in plain English. "
        "Answers come from pre-approved verified queries or validated, read-only generated SQL, "
        "with the SQL, raw data and a confidence score shown every time."
    )

    # ---- Sidebar: credentials, status, library ----
    api_key, api_base = load_credentials()
    with st.sidebar:
        st.header("Configuration")
        if not api_key:
            api_key = st.text_input("OpenAI API key", type="password",
                                    help="Not found in secrets, environment or config.json.")
            api_base_input = st.text_input("OpenAI API base URL (optional)", value=api_base or "")
            api_base = api_base_input or api_base

        db_ok = os.path.exists(DB_PATH)
        st.write(f"Database: {'✅ found' if db_ok else '❌ missing'} (`{os.path.basename(DB_PATH)}`)")
        st.write(f"LLM credentials: {'✅ set' if api_key else '❌ missing'}")
        st.write("Mode: 🔒 read-only")

        st.divider()
        st.subheader("Verified query library")
        for qid, entry in verified_query_library.items():
            st.markdown(f"**{qid}** — {entry['description']}")

    if not db_ok:
        st.error(f"Database file not found at `{DB_PATH}`. Place `credit_risk_portfolio.db` next to "
                 "`app.py` or set the `DB_PATH` environment variable.")
        st.stop()
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar (or configure Streamlit secrets) to continue.")
        st.stop()

    llm, evaluator_llm = build_llms(api_key, api_base)
    conn = get_connection(DB_PATH)
    ground_truth = load_ground_truth(TEST_QUERIES_PATH)

    tab_ask, tab_library, tab_db, tab_eval, tab_audit = st.tabs(
        ["💬 Ask a question", "📚 Verified Query Library", "🗄️ Database & Schema", "✅ Evaluation", "🧾 Audit Trail"]
    )

    # ------------------------------------------------------------------ Ask
    with tab_ask:
        example_questions = ["How much of our book is in real estate, and how much is NPA?"]
        if ground_truth is not None and 'User Query' in ground_truth.columns:
            example_questions = list(dict.fromkeys(ground_truth['User Query'].tolist() + example_questions))
        placeholder = "— choose an example question (optional) —"

        def use_example():
            choice = st.session_state.get('example_choice')
            if choice and choice != placeholder:
                st.session_state['question_input'] = choice

        st.selectbox("Example questions", [placeholder] + example_questions,
                     key="example_choice", on_change=use_example)
        st.text_area("Your question", key="question_input", height=100,
                     placeholder="e.g. Which sector has the highest NPA exposure?")

        if st.button("Run query", type="primary"):
            question = st.session_state.get('question_input', '').strip()
            if not question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Routing, validating and running your query..."):
                    try:
                        result = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
                        record_audit(result)
                        st.session_state['last_result'] = result
                    except Exception as e:
                        st.session_state.pop('last_result', None)
                        st.error(f"The pipeline failed unexpectedly and the question should be escalated to an analyst. Error: {e}")

        if st.session_state.get('last_result'):
            st.divider()
            render_result(st.session_state['last_result'], key_prefix="ask")

    # -------------------------------------------------------------- Library
    with tab_library:
        st.subheader("Verified Query Template Library")
        st.write(f"{len(verified_query_library)} pre-approved SQL templates. Each runs without modification; "
                 "the LLM then writes a focused narrative from the full result.")
        for qid, entry in verified_query_library.items():
            with st.expander(f"{qid}: {entry['description']}"):
                st.code(entry['sql'].strip(), language="sql")

    # ------------------------------------------------------------- Database
    with tab_db:
        st.subheader("Database schema (as provided to the LLM)")
        st.code(database_schema.strip(), language="text")
        st.subheader("Table previews (first 3 rows)")
        for table_name in ['borrower_rating', 'loan_master', 'provisioning', 'sector_master']:
            with st.expander(table_name):
                try:
                    st.dataframe(pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn), hide_index=True)
                except Exception as e:
                    st.error(f"Could not read {table_name}: {e}")

    # ----------------------------------------------------------- Evaluation
    with tab_eval:
        st.subheader("Evaluation against ground truth")
        if ground_truth is None:
            st.info(f"`test_queries.csv` not found at `{TEST_QUERIES_PATH}`. Add it next to `app.py` to enable evaluation.")
        else:
            st.write("Runs every test case in `test_queries.csv` through the pipeline and compares the selected "
                     "route and query ID with the expected values.")
            st.dataframe(ground_truth, hide_index=True)

            if st.button("Run evaluation on all test cases"):
                test_results = []
                progress = st.progress(0.0, text="Starting evaluation...")
                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    progress.progress(i / len(ground_truth), text=f"Running {gt['Test Case']} ({i + 1}/{len(ground_truth)})...")
                    try:
                        tr = run_pipeline(gt['User Query'], conn, verified_query_library, database_schema, verbose=True)
                        record_audit(tr)
                    except Exception as e:
                        tr = {'route': 'ERROR', 'query_id': None, 'confidence': None, 'row_count': None,
                              'narrative': f"Pipeline error: {e}", 'escalated': True, 'dataframe': None,
                              'executed_sql': None, 'candidate_sql': None, 'match_reason': None,
                              'retry_used': False, 'warnings': [], 'trace': [], 'gate_result': None}
                    test_results.append(tr)
                progress.progress(1.0, text="Evaluation complete")
                st.session_state['eval_results'] = test_results

            if st.session_state.get('eval_results'):
                test_results = st.session_state['eval_results']
                evaluation_rows = []

                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    tr = test_results[i]

                    evaluation_rows.append({
                        'Test Case': gt['Test Case'],
                        'Expected Route': gt['Expected Route'],
                        'Actual Route': tr['route'],
                        'Route Match': tr['route'] == gt['Expected Route'],
                        'Expected Query ID': gt['Expected Query ID'],
                        'Actual Query ID': tr['query_id'],
                        'Query ID Match': (
                            pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                        ) or tr['query_id'] == gt['Expected Query ID'],
                        'Confidence': tr['confidence'],
                        'Rows Returned': tr['row_count']
                    })

                evaluation_df = pd.DataFrame(evaluation_rows)

                path_accuracy = evaluation_df['Route Match'].mean() * 100

                verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
                query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

                average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

                m1, m2, m3 = st.columns(3)
                m1.metric("Selected Path Accuracy", f"{path_accuracy:.1f}%")
                m2.metric("Selected Query Accuracy", "n/a" if pd.isna(query_accuracy) else f"{query_accuracy:.1f}%")
                m3.metric("Average Confidence Score", "n/a" if pd.isna(average_confidence) else f"{average_confidence:.2f}")

                st.dataframe(evaluation_df, hide_index=True)

                st.markdown("**Per-test-case results**")
                for i, (_, gt) in enumerate(ground_truth.iterrows()):
                    with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                        if 'Expected Answer' in ground_truth.columns:
                            st.caption(f"Expected answer: {gt['Expected Answer']}")
                        render_result(test_results[i], key_prefix=f"eval_{i}")

    # ---------------------------------------------------------------- Audit
    with tab_audit:
        st.subheader("Audit trail (this session)")
        history = st.session_state.get('audit_history', [])
        if not history:
            st.info("No queries have been run in this session yet.")
        else:
            audit_df = pd.DataFrame(history)
            st.dataframe(audit_df, hide_index=True)
            st.download_button("Download audit trail (CSV)", audit_df.to_csv(index=False).encode('utf-8'),
                               file_name="audit_trail.csv", mime="text/csv")
            st.download_button("Download audit trail (JSON Lines)",
                               "\n".join(json.dumps(r, default=str) for r in history).encode('utf-8'),
                               file_name="audit_trail.jsonl", mime="application/json")
        if AUDIT_LOG_PATH:
            st.caption(f"Each run is also appended to `{os.path.basename(AUDIT_LOG_PATH)}` (best effort; "
                       "this file may be reset on hosted platforms such as Streamlit Community Cloud).")


main()
