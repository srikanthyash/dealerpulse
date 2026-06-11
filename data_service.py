"""
data_service.py — all fetch_* business-logic functions for DealerPulse.

Migration note: replaces every _session.sql(query).to_pandas() call in
DealerFinalVersion.py (~30 functions). The _session parameter is removed
from every function — queries are executed via athena_query() which also
applies Snowflake→Athena SQL dialect translations automatically.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from athena_client import athena_query, list_columns
from config_loader import get_config
from bedrock_client import load_yaml_from_s3

logger = logging.getLogger(__name__)
_DB: str = get_config()["athena"]["database"]

# ---------------------------------------------------------------------------
# Semantic model (replaces _load_local_semantic_model from local YAML file)
# Migration note: reads from S3 via bedrock_client instead of local filesystem
# ---------------------------------------------------------------------------

_semantic_cache: Optional[dict] = None


def _load_semantic_model() -> dict:
    """
    Load dealer semantic model from S3 config prefix.
    Migration note: replaces _load_local_semantic_model() which opened
    'DEALER_SEMANTIC_MODEL_V4.yml' from the local filesystem.
    """
    global _semantic_cache
    if _semantic_cache is not None:
        return _semantic_cache
    _semantic_cache = load_yaml_from_s3()   # uses s3.yaml_filename from config
    return _semantic_cache or {}


def get_expr_column(table_name: str, logical_column_name: str) -> str:
    """
    Return the physical column expression for a logical column from the
    semantic model. Falls back to logical_column_name unchanged.
    Source line 3665.
    """
    model = _load_semantic_model()
    for tbl in model.get("tables", []):
        if tbl.get("name") == table_name:
            for dim in tbl.get("dimensions", []):
                if dim.get("name") == logical_column_name:
                    return dim.get("expr", logical_column_name)
    return logical_column_name


# ---------------------------------------------------------------------------
# Filter / date helpers (no DB dependency — kept as-is from source)
# ---------------------------------------------------------------------------

def resolve_date_range(filters: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    """
    Return (from_date, to_date) from a filters dict.
    Source line 84.
    """
    if not filters:
        filters = {}

    if "from_date" in filters and "to_date" in filters:
        f = filters["from_date"]
        t = filters["to_date"]
        if isinstance(f, datetime):
            f = f.date()
        if isinstance(t, datetime):
            t = t.date()
        return f, t

    date_range_str  = filters.get("date_range",   "Last 30 Days")
    time_period_str = filters.get("time_period",  "Current Period")
    today = date.today()

    if date_range_str == "All Dates":
        cur_from = date(1900, 1, 1)
        cur_to   = date(9999, 12, 31)
    else:
        range_map = {
            "Last 7 Days":   7,
            "Last 30 Days":  30,
            "Last 90 Days":  90,
            "Last 6 Months": 180,
            "Year to Date":  (today - date(today.year, 1, 1)).days or 0,
        }
        period_days = range_map.get(date_range_str, 30)
        cur_to   = today
        cur_from = today - timedelta(days=period_days - 1)

    prev_to   = cur_from - timedelta(days=1)
    prev_from = prev_to  - timedelta(days=period_days - 1)
    yoy_to    = cur_to   - timedelta(days=365)
    yoy_from  = cur_from - timedelta(days=365)

    if time_period_str == "Previous Period":
        return prev_from, prev_to
    if time_period_str == "YoY Comparison":
        return yoy_from, yoy_to
    return cur_from, cur_to


def dealer_filter_clause(view_name: str, filters: Optional[dict]) -> str:
    """
    Return a SQL AND-clause filtering by dealer name.
    Source line 3681.
    """
    if not filters or filters.get("dealer") in (None, "All Dealers"):
        return ""
    col = get_expr_column(view_name, "DEALER_NAME")
    dealer = str(filters["dealer"]).replace("'", "''")
    return f" AND {col} = '{dealer}'"


def lineage_filter_clause(filters: Optional[dict]) -> str:
    """
    Return extra SQL filter clauses for transaction lineage queries.
    Source line ~4561.
    """
    if not filters:
        return ""
    clause = ""
    if filters.get("transaction_id"):
        t = str(filters["transaction_id"]).replace("'", "''")
        clause += f" AND TRANSACTION_ID = '{t}'"
    if filters.get("paid") in ("Y", "N"):
        clause += f" AND PAID_FLAG = '{filters['paid']}'"
    if filters.get("warranty_status"):
        ws = str(filters["warranty_status"]).upper().replace("'", "''")
        clause += f" AND UPPER(WARRANTY_STATUS) = '{ws}'"
    if filters.get("invoice_status") and filters["invoice_status"] != "All":
        inv = str(filters["invoice_status"]).replace("'", "''")
        clause += f" AND UPPER(INVOICE_STATUS) = UPPER('{inv}')"
    return clause


# ---------------------------------------------------------------------------
# Schema validation
# Migration note: replaces _session.sql("SELECT * FROM view LIMIT 1")
# with Athena information_schema column lookup via athena_client.list_columns()
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def validate_view_schemas() -> bool:
    """
    Validate that all required KPI views exist and have expected columns.
    Migration note: replaces validate_view_schemas(_session) which used
    SELECT * FROM view LIMIT 1 to probe columns.
    """
    required = {
        "vw_gross_profit_margin":              ["DEALER_NAME", "PERIOD_YEAR", "PERIOD_MONTH", "GROSS_PROFIT_MARGIN_PCT"],
        "vw_average_repair_turnaround_time":   ["DEALER_NAME", "PERIOD_YEAR", "PERIOD_MONTH", "AVG_TURNAROUND_HOURS"],
        "vw_order_lead_time":                  ["DEALER_NAME", "PERIOD_YEAR", "PERIOD_MONTH", "AVG_ORDER_LEAD_TIME_DAYS"],
        "vw_dealer_revenue_growth":            ["DEALER_NAME", "PERIOD_YEAR", "PERIOD_MONTH", "REVENUE_GROWTH_MOM_PERCENT"],
    }
    missing_columns: Dict[str, list] = {}
    for view_name, expected in required.items():
        try:
            cols_df = list_columns(view_name)
            actual  = [c.upper() for c in cols_df["column_name"].tolist()] if not cols_df.empty else []
            missing = [c for c in expected if c.upper() not in actual]
            if missing:
                missing_columns[view_name] = missing
        except Exception as exc:
            logger.warning("[data_service] validate_view_schemas failed for %s: %s", view_name, exc)

    if missing_columns:
        st.error("**Schema Mismatch**: Athena views are missing expected columns.")
        for view, cols in missing_columns.items():
            st.error(f"  - {view} missing: {', '.join(cols)}")
        return False
    return True


# ---------------------------------------------------------------------------
# fetch_dealer_health_scores
# Migration note: replaces fetch_dealer_health_scores(_session, filters)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_dealer_health_scores(filters=None) -> pd.DataFrame:
    """
    Fixed health score calculation with SLA-based thresholds.
    Migration note: _session parameter removed; uses athena_query().
    SQL dialect translated automatically by athena_client._translate_sql().
    """
    MARGIN_SLA      = 30
    TAT_SLA_HOURS   = 48
    LEAD_TIME_SLA   = 7

    dealer_col_margin = get_expr_column("VW_GROSS_PROFIT_MARGIN",             "dealer_name")
    dealer_col_tat    = get_expr_column("VW_AVERAGE_REPAIR_TURNAROUND_TIME",  "dealer_name")
    dealer_col_lead   = get_expr_column("VW_ORDER_LEAD_TIME",                 "dealer_name")
    dealer_col_ccc    = get_expr_column("VW_CASH_CONVERSION_CYCLE",           "dealer_name")
    dealer_name_ccc   = get_expr_column("VW_CASH_CONVERSION_CYCLE",           "DEALER_NAME")

    try:
        margin_df = athena_query(f"""
            SELECT {dealer_col_margin} AS dealer_name, AVG(GROSS_PROFIT_MARGIN_PCT) AS MARGIN
            FROM {_DB}.VW_GROSS_PROFIT_MARGIN
            WHERE GROSS_PROFIT_MARGIN_PCT IS NOT NULL
            GROUP BY {dealer_col_margin}
        """)
        margin_df.columns = margin_df.columns.str.lower()
    except Exception:
        margin_df = pd.DataFrame()

    try:
        tat_df = athena_query(f"""
            SELECT {dealer_col_tat} AS dealer_name, AVG(AVG_TURNAROUND_HOURS) AS AVG_TAT
            FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME
            GROUP BY {dealer_col_tat}
        """)
        tat_df.columns = tat_df.columns.str.lower()
    except Exception:
        tat_df = pd.DataFrame()

    try:
        lead_df = athena_query(f"""
            SELECT {dealer_col_lead} AS dealer_name, AVG(AVG_ORDER_LEAD_TIME_DAYS) AS AVG_LEAD
            FROM {_DB}.VW_ORDER_LEAD_TIME
            GROUP BY {dealer_col_lead}
        """)
        lead_df.columns = lead_df.columns.str.lower()
    except Exception:
        lead_df = pd.DataFrame()

    dfs = [df for df in [margin_df, tat_df, lead_df] if not df.empty]
    if not dfs:
        return pd.DataFrame()

    merged = None
    for df in dfs:
        merged = df if merged is None else pd.merge(merged, df, on="dealer_name", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = merged.fillna(0)

    try:
        name_df = athena_query(f"""
            SELECT DISTINCT {dealer_col_ccc} AS dealer_name, {dealer_name_ccc} AS DEALER_NAME
            FROM {_DB}.VW_CASH_CONVERSION_CYCLE
            WHERE {dealer_col_ccc} IS NOT NULL AND {dealer_name_ccc} IS NOT NULL
        """)
        name_df.columns = [c.lower() for c in name_df.columns]
        if not name_df.empty:
            merged = pd.merge(merged, name_df, on="dealer_name", how="left", suffixes=("_x", ""))
    except Exception:
        pass

    def margin_score(m):
        if m >= 40: return 100
        if m >= 30: return 75
        if m >= 20: return 50
        if m > 0:   return 25
        return 0

    def tat_score(t):
        if t <= TAT_SLA_HOURS:          return 100
        if t <= TAT_SLA_HOURS * 1.25:   return 75
        if t <= TAT_SLA_HOURS * 1.5:    return 50
        if t <= TAT_SLA_HOURS * 2:      return 25
        return 0

    def lead_score(l):
        if l <= LEAD_TIME_SLA:      return 100
        if l <= LEAD_TIME_SLA * 1.5: return 75
        if l <= LEAD_TIME_SLA * 2:   return 50
        if l <= LEAD_TIME_SLA * 3:   return 25
        return 0

    m_col = next((c for c in ("margin", "MARGIN") if c in merged.columns), None)
    t_col = next((c for c in ("avg_tat", "AVG_TAT") if c in merged.columns), None)
    l_col = next((c for c in ("avg_lead", "AVG_LEAD") if c in merged.columns), None)

    merged["margin_score"] = merged[m_col].apply(margin_score) if m_col else 0
    merged["tat_score"]    = merged[t_col].apply(tat_score)    if t_col else 0
    merged["lead_score"]   = merged[l_col].apply(lead_score)   if l_col else 0

    merged["health_score"]    = merged["margin_score"] * 0.6 + merged["tat_score"] * 0.2 + merged["lead_score"] * 0.2
    merged["change_percent"]  = 0.0
    merged["last_updated"]    = datetime.now()

    if filters and filters.get("dealer") not in (None, "All Dealers"):
        merged = merged[merged["dealer_name"] == filters["dealer"]]

    out = merged[["dealer_name", "health_score", "change_percent", "last_updated"]].copy()
    out.columns = ["DEALER_NAME", "HEALTH_SCORE", "CHANGE_PERCENT", "LAST_UPDATED"]
    return out


# ---------------------------------------------------------------------------
# At-risk helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_at_risk_dealers(filters=None) -> Optional[int]:
    """Migration note: _session removed."""
    try:
        df = fetch_dealer_health_scores(filters)
        return int((df["HEALTH_SCORE"] < 50).sum()) if not df.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_at_risk_dealers_list(filters=None) -> pd.DataFrame:
    """Migration note: _session removed."""
    try:
        health_df = fetch_dealer_health_scores(filters)
        if health_df is None or health_df.empty:
            return pd.DataFrame()
        at_risk = health_df[health_df["HEALTH_SCORE"] < 50].copy().sort_values("HEALTH_SCORE")
        if at_risk.empty:
            return pd.DataFrame()
        repair_tat_val = fetch_repair_turnaround_time(filters)
        lead_time_val  = fetch_order_lead_time(filters)
        issues = []
        for _, row in at_risk.iterrows():
            if repair_tat_val and repair_tat_val > 48:
                issues.append("Service TAT breach")
            elif lead_time_val and lead_time_val > 7:
                issues.append("Order delays")
            else:
                issues.append("Performance at risk")
        at_risk["PRIMARY_ISSUE"] = issues
        return at_risk[["DEALER_NAME", "HEALTH_SCORE", "PRIMARY_ISSUE"]]
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# KPI scalar fetch functions
# All: _session removed, athena_query() used directly, SQL unchanged
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_order_fulfillment(filters=None, sla_days: int = 7) -> Optional[float]:
    """Migration note: _session removed. Source line 3947."""
    try:
        query = f"""
        SELECT COUNT(*) AS TOTAL,
               SUM(CASE WHEN AVG_ORDER_LEAD_TIME_DAYS <= {int(sla_days)} THEN 1 ELSE 0 END) AS ON_TIME
        FROM {_DB}.VW_ORDER_LEAD_TIME
        WHERE 1=1
        """
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and "from_date" in filters and "to_date" in filters:
            query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}'"
            query += f" AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        if result.empty or result.iloc[0]["TOTAL"] == 0:
            return None
        return 100.0 * int(result.iloc[0]["ON_TIME"]) / int(result.iloc[0]["TOTAL"])
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_avg_tat(filters=None) -> Optional[int]:
    """Migration note: _session removed. Source line 3975.f"""
    try:
        query = f"SELECT AVG(AVG_TURNAROUND_HOURS) AS AVG_TAT FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and "from_date" in filters and "to_date" in filters:
            fd, td = filters["from_date"], filters["to_date"]
            query += f" AND (PERIOD_YEAR > {fd.year} OR (PERIOD_YEAR = {fd.year} AND PERIOD_MONTH >= {fd.month}))"
            query += f" AND (PERIOD_YEAR < {td.year} OR (PERIOD_YEAR = {td.year} AND PERIOD_MONTH <= {td.month}))"
        result = athena_query(query)
        val = result.iloc[0]["AVG_TAT"] if not result.empty else None
        return int(val) if val is not None and not (isinstance(val, float) and val != val) else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_revenue_metrics(filters=None) -> Optional[dict]:
    """Migration note: _session removed. Source line 4003.f"""
    try:
        query = f"SELECT SUM(TOTAL_REVENUE) AS TOTAL_REVENUE, AVG(GROSS_PROFIT_MARGIN_PCT) AS AVG_MARGIN FROM {_DB}.VW_GROSS_PROFIT_MARGIN WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and "from_date" in filters and "to_date" in filters:
            fd, td = filters["from_date"], filters["to_date"]
            query += f" AND (PERIOD_YEAR > {fd.year} OR (PERIOD_YEAR = {fd.year} AND PERIOD_MONTH >= {fd.month}))"
            query += f" AND (PERIOD_YEAR < {td.year} OR (PERIOD_YEAR = {td.year} AND PERIOD_MONTH <= {td.month}))"
        result = athena_query(query)
        if not result.empty:
            return {
                "revenue": result.iloc[0]["TOTAL_REVENUE"] or 0,
                "margin":  result.iloc[0]["AVG_MARGIN"]    or 0,
            }
        return None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def fetch_sales_vs_target(filters=None) -> float:
    """Migration note: _session removed. Source line 4035.f"""
    try:
        query = f"SELECT SUM(TOTAL_REVENUE) AS TOTAL_REVENUE FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and filters.get("product") not in (None, "Product"):
            query += f" AND PRODUCT_CATEGORY = '{filters['product']}'"
        if filters and "from_date" in filters and "to_date" in filters:
            fd, td = filters["from_date"], filters["to_date"]
            query += f" AND (PERIOD_YEAR > {fd.year} OR (PERIOD_YEAR = {fd.year} AND PERIOD_MONTH >= {fd.month}))"
            query += f" AND (PERIOD_YEAR < {td.year} OR (PERIOD_YEAR = {td.year} AND PERIOD_MONTH <= {td.month}))"
        result = athena_query(query)
        val = result.iloc[0]["TOTAL_REVENUE"] if not result.empty else None
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def fetch_strategic_insights() -> pd.DataFrame:
    """Migration note: _session removed. Source line 4063."""
    try:
        query = f"""
        SELECT INSIGHT_TEXT, PRIORITY_LEVEL, DEALER_COUNT, IMPACT_PERCENTAGE
        FROM {_DB}.VW_STRATEGIC_INSIGHTS
        ORDER BY PRIORITY_LEVEL DESC
        """
        return athena_query(query)
    except Exception:
        return _generate_mock_insights()


@st.cache_data(ttl=3600)
def fetch_dealers() -> List[str]:
    """Migration note: _session removed. Source line 4291."""
    try:
        result = athena_query(f"""
            SELECT DISTINCT DEALER_NAME
            FROM {_DB}.VW_CASH_CONVERSION_CYCLE
            WHERE DEALER_NAME IS NOT NULL
            ORDER BY DEALER_NAME
        """)
        return result["DEALER_NAME"].dropna().tolist() if not result.empty else []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def fetch_cash_conversion_cycle(filters=None) -> float:
    """Migration note: _session removed. Source line 4308.f"""
    try:
        query = f"SELECT AVG(CCC) AS ccc_days FROM {_DB}.VW_CASH_CONVERSION_CYCLE WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_MONTH BETWEEN '{filters['from_date'].strftime('%Y-%m')}' AND '{filters['to_date'].strftime('%Y-%m')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["ccc_days"] if not result.empty else None
        return float(val) if val is not None and not (isinstance(val, float) and val != val) else 30.0
    except Exception:
        return 30.0


@st.cache_data(ttl=3600)
def fetch_repair_turnaround_time(filters=None) -> int:
    """Migration note: _session removed. Source line 4335.f"""
    try:
        query = f"SELECT AVG(AVG_TURNAROUND_HOURS) AS avg_turnaround_hours FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["avg_turnaround_hours"] if not result.empty else None
        return int(round(float(val))) if val is not None and not (isinstance(val, float) and val != val) else 24
    except Exception:
        return 24


def fetch_revenue_growth(filters=None) -> float:
    """Migration note: _session removed. Source line 4361.f"""
    try:
        query = f"SELECT AVG(REVENUE_GROWTH_MOM_PERCENT) AS revenue_growth_pct FROM {_DB}.VW_DEALER_REVENUE_GROWTH WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_MONTH BETWEEN '{filters['from_date'].strftime('%Y-%m')}' AND '{filters['to_date'].strftime('%Y-%m')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["revenue_growth_pct"] if not result.empty else None
        return float(val) if val is not None else 2.5
    except Exception:
        return 2.5


@st.cache_data(ttl=3600)
def fetch_gross_profit_margin(filters=None) -> float:
    """Migration note: _session removed. Source line 4384.f"""
    try:
        query = f"SELECT AVG(GROSS_PROFIT_MARGIN_PCT) AS gross_profit_margin_pct FROM {_DB}.VW_GROSS_PROFIT_MARGIN WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_MONTH BETWEEN '{filters['from_date'].strftime('%Y-%m')}' AND '{filters['to_date'].strftime('%Y-%m')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["gross_profit_margin_pct"] if not result.empty else None
        return float(val) if val is not None else 25.0
    except Exception:
        return 25.0


@st.cache_data(ttl=3600)
def fetch_sales_per_product_category(filters=None) -> float:
    """Migration note: _session removed. Source line 4407.f"""
    try:
        query = f"SELECT AVG(TOTAL_REVENUE) AS avg_revenue_per_product FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and filters.get("product") not in (None, "Product"):
            query += f" AND PRODUCT_CATEGORY = '{filters['product']}'"
        if filters and "from_date" in filters:
            query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["avg_revenue_per_product"] if not result.empty else None
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


@st.cache_data(ttl=3600)
def fetch_order_lead_time(filters=None) -> int:
    """Migration note: _session removed. Source line 4432.f"""
    try:
        query = f"SELECT AVG(AVG_ORDER_LEAD_TIME_DAYS) AS avg_lead_time FROM {_DB}.VW_ORDER_LEAD_TIME WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
        if filters and "from_date" in filters:
            query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["avg_lead_time"] if not result.empty else None
        return int(round(float(val))) if val is not None and not (isinstance(val, float) and val != val) else 0
    except Exception:
        return 0


@st.cache_data(ttl=3600)
def fetch_stock_availability(filters=None) -> float:
    """Migration note: _session removed. Source line 4461.f"""
    try:
        query = f"SELECT AVG(STOCK_AVAILABILITY_PCT) AS stock_availability_pct FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["stock_availability_pct"] if not result.empty else None
        return float(val) if val is not None else 85.0
    except Exception:
        return 85.0


@st.cache_data(ttl=3600)
def fetch_sales_volume(filters=None) -> int:
    """Migration note: _session removed. Source line 4484.f"""
    try:
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query = f"SELECT SUM(UNITS_SOLD) AS total_units_sold FROM {_DB}.VW_SALES_VOLUME WHERE DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in filters:
                query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
            col = "total_units_sold"
        else:
            query = f"SELECT AVG(UNITS_SOLD) AS avg_units_sold FROM {_DB}.VW_SALES_VOLUME WHERE 1=1"
            col = "avg_units_sold"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0].get(col) if not result.empty else None
        return int(round(float(val))) if val is not None and not (isinstance(val, float) and val != val) else 1000
    except Exception:
        return 1000


@st.cache_data(ttl=3600)
def fetch_contribution_margin(filters=None) -> float:
    """Migration note: _session removed. Source line 4516.f"""
    try:
        query = f"SELECT AVG(CONTRIBUTION_MARGIN_PCT) AS contribution_margin_pct FROM {_DB}.VW_DEALER_CONTRIBUTION_MARGIN WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["contribution_margin_pct"] if not result.empty else None
        return float(val) if val is not None else 20.0
    except Exception:
        return 20.0


@st.cache_data(ttl=3600)
def fetch_backorder_incidence(filters=None) -> float:
    """Migration note: _session removed. Source line 4539.f"""
    try:
        query = f"SELECT AVG(BACKORDER_INCIDENCE_PCT) AS backorder_incidence_pct FROM {_DB}.VW_BACKORDER_INCIDENCE WHERE 1=1"
        if filters and filters.get("dealer") not in (None, "All Dealers"):
            query += f" AND DEALER_NAME = '{filters['dealer']}'"
            if "from_date" in (filters or {}):
                query += f" AND PERIOD_START_DATE >= '{filters['from_date'].strftime('%Y-%m-%d')}' AND PERIOD_START_DATE <= '{filters['to_date'].strftime('%Y-%m-%d')}'"
        result = athena_query(query)
        result.columns = result.columns.str.lower()
        val = result.iloc[0]["backorder_incidence_pct"] if not result.empty else None
        return float(val) if val is not None else 5.0
    except Exception:
        return 5.0


# ---------------------------------------------------------------------------
# Transaction lineage
# Migration note: DATEDIFF → date_diff handled by _translate_sql
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_journey_counts(filters=None) -> pd.DataFrame:
    """
    Migration note: _session removed. Source line 4593.
    DATEDIFF → date_diff translated automatically by athena_client._translate_sql().
    """
    try:
        from_date, to_date = resolve_date_range(filters or {})
        fd = from_date.strftime("%Y-%m-%d") if hasattr(from_date, "strftime") else str(from_date)
        td = to_date.strftime("%Y-%m-%d")   if hasattr(to_date,   "strftime") else str(to_date)

        where = f"WHERE ORDER_DATE BETWEEN '{fd}' AND '{td}'"
        where += dealer_filter_clause("VW_TRANSACTION_LINEAGE", filters)
        where += lineage_filter_clause(filters)

        df = athena_query(f"""
            SELECT
                SUM(CASE WHEN ORDER_FLAG = 'Y' THEN 1 ELSE 0 END)        AS ORDER_COUNT,
                SUM(CASE WHEN DELIVERY_FLAG = 'Y' THEN 1 ELSE 0 END)     AS DELIVERY_COUNT,
                SUM(CASE WHEN INVOICE_FLAG = 'Y' THEN 1 ELSE 0 END)      AS INVOICE_COUNT,
                SUM(CASE WHEN PAID_FLAG = 'Y' THEN 1 ELSE 0 END)         AS PAID_COUNT,
                SUM(CASE WHEN WARRANTY_FLAG = 'Y' THEN 1 ELSE 0 END)     AS WARRANTY_COUNT,
                SUM(CASE WHEN WARRANTY_STATUS = 'ACTIVE' THEN 1 ELSE 0 END)  AS ACTIVE_WARRANTY_COUNT,
                SUM(CASE WHEN WARRANTY_STATUS = 'EXPIRED' THEN 1 ELSE 0 END) AS EXPIRED_WARRANTY_COUNT,
                ROUND(AVG(
                    CASE WHEN DELIVERY_DATE IS NOT NULL AND ORDER_DATE IS NOT NULL
                         THEN DATEDIFF('day', ORDER_DATE, DELIVERY_DATE)
                         ELSE LEAD_TIME_DAYS END
                ), 1) AS AVG_LEAD_DAYS
            FROM {_DB}.VW_TRANSACTION_LINEAGE
            {where}
        """)
        df.columns = df.columns.str.upper()
        for col in ["ORDER_COUNT", "DELIVERY_COUNT", "INVOICE_COUNT", "PAID_COUNT", "WARRANTY_COUNT"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)
        if "AVG_LEAD_DAYS" not in df.columns:
            df["AVG_LEAD_DAYS"] = 0.0
        df["AVG_LEAD_DAYS"] = df["AVG_LEAD_DAYS"].fillna(0.0)
        return df
    except Exception:
        return pd.DataFrame([{
            "ORDER_COUNT": 0, "DELIVERY_COUNT": 0,
            "INVOICE_COUNT": 0, "PAID_COUNT": 0, "WARRANTY_COUNT": 0,
        }])


@st.cache_data(ttl=60)
def fetch_transaction_lineage(filters=None, page: int = None, page_size: int = 10) -> pd.DataFrame:
    """Migration note: _session removed. Source line 4657."""
    try:
        if page is None:
            page = st.session_state.get("lineage_page", 1)
        from_date, to_date = resolve_date_range(filters or {})
        fd = from_date.strftime("%Y-%m-%d") if hasattr(from_date, "strftime") else str(from_date)
        td = to_date.strftime("%Y-%m-%d")   if hasattr(to_date,   "strftime") else str(to_date)
        where = f"WHERE ORDER_DATE BETWEEN '{fd}' AND '{td}'"
        where += dealer_filter_clause("VW_TRANSACTION_LINEAGE", filters)
        where += lineage_filter_clause(filters)
        offset = (page - 1) * page_size

        # Migration note: Athena v2 (Presto) does not support LIMIT n OFFSET m.
        # Use ROW_NUMBER() subquery for pagination instead.
        df = athena_query(f"""
            SELECT
                TRANSACTION_ID, DEALER_NAME, PRODUCT_CATEGORY, PRODUCT_DESC,
                ORDER_DATE, ORDER_DONE, DELIVERY_DATE, DELIVERY_DONE,
                LEAD_TIME_DAYS, INVOICE_DATE, INVOICE_DONE,
                PAYMENT_DATE, INVOICE_AMOUNT, INVOICE_STATUS,
                WARRANTY_END_DATE, WARRANTY_STATUS
            FROM (
                SELECT
                    TRANSACTION_ID, DEALER_NAME, PRODUCT_CATEGORY, PRODUCT_DESC,
                    ORDER_DATE, ORDER_FLAG AS ORDER_DONE,
                    DELIVERY_DATE, DELIVERY_FLAG AS DELIVERY_DONE,
                    LEAD_TIME_DAYS, INVOICE_DATE, INVOICE_FLAG AS INVOICE_DONE,
                    PAYMENT_DATE, INVOICE_AMOUNT, INVOICE_STATUS,
                    WARRANTY_END_DATE, WARRANTY_STATUS,
                    ROW_NUMBER() OVER (ORDER BY ORDER_DATE DESC) AS _rn
                FROM {_DB}.VW_TRANSACTION_LINEAGE
                {where}
            )
            WHERE _rn BETWEEN {offset + 1} AND {offset + page_size}
        """)
        if "LEAD_TIME_DAYS" in df.columns:
            df["LEAD_TIME_DAYS"] = df["LEAD_TIME_DAYS"].round().astype("Int64")
        df.columns = df.columns.str.upper()
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "TRANSACTION_ID", "DEALER_NAME", "PRODUCT_CATEGORY", "PRODUCT_DESC",
            "ORDER_DATE", "ORDER_DONE", "DELIVERY_DATE", "DELIVERY_DONE",
            "LEAD_TIME_DAYS", "INVOICE_DATE", "PAYMENT_DATE", "INVOICE_DONE",
            "INVOICE_AMOUNT", "INVOICE_STATUS", "WARRANTY_END_DATE", "WARRANTY_STATUS",
        ])


# ---------------------------------------------------------------------------
# Filter-list fetchers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def fetch_regions() -> List[str]:
    """Migration note: _session removed. Source line 4725."""
    _default = ["North", "South", "East", "West", "Central", "North-East", "North-West"]
    try:
        result = athena_query(f"""
            SELECT DISTINCT REGION FROM {_DB}.VW_DEALER_REVENUE_GROWTH
            WHERE REGION IS NOT NULL ORDER BY REGION
        """)
        regions = result["REGION"].dropna().tolist() if not result.empty else []
        return regions if regions else _default
    except Exception:
        return _default


@st.cache_data(ttl=3600)
def fetch_products() -> List[str]:
    """Migration note: _session removed. Source line 4744."""
    _default = ["Sedan", "SUV", "Hatchback", "Truck", "MUV", "Commercial", "EV"]
    try:
        result = athena_query(f"""
            SELECT DISTINCT PRODUCT_CATEGORY FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
            WHERE PRODUCT_CATEGORY IS NOT NULL ORDER BY PRODUCT_CATEGORY
        """)
        products = result["PRODUCT_CATEGORY"].dropna().tolist() if not result.empty else []
        return products if products else _default
    except Exception:
        return _default


# ---------------------------------------------------------------------------
# KPI alert generator
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def generate_kpi_alerts(filters=None) -> pd.DataFrame:
    """Migration note: _session removed. Source line 4081.f"""
    thresholds = {
        "ccc":         {"high": 60,  "medium": 45},
        "lead_time":   {"high": 10,  "medium": 7},
        "repair_tat":  {"high": 24,  "medium": 12},
        "stock_avail": {"low": 70,   "medium": 85},
        "backorder":   {"high": 20,  "medium": 10},
        "gross_margin":{"low": 15,   "medium": 20},
    }
    alerts = []

    _views = [
        (f"SELECT DEALER_NAME, AVG(CCC) AS val FROM {_DB}.VW_CASH_CONVERSION_CYCLE WHERE CCC IS NOT NULL GROUP BY DEALER_NAME",           "ccc",         True),
        (f"SELECT DEALER_NAME, AVG(AVG_ORDER_LEAD_TIME_DAYS) AS val FROM {_DB}.VW_ORDER_LEAD_TIME WHERE AVG_ORDER_LEAD_TIME_DAYS IS NOT NULL GROUP BY DEALER_NAME", "lead_time",   True),
        (f"SELECT DEALER_NAME, AVG(AVG_TURNAROUND_HOURS) AS val FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME WHERE AVG_TURNAROUND_HOURS IS NOT NULL GROUP BY DEALER_NAME", "repair_tat",  True),
        (f"SELECT DEALER_NAME, AVG(STOCK_AVAILABILITY_PCT) AS val FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER WHERE STOCK_AVAILABILITY_PCT IS NOT NULL GROUP BY DEALER_NAME",  "stock_avail", False),
        (f"SELECT DEALER_NAME, AVG(BACKORDER_INCIDENCE_PCT) AS val FROM {_DB}.VW_BACKORDER_INCIDENCE WHERE BACKORDER_INCIDENCE_PCT IS NOT NULL GROUP BY DEALER_NAME",     "backorder",   True),
        (f"SELECT DEALER_NAME, AVG(GROSS_PROFIT_MARGIN_PCT) AS val FROM {_DB}.VW_GROSS_PROFIT_MARGIN WHERE GROSS_PROFIT_MARGIN_PCT IS NOT NULL GROUP BY DEALER_NAME",     "gross_margin",False),
    ]

    for sql, key, higher_is_worse in _views:
        try:
            df = athena_query(sql)
            df.columns = df.columns.str.upper()
            t = thresholds[key]
            for _, row in df.iterrows():
                dealer = str(row["DEALER_NAME"])
                val    = float(row["VAL"])
                if higher_is_worse:
                    if val >= t["high"]:
                        alerts.append({"SEVERITY_LEVEL": "High",   "ISSUE_DESCRIPTION": f"[{dealer}] Critical: {key} at {val:.1f}", "ISSUE_DETAILS": f"{key} exceeds high threshold."})
                    elif val >= t["medium"]:
                        alerts.append({"SEVERITY_LEVEL": "Medium",  "ISSUE_DESCRIPTION": f"[{dealer}] Warning: {key} at {val:.1f}",  "ISSUE_DETAILS": f"{key} approaching threshold."})
                else:
                    if val < t["low"]:
                        alerts.append({"SEVERITY_LEVEL": "High",   "ISSUE_DESCRIPTION": f"[{dealer}] Critical: {key} at {val:.1f}", "ISSUE_DETAILS": f"{key} below low threshold."})
                    elif val < t["medium"]:
                        alerts.append({"SEVERITY_LEVEL": "Medium",  "ISSUE_DESCRIPTION": f"[{dealer}] Warning: {key} at {val:.1f}",  "ISSUE_DETAILS": f"{key} below medium threshold."})
        except Exception:
            pass

    result_df = pd.DataFrame(alerts) if alerts else pd.DataFrame(columns=["SEVERITY_LEVEL", "ISSUE_DESCRIPTION", "ISSUE_DETAILS"])
    if not result_df.empty:
        sev_order = {"High": 0, "Medium": 1, "Low": 2}
        result_df["__s"] = result_df["SEVERITY_LEVEL"].map(sev_order)
        result_df = result_df.sort_values(["__s", "ISSUE_DESCRIPTION"]).drop(columns="__s").reset_index(drop=True)
    return result_df


def fetch_attention_items(filters=None) -> pd.DataFrame:
    """Migration note: _session removed. Source line 4276."""
    if "kpi_alerts_df" in st.session_state and st.session_state.kpi_alerts_df is not None:
        return st.session_state.kpi_alerts_df
    alerts_df = generate_kpi_alerts(filters=filters)
    st.session_state.kpi_alerts_df = alerts_df
    return alerts_df


# ---------------------------------------------------------------------------
# Analytics / trend functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def generate_dynamic_insights(filters=None) -> pd.DataFrame:
    """Migration note: _session removed. Source line 6364."""
    insights = []
    try:
        at_risk = fetch_at_risk_dealers_list(filters)
        if not at_risk.empty:
            insights.append(f"{len(at_risk)} dealers flagged as AT-Risk. Avg health score: {at_risk['HEALTH_SCORE'].mean():.0f}%")
        ccc_val = fetch_cash_conversion_cycle(filters)
        if ccc_val > 0:
            insights.append(f"Cash Conversion Cycle is {ccc_val:.0f} days — {'high working capital needs' if ccc_val > 60 else 'efficient management'}.")
        rev_growth = fetch_revenue_growth(filters)
        if rev_growth > 0:
            insights.append(f"Revenue growth is {rev_growth:.1f}%. Sales momentum is {'strong' if rev_growth > 10 else 'positive'}.")
        gpm = fetch_gross_profit_margin(filters)
        if gpm > 0:
            insights.append(f"Gross Profit Margin at {gpm:.1f}% — {'below target' if gpm < 15 else 'healthy profitability'}.")
        lead = fetch_order_lead_time(filters)
        if lead > 0:
            insights.append(f"Order Lead Time at {lead} days — {'affecting customer satisfaction' if lead > 10 else 'acceptable'}.")
        if not insights:
            insights.append("All KPI metrics are operating within normal parameters.")
        return pd.DataFrame({
            "INSIGHT_TEXT":   insights[:5],
            "PRIORITY_LEVEL": list(range(1, min(6, len(insights) + 1))),
            "CREATED_AT":     [datetime.now()] * len(insights[:5]),
        })
    except Exception:
        return _generate_mock_insights()


@st.cache_data(ttl=3600)
def fetch_revenue_trend() -> pd.DataFrame:
    """Migration note: _session removed. Source line 6365."""
    try:
        return athena_query(f"""
            SELECT dealer_name, PERIOD_YEAR, REVENUE, PREV_REVENUE
            FROM {_DB}.VW_DEALER_REVENUE_GROWTH
            ORDER BY dealer_name, PERIOD_YEAR DESC LIMIT 50
        """)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_profit_margin_by_dealer() -> pd.DataFrame:
    """Migration note: _session removed. Source line 6384."""
    try:
        result = athena_query(f"""
            SELECT DEALER_NAME,
                   AVG(GROSS_PROFIT_MARGIN_PCT) AS GROSS_PROFIT_MARGIN_PCT,
                   SUM(TOTAL_REVENUE) AS TOTAL_REVENUE
            FROM {_DB}.VW_GROSS_PROFIT_MARGIN
            WHERE DEALER_NAME IS NOT NULL
            GROUP BY DEALER_NAME ORDER BY GROSS_PROFIT_MARGIN_PCT DESC LIMIT 15
        """)
        result.columns = result.columns.str.lower()
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_sales_by_product_category() -> pd.DataFrame:
    """Migration note: _session removed. Source line 6406."""
    try:
        result = athena_query(f"""
            SELECT PRODUCT_CATEGORY,
                   SUM(TOTAL_REVENUE) AS total_revenue,
                   SUM(TOTAL_QUANTITY) AS total_quantity
            FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
            GROUP BY PRODUCT_CATEGORY ORDER BY total_revenue DESC LIMIT 20
        """)
        result.columns = result.columns.str.lower()
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_cash_conversion_cycle_trend() -> pd.DataFrame:
    """Migration note: _session removed. Source line 6427."""
    try:
        result = athena_query(f"""
            SELECT DEALER_NAME,
                   AVG(DSO) AS DSO, AVG(DIO) AS DIO,
                   AVG(DPO) AS DPO, AVG(CCC) AS CCC
            FROM {_DB}.VW_CASH_CONVERSION_CYCLE
            WHERE DEALER_NAME IS NOT NULL
            GROUP BY DEALER_NAME ORDER BY CCC DESC LIMIT 10
        """)
        result.columns = result.columns.str.lower()
        return result
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def fetch_order_lead_time_distribution() -> pd.DataFrame:
    """Migration note: _session removed. Source line 6451."""
    try:
        result = athena_query(f"""
            SELECT DEALER_NAME,
                   AVG(AVG_ORDER_LEAD_TIME_DAYS) AS avg_lead_time,
                   COUNT(*) AS order_count
            FROM {_DB}.VW_ORDER_LEAD_TIME
            GROUP BY DEALER_NAME ORDER BY avg_lead_time DESC LIMIT 10
        """)
        result.columns = result.columns.str.lower()
        return result
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Private mock fallback (replaces generate_mock_insights in source)
# ---------------------------------------------------------------------------

def _generate_mock_insights() -> pd.DataFrame:
    return pd.DataFrame({
        "INSIGHT_TEXT":   ["No live insights available — check Athena connectivity."],
        "PRIORITY_LEVEL": [1],
        "CREATED_AT":     [datetime.now()],
    })
