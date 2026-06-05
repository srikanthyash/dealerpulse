"""
app.py — Streamlit UI layer for DealerPulse (AWS migration).
Migration note: all Snowflake imports removed; AWS modules wired in.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import boto3
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import json
import os
import time
import html
import re
import hashlib
import base64
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import logging
import textwrap
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Migration note: snowflake.snowpark imports removed; replaced with AWS modules
from config_loader import get_config, get_aws_session
from athena_client import athena_query
from data_service import *
from ai_service import *
from ai_service import (
    _build_routing_from_yaml,
    _build_intents_from_yaml,
    _build_entity_patterns_from_yaml,
    _build_keyword_tables_from_yaml,
    _initialize_routing_from_yaml,
    _get_dynamic_intents,
    _get_dynamic_entity_patterns,
    _get_dynamic_keyword_tables,
    RouteResult,
)
from utils import DynamoQueryCache, DynamoChatPersistence
_DB: str = get_config()["athena"]["database"]


# ════════════════════════════════════════════════════════════════════════════
# DEALERAPP.PY - FULLY INTEGRATED SINGLE FILE APPLICATION
# ════════════════════════════════════════════════════════════════════════════
# This file combines three systems into one:
#
# 1. YAML UPDATE ENGINE (integrated from update_semantic.py)
#    - Auto-syncs views to YAML semantic model
#    - Detects new VW_* views and adds them automatically
#    - Function: _update_yaml_from_athena()
#
# 2. YAML ROUTING ENGINE (integrated from yaml_routing_engine.py)
#    - Builds routing logic dynamically from YAML at startup
#    - Generates INTENTS, ENTITY_PATTERNS, KEYWORD_TABLES from YAML
#    - Data-driven: no hardcoded routing logic
#    - Functions: _build_routing_from_yaml(), _build_intents_from_yaml(), etc.
#
# 3. STREAMLIT DEALER APP (the main application)
#    - Smart routing with intent detection and entity extraction
#    - Genie query caching with semantic similarity
#    - Cortex Analyst for AI-driven insights
#
# ════════════════════════════════════════════════════════════════════════════
# HOW IT WORKS:
# ════════════════════════════════════════════════════════════════════════════
# Startup Flow:
#   1. Streamlit app starts → st.set_page_config()
#   2. initialize_app_routing() called (cached)
#   3. Loads YAML from S3 or local file
#   4. Optionally syncs new views → updates YAML
#   5. Calls _initialize_routing_from_yaml() with YAML content
#   6. Dynamic routing variables populated (_INTENTS, _ENTITY_PATTERNS, _KEYWORD_TABLES)
#   7. App ready to route questions
#
# User Question Flow:
#   1. User asks question
#   2. route_question() called with YAML verified queries
#   3. Questions compared to dynamic routing config
#   4. Entities extracted using dynamic patterns (state, city, tier, etc.)
#   5. Best verified query found OR Bedrock fallback with dynamic schema
#   6. Results cached with question hash
#
# Adding New Views:
#   1. Add VW_NEW_VIEW to Athena
#   2. Run update_semantic.py OR set auto_add_views=True in initialize_app_routing()
#   3. Run app.py again
#   4. New view automatically in YAML and routing system
#   5. NO PYTHON CODE CHANGES NEEDED!
#
# ════════════════════════════════════════════════════════════════════════════



# ============================================================================
# GLOBAL STATE: Dynamic routing configuration (built from YAML at startup)
# ============================================================================
_INTENTS: Dict[str, Dict] = {}          # Will be populated from YAML
_ENTITY_PATTERNS: Dict[str, Any] = {}   # Will be populated from YAML
_KEYWORD_TABLES: Dict[str, List] = {}   # Will be populated from YAML
_ROUTING_INITIALIZED = False
_YAML_MODEL: Dict[str, Any] = {}




# ============================================================================
# YAML UPDATE ENGINE - Auto-sync views with YAML (from update_semantic.py)
# ============================================================================

def _get_views_from_athena() -> List[str]:  # Migration note: _get_views_from_snowflake → _get_views_from_athena
    """Fetch list of VW_* views from Athena information_schema."""
    try:
        cfg = get_config()
        schema = cfg["athena"]["database"]
        result = athena_query(
            f"SELECT table_name FROM information_schema.views "
            f"WHERE table_schema = '{schema}'"
        )
        if result.empty:
            return []
        names = result['table_name'].str.upper().tolist()
        return [n for n in names if n.startswith('VW_')]
    except Exception as e:
        logging.warning(f"[ATHENA] Could not fetch views: {e}")
        return []


def _build_table_definition_from_glue(view_name: str) -> Dict:  # Migration note: _build_table_definition_from_snowflake → _build_table_definition_from_glue
    """Build semantic table definition using Glue Data Catalog for column comments."""
    try:
        cfg = get_config()
        glue = get_aws_session().client("glue", region_name=cfg["aws"]["region"])
        response = glue.get_table(DatabaseName=cfg["athena"]["database"], Name=view_name.lower())
        columns = response["Table"]["StorageDescriptor"]["Columns"]

        dimensions = []
        facts = []

        for col in columns:
            col_name = col["Name"]
            col_comment = col.get("Comment", "")
            col_name_upper = col_name.upper()

            if any(suffix in col_name_upper for suffix in ("_PCT", "_PERCENT", "_DAYS", "_HOURS", "_AMOUNT", "_RATE", "_COUNT", "_TOTAL", "_MARGIN")):
                facts.append({
                    "name": col_name,
                    "description": col_comment or f"{col_name} from {view_name}",
                    "type": "NUMERIC",
                })
            elif col_name_upper in ("PERIOD_YEAR", "PERIOD_MONTH", "PERIOD_WEEK"):
                dimensions.append({
                    "name": col_name,
                    "description": col_comment or f"Time period {col_name}",
                    "type": "TIME",
                })
            else:
                dimensions.append({
                    "name": col_name,
                    "description": col_comment or f"Dimension {col_name}",
                    "type": "STRING",
                })

        return {
            "name": view_name,
            "table": f"{cfg['athena']['database']}.{view_name}",
            "description": f"Auto-generated table definition for {view_name}",
            "dimensions": dimensions[:15],
            "facts": facts[:10],
        }
    except Exception as e:
        logging.warning(f"[GLUE] Could not build table def for {view_name}: {e}")
        return {}


def _update_yaml_from_athena(current_yaml: str, auto_add_views: bool = True) -> str:  # Migration note: _update_yaml_from_snowflake → _update_yaml_from_athena
    """
    Sync Athena views with YAML model using Athena + Glue Data Catalog.
    Adds missing VW_* views to YAML automatically.
    """
    if not auto_add_views:
        return current_yaml

    try:
        model = yaml.safe_load(current_yaml)
        if not model:
            model = {}
    except Exception:
        model = {}

    existing_tables = {t.get("name") for t in model.get("tables", []) if isinstance(t, dict)}
    views = _get_views_from_athena()

    added_views = []
    for view_name in views:
        if view_name not in existing_tables:
            table_def = _build_table_definition_from_glue(view_name)
            if table_def:
                model.setdefault("tables", []).append(table_def)
                added_views.append(view_name)
                logging.info(f"[YAML UPDATE] ✅ Added {view_name} to model")

    if added_views:
        logging.info(f"[YAML UPDATE] Added {len(added_views)} new views: {', '.join(added_views)}")
    else:
        logging.info(f"[YAML UPDATE] No new views to add - YAML is up to date")

    return yaml.dump(model, default_flow_style=False, sort_keys=False)


def _save_yaml_locally(yaml_content: str, filename: str = "dealer_model.yml") -> bool:
    """Save YAML content to local file."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        logging.info(f"[YAML SAVE] ✅ Saved to local file: {filename}")
        return True
    except Exception as e:
        logging.error(f"[YAML SAVE] ❌ Failed to save locally: {e}")
        return False


def _upload_yaml_to_s3(yaml_content: str) -> bool:  # Migration note: _upload_yaml_to_snowflake → _upload_yaml_to_s3
    """Upload YAML content to S3 config path."""
    try:
        cfg = get_config()
        s3 = get_aws_session().client("s3", region_name=cfg["aws"]["region"])
        s3.put_object(
            Bucket=cfg["s3"]["bucket"],
            Key=cfg["s3"]["config_prefix"] + cfg["s3"]["yaml_filename"],
            Body=yaml_content.encode("utf-8"),
            ContentType="application/x-yaml",
        )
        logging.info(f"[S3 UPLOAD] ✅ Uploaded YAML to s3://{cfg['s3']['bucket']}/{cfg['s3']['config_prefix']}{cfg['s3']['yaml_filename']}")
        return True
    except Exception as e:
        logging.error(f"[S3 UPLOAD] ❌ Failed: {e}")
        return False





# ============================================================================
# LOAD UI CONFIGURATION FROM ui_config.yaml
# ============================================================================
# Migration note: get_active_session() + session.file.get_stream(@STAGE) → load_yaml_from_s3()
def _load_ui_config():
    """
    Load UI configuration from S3 (primary) or local file (fallback).
    Migration note: Snowflake stage read replaced with S3 load via bedrock_client.
    """
    # Try S3 first
    try:
        from bedrock_client import load_yaml_from_s3
        config_data = load_yaml_from_s3('ui_config.yaml')
        if config_data:
            print(f"✅ Loaded ui_config.yaml from S3")
            return config_data
    except Exception as e:
        print(f"⚠️ Could not load from S3: {e}")

    # Fallback: Try local file
    local_err = None
    try:
        config_paths = [
            os.path.join(os.path.dirname(__file__), "ui_config.yaml"),
            os.path.join(os.getcwd(), "ui_config.yaml"),
            "ui_config.yaml"
        ]

        for path in config_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)
                    print(f"✅ Loaded ui_config.yaml from local file: {path}")
                    return config_data or {}

        local_err = "No local ui_config.yaml found"
    except Exception as e:
        local_err = str(e)

    print(
        f"⚠️ Failed to load ui_config.yaml from S3 or local file.\n"
        f"Local error: {local_err}\n"
        f"Using empty config - UI styling may not load correctly"
    )
    return {}

# Load configuration
_ui_config = _load_ui_config()

def get_config_value(key: str, default=None):
    """Get configuration value using dot notation (e.g., 'theme.colors.primary')"""
    keys = key.split(".")
    value = _ui_config
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
            if value is None:
                return default
        else:
            return default
    return value if value is not None else default

# Apply page config before any st.* commands
st.set_page_config(
    page_title=get_config_value("app.page_title", "Dealers Dashboard"),
    page_icon=get_config_value("app.page_icon", "📊"),
    layout=get_config_value("app.layout", "wide"),
    initial_sidebar_state=get_config_value("app.initial_sidebar_state", "collapsed")
)


# ════════════════════════════════════════════════════════════════════════════
# YAML UPDATE ENGINE - Integrated from update_semantic.py (v3)
# ════════════════════════════════════════════════════════════════════════════

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_YAML_FILENAME = "dealer_model.yml"
_YAML_LOCAL_PATH = os.path.join(_SCRIPT_DIR, _YAML_FILENAME)

# Migration note: Snowflake database/schema/stage keys replaced with AWS equivalents
_SEMANTIC_CONFIG = {
    "yaml_filename":      _YAML_FILENAME,
    "local_output":       _YAML_LOCAL_PATH,
    "local_fallback":     _YAML_LOCAL_PATH,
    "view_prefix":        "VW_",
    "model_name":         "DEALER_KPI_MODEL",
    "model_description":  "Semantic model for Dealer KPIs - Sales, Performance, Cash Management",
    "core_kpi_tables": [
        ("VW_GROSS_PROFIT_MARGIN", "m"),
        ("VW_DEALER_REVENUE_GROWTH", "r"),
        ("VW_CASH_CONVERSION_CYCLE", "c"),
        ("VW_AVERAGE_REPAIR_TURNAROUND_TIME", "t"),
        ("VW_DEALER_CONTRIBUTION_MARGIN", "cm"),
        ("VW_SALES_VOLUME", "v"),
    ],
    "numeric_suffixes": (
        "_PCT", "_PERCENT", "_DAYS", "_HOURS", "_AMOUNT", "_RATE",
        "_COUNT", "_TOTAL", "_REVENUE", "_MARGIN", "_COST", "_UNITS",
        "_SCORE", "_INDEX", "_RATIO", "_QTY", "_QUANTITY", "CCC", "DSO", "DIO", "DPO",
    ),
    "time_suffixes": (
        "_DATE", "_MONTH", "_YEAR", "_PERIOD", "_AT", "_TIME", "_WEEK",
    ),
    "dimension_suffixes": (
        "_NAME", "_ID", "_CODE", "_TYPE", "_STATUS", "_REGION",
        "_CATEGORY", "_SEGMENT", "_TIER", "_BRAND", "_CITY",
        "_STATE", "_COUNTRY", "_POSTAL_CODE", "_ADDRESS",
    ),
}

_SYNONYM_MAP = {
    "DEALER_NAME": ["dealer", "distributor name"],
    "GROSS_PROFIT_MARGIN_PCT": ["gross margin", "GPM"],
    "REVENUE_GROWTH_MOM_PERCENT": ["revenue growth", "MoM growth"],
    "AVG_TURNAROUND_HOURS": ["repair TAT", "service turnaround"],
    "AVG_ORDER_LEAD_TIME_DAYS": ["lead time", "order lead time"],
    "STOCK_AVAILABILITY_PCT": ["stock availability", "in-stock rate"],
    "BACKORDER_INCIDENCE_PCT": ["backorder rate"],
    "CCC": ["cash conversion cycle", "CCC days"],
    "DSO": ["days sales outstanding"],
    "DIO": ["days inventory outstanding"],
    "DPO": ["days payable outstanding"],
}

def _semantic_role(col_name: str, data_type: str) -> str:
    """Determine column type: time_dimension, fact, or dimension."""
    u = col_name.upper()
    dt = data_type.upper()
    if dt in ("DATE", "TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ", "TIME", "TIMESTAMP"):
        return "time_dimension"
    for s in _SEMANTIC_CONFIG["time_suffixes"]:
        if u.endswith(s):
            return "time_dimension"
    if dt in ("NUMBER", "FLOAT", "INT", "INTEGER", "BIGINT", "SMALLINT", "DECIMAL", "NUMERIC", "DOUBLE", "REAL", "DOUBLE PRECISION"):
        for s in _SEMANTIC_CONFIG["dimension_suffixes"]:
            if u.endswith(s):
                return "dimension"
        return "fact"
    for s in _SEMANTIC_CONFIG["numeric_suffixes"]:
        if u.endswith(s) or u == s.lstrip("_"):
            return "fact"
    return "dimension"

def _semantic_label(col_name: str) -> str:
    """Convert column name to user-friendly label."""
    return col_name.replace("_", " ").title()

def _semantic_human(view_name: str) -> str:
    """Convert view name to human-readable format."""
    return view_name.replace("VW_", "", 1).replace("_", " ").title()

def _semantic_fingerprint(columns: List[Dict]) -> str:
    """Create hash of column structure for change detection."""
    sig = json.dumps(sorted([(c["name"], c["data_type"]) for c in columns]), sort_keys=True)
    return hashlib.md5(sig.encode()).hexdigest()

def _semantic_list_views(db: str, schema: str, prefix: str) -> List[str]:
    """Fetch all views matching prefix from Athena information_schema."""
    # Migration note: session.sql().to_pandas() → athena_query()
    try:
        df = athena_query(
            f"SELECT table_name FROM information_schema.views "
            f"WHERE table_schema = '{schema.lower()}'"
        )
        if df.empty:
            return []
        views = [v.upper() for v in df["table_name"].tolist() if v.upper().startswith(prefix)]
        logging.info(f"[SEMANTIC] Found {len(views)} views matching '{prefix}*'")
        return views
    except Exception as e:
        logging.warning(f"[SEMANTIC] Could not list views: {e}")
        return []

def _semantic_get_columns(db: str, schema: str, view: str) -> List[Dict]:
    """Fetch column definitions from Glue Data Catalog (includes comments)."""
    # Migration note: session.sql(INFORMATION_SCHEMA.COLUMNS) → glue.get_table()
    try:
        cfg_data = get_config()
        glue = get_aws_session().client("glue", region_name=cfg_data["aws"]["region"])
        response = glue.get_table(DatabaseName=cfg_data["athena"]["database"], Name=view.lower())
        columns = response["Table"]["StorageDescriptor"]["Columns"]
        return [
            {"name": col["Name"].upper(), "data_type": col["Type"].upper(),
             "nullable": True, "comment": col.get("Comment", "")}
            for col in columns
        ]
    except Exception as e:
        logging.warning(f"[SEMANTIC] Could not get columns for {view}: {e}")
        return []

def _semantic_build_table_block(view: str, columns: List[Dict], db: str, schema: str) -> Dict:
    """Build semantic table block with dimensions, facts, and time dimensions."""
    dims, tdims, facts = [], [], []
    for col in columns:
        r = _semantic_role(col["name"], col["data_type"])
        syns = _SYNONYM_MAP.get(col["name"].upper(), [])
        base: Dict[str, Any] = {
            "name": col["name"],
            "expr": col["name"],
            "description": col["comment"] or _semantic_label(col["name"]),
            "label": _semantic_label(col["name"]),
        }
        if syns:
            base["synonyms"] = syns
        if r == "time_dimension":
            base["time_granularity"] = "month" if "MONTH" in col["name"].upper() else "year" if "YEAR" in col["name"].upper() else "day"
            tdims.append(base)
        elif r == "fact":
            base["default_aggregation"] = "avg" if any(col["name"].upper().endswith(s) for s in ("_PCT", "_PERCENT", "_RATE")) else "sum"
            facts.append(base)
        else:
            dims.append(base)

    block: Dict[str, Any] = {
        "name": view,
        "base_table": {"database": db, "schema": schema, "table": view},
        "description": f"{_semantic_human(view)} — columns: " + ", ".join(c["name"] for c in columns[:8]) + ("…" if len(columns) > 8 else ""),
    }
    if dims:   block["dimensions"] = dims
    if tdims:  block["time_dimensions"] = tdims
    if facts:  block["facts"] = facts
    return block

def _semantic_build_verified_query(view: str, columns: List[Dict], db: str, schema: str) -> Dict:
    """Build overview and cross-metric verified queries."""
    tbl = f"{db}.{schema}.{view}"  # Migration note: Athena uses database.table; update schema join as needed
    alias = "dl"
    human = _semantic_human(view)
    qname = view.replace("VW_", "", 1).lower() + "_overview"
    nw = view.replace("VW_", "").replace("_", " ").lower()

    dims = [c for c in columns if _semantic_role(c["name"], c["data_type"]) == "dimension"]
    facts = [c for c in columns if _semantic_role(c["name"], c["data_type"]) == "fact"]
    has_dealer = any(c["name"].upper() == "DEALER_NAME" for c in columns)

    sel = [f"  {alias}.{c['name']}" for c in dims]
    for fc in facts:
        agg = "AVG" if any(fc["name"].upper().endswith(s) for s in ("_PCT", "_PERCENT", "_RATE")) else "SUM"
        sel.append(f"  ROUND({agg}({alias}.{fc['name']}), 2) AS {fc['name']}")

    order = f"{alias}.{dims[0]['name']}" if dims else "1"

    if facts and dims:
        grp = ", ".join(f"{alias}.{c['name']}" for c in dims)
        sql = f"SELECT\n" + ",\n".join(sel) + f"\nFROM {tbl} {alias}\nGROUP BY {grp}\nORDER BY {order};"
    elif sel:
        sql = f"SELECT\n" + ",\n".join(sel) + f"\nFROM {tbl} {alias}\nORDER BY {order};"
    else:
        sql = f"SELECT * FROM {tbl} LIMIT 1000;"

    entry: Dict[str, Any] = {
        "name": qname,
        "question": f"Show me {nw}. Where are my {nw}? {human} overview.",
        "use_as_onboarding_question": True,
        "sql": sql,
    }

    if has_dealer and view != "VW_GROSS_PROFIT_MARGIN":
        cross_name = view.replace("VW_", "", 1).lower() + "_with_performance"
        perf_tbl = f"{db}.{schema}.VW_GROSS_PROFIT_MARGIN"
        dim_lines = "\n".join(f"    {alias}.{c['name']}," for c in dims)
        fact_lines = ""
        for fc in facts:
            agg = "AVG" if any(fc["name"].upper().endswith(s) for s in ("_PCT", "_PERCENT", "_RATE")) else "SUM"
            fact_lines += f"    ROUND({agg}({alias}.{fc['name']}), 2) AS {fc['name']},\n"
        grp2 = ", ".join(f"{alias}.{c['name']}" for c in dims)
        cross_sql = (
            f"WITH base AS (\n  SELECT\n{dim_lines}\n{fact_lines}  FROM {tbl} {alias}\n" +
            (f"  GROUP BY {grp2}\n" if grp2 else "") +
            f"),\nperf AS (\n  SELECT DEALER_NAME,\n    AVG(GROSS_PROFIT_MARGIN_PCT) AS avg_margin,\n    SUM(TOTAL_REVENUE) AS total_revenue\n  FROM {perf_tbl}\n  GROUP BY DEALER_NAME\n)\n"
            f"SELECT b.*, ROUND(p.avg_margin,2) AS avg_gross_margin_pct, ROUND(p.total_revenue,0) AS total_revenue\n"
            f"FROM base b\nLEFT JOIN perf p ON b.DEALER_NAME = p.DEALER_NAME\nORDER BY p.total_revenue DESC NULLS LAST;"
        )
        entry["_cross"] = {"name": cross_name, "question": f"Show {human} with dealer performance.", "sql": cross_sql}

    return entry

def _semantic_build_routing_trigger(view: str, query_name: str) -> Tuple[Dict, List[str]]:
    """Build routing trigger with keywords."""
    nw = view.replace("VW_", "").replace("_", " ").lower()
    parts = nw.split()
    tid = "p9_" + view.replace("VW_", "").lower()
    keywords = list(dict.fromkeys([nw, _semantic_human(view).lower(), f"show {nw}"] +
                    [p for p in parts if len(p) > 3 and p not in ("dealer", "average")]))
    trigger_dict = {
        "id": tid, "priority": 9, "trigger": keywords, "pre_built_query": query_name,
        "description": f"Auto-generated fast path for {_semantic_human(view)}.",
    }
    return trigger_dict, keywords

def _semantic_build_expert_attr(view: str, columns: List[Dict]) -> str:
    """Build expert SQL attribute."""
    col_str = ", ".join(c["name"] for c in columns)
    return f"Know {_semantic_human(view).lower()}: {view} ({col_str})"

def _semantic_build_join_matrix(view: str, columns: List[Dict]) -> Optional[Dict]:
    """Build join matrix for multi-table queries."""
    if not any(c["name"].upper() == "DEALER_NAME" for c in columns):
        return None
    alias = "dl"
    joinable = [
        {"table": kt, "alias": ka, "join_key": "DEALER_NAME", "join_type": "LEFT JOIN",
         "on_clause": f"ON {alias}.DEALER_NAME = {ka}.DEALER_NAME"}
        for kt, ka in _SEMANTIC_CONFIG["core_kpi_tables"] if kt != view
    ]
    return {
        "primary_table": view, "alias": alias, "description": f"Auto-generated join rules for {_semantic_human(view)}.",
        "joinable_to": joinable,
    }

def _semantic_merge_auto_vq_index(model: Dict, view: str, query_name: str, cross_name: Optional[str], keywords: List[str]) -> None:
    """Merge auto-generated VQ index for runtime discovery."""
    index = model.setdefault("auto_verified_queries", [])
    existing = {e["query_name"] for e in index}
    if query_name not in existing:
        index.append({"query_name": query_name, "view": view, "keywords": keywords, "description": f"Auto-generated for {_semantic_human(view)}."})
    if cross_name and cross_name not in existing:
        cross_kw = list(dict.fromkeys(keywords + ["performance", "revenue", "profit", "compare"]))
        index.append({"query_name": cross_name, "view": view, "keywords": cross_kw, "description": f"Cross-metric for {_semantic_human(view)}."})

def _semantic_merge_table(model: Dict, block: Dict) -> str:
    """Merge table block into model."""
    idx = {t["name"]: i for i, t in enumerate(model.get("tables", []))}
    tables = model.setdefault("tables", [])
    if block["name"] not in idx:
        tables.append(block)
        return "added"
    i, ex = idx[block["name"]], tables[idx[block["name"]]]
    if ex.get("description") and not ex["description"].startswith(block["name"]):
        block["description"] = ex["description"]
    for k in ("dimensions", "time_dimensions", "facts"):
        if block.get(k):
            em = {c["name"]: c for c in (ex.get(k, []) or [])}
            for nc in block[k]:
                ec = em.get(nc["name"])
                if ec and ec.get("synonyms") and not nc.get("synonyms"):
                    nc["synonyms"] = ec["synonyms"]
    tables[i] = block
    return "updated"

def _semantic_merge_vq(model: Dict, entry: Dict) -> bool:
    """Merge verified query into model."""
    existing = {q["name"] for q in model.get("verified_queries", [])}
    vqs = model.setdefault("verified_queries", [])
    added = False
    cross = entry.pop("_cross", None)
    if entry["name"] not in existing:
        vqs.append(entry)
        added = True
    if cross and cross["name"] not in existing:
        vqs.append(cross)
    return added

def _semantic_merge_trigger(model: Dict, trigger_dict: Dict) -> bool:
    """Merge routing trigger into model."""
    patterns = model.setdefault("cortex_optimization", {}).setdefault("cortex_training_patterns", {}).setdefault("priority_9_auto_generated", [])
    if trigger_dict["id"] not in {p.get("id") for p in patterns}:
        patterns.append(trigger_dict)
        return True
    return False

def _semantic_merge_expert(model: Dict, attr_str: str) -> bool:
    """Merge expert SQL attribute into model."""
    attrs = model.setdefault("semantic_matching_integration", {}).setdefault("expert_sql_attributes", [])
    view_token = attr_str.split("(")[0].split()[-1] if "(" in attr_str else ""
    if not any(view_token in a for a in attrs):
        attrs.append(attr_str)
        return True
    return False

def _semantic_merge_jm(model: Dict, entry: Optional[Dict]) -> bool:
    """Merge join matrix into model."""
    if not entry:
        return False
    rules = model.setdefault("table_join_matrix", {}).setdefault("join_rules", [])
    if entry["primary_table"] not in {r.get("primary_table") for r in rules}:
        rules.append(entry)
        return True
    return False

def _semantic_load_yaml(cfg: Dict) -> Dict:
    """Load YAML from S3 (primary) or local file (fallback)."""
    # Migration note: session.file.get_stream(@STAGE) → load_yaml_from_s3()
    from bedrock_client import load_yaml_from_s3 as _s3_load
    try:
        data = _s3_load(cfg["yaml_filename"])
        if data:
            logging.info(f"[SEMANTIC] ✅ Loaded YAML from S3: {cfg['yaml_filename']}")
            return data
    except Exception as e:
        logging.warning(f"[SEMANTIC] S3 load failed: {e}")

    local_path = cfg["local_fallback"]
    try:
        if os.path.exists(local_path):
            with open(local_path, encoding="utf-8") as f:
                data = yaml.safe_load(f.read()) or {}
            logging.info(f"[SEMANTIC] ✅ Loaded YAML from LOCAL: {local_path}")
            return data
        else:
            logging.warning(f"[SEMANTIC] Local file not found at: {local_path}")
    except Exception as e:
        logging.warning(f"[SEMANTIC] Local load failed: {e}")

    logging.warning(f"[SEMANTIC] Could not load YAML from S3 or local - will use minimal fallback")
    return {}

def _semantic_upload_yaml(yaml_content: str) -> bool:
    """Upload YAML to S3."""
    # Migration note: session.file.put_stream(@STAGE) → _upload_yaml_to_s3()
    return _upload_yaml_to_s3(yaml_content)

def _semantic_run(cfg: Dict = None) -> Dict:
    """Main orchestrator - builds complete YAML from Athena/Glue views."""
    # Migration note: get_active_session() removed; Athena/Glue used directly
    if cfg is None:
        cfg = _SEMANTIC_CONFIG

    logging.info("[SEMANTIC] ════════════════════════════════════════════════════════════")
    logging.info("[SEMANTIC] SEMANTIC RUN STARTED")
    logging.info("[SEMANTIC] ════════════════════════════════════════════════════════════")

    cfg_data = get_config()
    db = cfg_data["athena"]["database"]
    schema = cfg_data["athena"]["database"]
    prefix = cfg["view_prefix"]
    filename = cfg["yaml_filename"]

    model = _semantic_load_yaml(cfg)
    model.setdefault("name", cfg["model_name"])
    model.setdefault("description", cfg["model_description"])
    model.setdefault("tables", [])

    existing_idx = {t["name"]: i for i, t in enumerate(model.get("tables", []))}
    all_views = _semantic_list_views(db, schema, prefix)
    logging.info(f"[SEMANTIC] Found {len(all_views)} VW_* views in Athena")
    logging.info(f"[SEMANTIC] YAML has {len(existing_idx)} existing tables")

    new_views, changed_views = [], []

    for view in all_views:
        cols = _semantic_get_columns(db, schema, view)
        if not cols:
            logging.warning(f"[SEMANTIC] Could not get columns for {view} - skipping")
            continue

        if view not in existing_idx:
            logging.info(f"[SEMANTIC] [NEW] View detected: {view}")
            new_views.append((view, cols))
        else:
            existing_fps = {t["name"]: _semantic_fingerprint(t.get("dimensions", []) + t.get("time_dimensions", []) + t.get("facts", []))
                           for t in model.get("tables", [])}
            current_fp = _semantic_fingerprint(cols)
            if view in existing_fps and existing_fps.get(view) != current_fp:
                logging.info(f"[SEMANTIC] [CHANGED] View schema changed: {view}")
                changed_views.append((view, cols))

    logging.info(f"[SEMANTIC] Detection complete: {len(new_views)} new, {len(changed_views)} changed")

    added_list, updated_list = [], []
    for view, cols in new_views + changed_views:
        if view not in existing_idx:
            block = _semantic_build_table_block(view, cols, db, schema)
            _semantic_merge_table(model, block)
            added_list.append(view)
        else:
            updated_list.append(view)

        vq = _semantic_build_verified_query(view, cols, db, schema)
        cross_name = vq.get("_cross", {}).get("name") if "_cross" in vq else None
        _semantic_merge_vq(model, vq)

        qname = view.replace("VW_", "", 1).lower() + "_overview"
        trigger_dict, keywords = _semantic_build_routing_trigger(view, qname)
        _semantic_merge_trigger(model, trigger_dict)
        _semantic_merge_expert(model, _semantic_build_expert_attr(view, cols))
        _semantic_merge_jm(model, _semantic_build_join_matrix(view, cols))

        has_dealer = any(c["name"].upper() == "DEALER_NAME" for c in cols)
        resolved_cross = cross_name if (has_dealer and view != "VW_GROSS_PROFIT_MARGIN") else None
        _semantic_merge_auto_vq_index(model, view, qname, resolved_cross, keywords)

    # Save YAML
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    hdr = f"# === Auto-updated {ts} (integrated) ===\n"
    if added_list:
        hdr += f"# Added  : {', '.join(added_list)}\n"
    if updated_list:
        hdr += f"# Updated: {', '.join(updated_list)}\n"

    yaml_str = hdr + "\n" + yaml.dump(model, default_flow_style=False, allow_unicode=True, sort_keys=False, width=120)

    local_path = cfg["local_output"]
    saved_locally = False
    uploaded_to_s3 = False  # Migration note: Snowflake stage → S3

    # Step 1: Save to local file
    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(yaml_str)
        file_size = os.path.getsize(local_path) / 1024
        logging.info(f"[SEMANTIC] ✅ Saved locally: {local_path} ({file_size:.1f}KB)")
        saved_locally = True
    except Exception as e:
        logging.error(f"[SEMANTIC] ❌ Local save failed: {type(e).__name__}: {str(e)}")
        return {"added": added_list, "updated": updated_list, "local_path": None}

    # Step 2: Upload to S3 (Migration note: Snowflake stage → S3)
    logging.info(f"[SEMANTIC] Starting S3 upload...")
    uploaded_to_s3 = _semantic_upload_yaml(yaml_str)

    if uploaded_to_s3:
        logging.info(f"[SEMANTIC] SUCCESS: YAML updated on Local + S3")
    elif saved_locally:
        logging.warning(f"[SEMANTIC] WARNING: YAML updated on Local only (S3 upload failed)")

    logging.info("[SEMANTIC] ════════════════════════════════════════════════════════════")
    logging.info(f"[SEMANTIC] SEMANTIC RUN COMPLETED")
    logging.info(f"[SEMANTIC] Added tables:   {len(added_list)} - {added_list}")
    logging.info(f"[SEMANTIC] Updated tables: {len(updated_list)} - {updated_list}")
    logging.info(f"[SEMANTIC] Local path:     {local_path}")
    logging.info(f"[SEMANTIC] Upload status:  {'SUCCESS' if uploaded_to_s3 else 'FAILED' if saved_locally else 'SKIPPED'}")
    logging.info("[SEMANTIC] ════════════════════════════════════════════════════════════")

    return {"added": added_list, "updated": updated_list, "local_path": local_path}


# ════════════════════════════════════════════════════════════════════════════
# APP INITIALIZATION - Run once per Python process
# ════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def _init_routing_cached():
    """Cached initialization — runs ONCE per Python process."""
    logging.info("[CACHED INIT] 🔧 Starting cached routing initialization...")
    try:
        # Step 1: Build YAML from Athena (Migration note: Snowflake views → Athena views)
        logging.info("[APP INIT] ✅ Building complete YAML from Athena views...")
        try:
            result = _semantic_run()
            num_added = len(result.get("added", []))
            num_updated = len(result.get("updated", []))
            logging.info(f"[APP INIT] ✅ YAML update completed: {num_added} added, {num_updated} updated")
        except Exception as e:
            logging.warning(f"[APP INIT] ⚠️ YAML update failed: {e} - loading existing file")

        # Step 2: Load YAML
        logging.info("[APP INIT] Loading YAML model...")
        yaml_model = load_yaml_model()

        if not yaml_model or not isinstance(yaml_model, dict):
            logging.error("[APP INIT] ❌ YAML model is invalid")
            return None, None, None, None, False

        logging.info(f"[APP INIT] ✅ YAML loaded: {yaml_model.get('name', 'unknown')} with {len(yaml_model.get('tables', []))} tables")

        # Step 3: Build routing from YAML
        logging.info("[APP INIT] Converting YAML to string and building routing...")
        yaml_str = yaml.dump(yaml_model, default_flow_style=False)
        logging.info(f"[APP INIT] YAML string length: {len(yaml_str)} chars")

        logging.info("[APP INIT] Calling _build_routing_from_yaml()...")
        intents, entity_patterns, keyword_tables, model = _build_routing_from_yaml(yaml_str)

        # Populate ai_service.py module globals so _score_intent / _find_best_verified_query
        # use the YAML intents, not the hardcoded fallback INTENTS dict.
        _initialize_routing_from_yaml(yaml_str)

        logging.info(f"[APP INIT] ✅ Built routing: {len(intents)} intents, {len(entity_patterns)} patterns, {len(keyword_tables)} tables")

        if intents:
            num_tables = len(model.get('tables', []))
            logging.info(f"[APP INIT] ✅ Routing initialized with {len(intents)} intents and {num_tables} tables from YAML")
            logging.info("[CACHED INIT] ✅ SUCCESS: Routing initialized")
            logging.info(f"[CACHED INIT]   - Intents: {len(intents)}")
            logging.info(f"[CACHED INIT]   - Entity patterns: {len(entity_patterns)}")
            logging.info(f"[CACHED INIT]   - Keyword tables: {len(keyword_tables)}")
            logging.info(f"[CACHED INIT]   - YAML tables: {len(model.get('tables', []))}")
            return intents, entity_patterns, keyword_tables, model, True
        else:
            logging.error("[APP INIT] ❌ No intents built from YAML")
            return None, None, None, None, False

    except Exception as e:
        logging.error(f"[CACHED INIT] ❌ EXCEPTION: {e}", exc_info=True)
        return None, None, None, None, False


# Execute the initialization and assign results to globals
logging.info("[APP START] Calling cached routing initialization...")
_init_intents, _init_patterns, _init_keywords, _init_model, _init_success = _init_routing_cached()

if _init_success and _init_intents:
    _INTENTS = _init_intents
    _ENTITY_PATTERNS = _init_patterns
    _KEYWORD_TABLES = _init_keywords
    _YAML_MODEL = _init_model
    _ROUTING_INITIALIZED = True
    logging.info(f"[APP START] ✅ Globals assigned: _ROUTING_INITIALIZED=True, _INTENTS count={len(_INTENTS)}")
else:
    logging.error(f"[APP START] ❌ Initialization failed or returned None")
    logging.error(f"[APP START] Current state: _ROUTING_INITIALIZED={_ROUTING_INITIALIZED}, _INTENTS count={len(_INTENTS)}")

# Migration note: SNOWFLAKE_AVAILABLE detection block removed (lines 1695-1700 source)
# Migration note: GenieQueryCache class skipped — replaced by DynamoQueryCache in utils.py (lines 1707+ source)
# Migration note: GenieLongTermMemory class skipped — already in ai_service.py (source lines follow)
# Migration note: GenieChatPersistence class skipped — replaced by DynamoChatPersistence in utils.py


# ============================================================================
# DEALER PERFORMANCE FORECASTER
# ============================================================================

class DealerPerformanceForecaster:
    """Generate time-series forecasts for dealer metrics."""

    def __init__(self):
        # Migration note: session parameter removed; uses athena_query() directly
        pass

    def _generate_mock_data(self, dealer_name: str) -> Dict:
        """Generate mock forecast data for testing when real data is unavailable."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            return {"error": "statsmodels not installed"}

        np.random.seed(42)
        base_revenue = 100000
        trend = np.linspace(0, 15000, 52)
        noise = np.random.normal(0, 5000, 52)
        historical_values = base_revenue + trend + noise

        model = ExponentialSmoothing(
            historical_values,
            trend='add',
            seasonal='add',
            seasonal_periods=13
        )
        fitted_model = model.fit(optimized=True)

        weeks = 8
        forecast_values = fitted_model.forecast(steps=weeks)

        residuals = fitted_model.fittedvalues - historical_values
        rmse = np.sqrt(np.mean(residuals ** 2))
        mape = np.mean(np.abs(residuals / historical_values)) * 100
        ci_95 = 1.96 * rmse

        return {
            "success": True,
            "dealer": dealer_name,
            "forecast_weeks": weeks,
            "recent_revenue": float(historical_values[-1]),
            "forecast_values": forecast_values.tolist(),
            "forecast_upper_bound": (forecast_values + ci_95).tolist(),
            "forecast_lower_bound": (forecast_values - ci_95).tolist(),
            "trend": "improving" if forecast_values[-1] > historical_values[-1] else "declining",
            "change_percent": ((forecast_values[-1] - historical_values[-1]) / historical_values[-1] * 100),
            "mape": mape,
            "confidence": "high" if mape < 15 else "medium" if mape < 30 else "low",
            "mock": True
        }

    def forecast_revenue(self, dealer_name: str, weeks: int = 8) -> Dict:
        """Forecast revenue for dealer using exponential smoothing."""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            return {"error": "statsmodels not installed", "values": []}

        query = f"""
        SELECT REVENUE
        FROM {_DB}.VW_DEALER_REVENUE_GROWTH
        WHERE DEALER_NAME = '{dealer_name}'
        AND PERIOD_YEAR >= year(current_date) - 1
        ORDER BY PERIOD_YEAR DESC
        LIMIT 52
        """

        try:
            df = athena_query(query)  # Migration note: session.sql().to_pandas() → athena_query()
            if df.empty or len(df) < 8:
                return self._generate_mock_data(dealer_name)

            values = df['REVENUE'].dropna().values.astype(float)
            if len(values) < 8:
                return self._generate_mock_data(dealer_name)

            model = ExponentialSmoothing(
                values,
                trend='add',
                seasonal='add',
                seasonal_periods=min(13, len(values)//2)
            )
            fitted_model = model.fit(optimized=True)
            forecast_values = fitted_model.forecast(steps=weeks)

            residuals = fitted_model.fittedvalues - values
            rmse = np.sqrt(np.mean(residuals ** 2))
            mape = np.mean(np.abs(residuals / values)) * 100 if np.any(values != 0) else 0
            ci_95 = 1.96 * rmse

            return {
                "success": True,
                "dealer": dealer_name,
                "forecast_weeks": weeks,
                "recent_revenue": float(values[-1]),
                "forecast_values": forecast_values.tolist(),
                "forecast_upper_bound": (forecast_values + ci_95).tolist(),
                "forecast_lower_bound": (forecast_values - ci_95).tolist(),
                "trend": "improving" if forecast_values[-1] > values[-1] else "declining",
                "change_percent": ((forecast_values[-1] - values[-1]) / values[-1] * 100) if values[-1] > 0 else 0,
                "mape": mape,
                "confidence": "high" if mape < 15 else "medium" if mape < 30 else "low"
            }

        except Exception:
            return self._generate_mock_data(dealer_name)


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class AnomalyDetector:
    """Detect anomalies in dealer metrics."""

    def __init__(self, contamination: float = 0.15):
        # Migration note: session parameter removed; uses athena_query() directly
        self.contamination = contamination

    def _generate_mock_anomalies(self, dealer_name: str) -> Dict:
        """Generate mock anomaly detection data for testing."""
        np.random.seed(42)

        base_revenue = 2500000
        revenues = base_revenue + np.random.normal(0, 250000, 24)
        revenues[5] = base_revenue * 0.5
        revenues[15] = base_revenue * 1.6

        growth_rates = []
        for i, r in enumerate(revenues):
            if i == 0:
                growth = 0
            else:
                prev = revenues[i-1]
                growth = ((r - prev) / prev * 100) if prev != 0 else 0
            growth_rates.append(growth)

        periods = [f"2025-{str(i % 12 + 1).zfill(2)}" for i in range(24)]

        anomalies = []
        for i, (period, revenue, growth) in enumerate(zip(periods, revenues, growth_rates)):
            if abs(revenue - base_revenue) > 2 * 250000 or abs(growth) > 30:
                severity = "critical" if revenue < base_revenue * 0.6 else "high"
                anomalies.append({
                    "period": period,
                    "revenue": float(revenue),
                    "growth_rate": float(growth),
                    "severity": severity,
                    "z_score": abs((revenue - base_revenue) / 250000) if 250000 > 0 else 0
                })

        return {
            "dealer": dealer_name,
            "total_periods": len(revenues),
            "anomalies_count": len(anomalies),
            "anomaly_rate": (len(anomalies) / len(revenues) * 100) if len(revenues) > 0 else 0,
            "risk_level": "high" if len(anomalies) > len(revenues) * 0.2 else "medium" if len(anomalies) > 0 else "low",
            "anomalies": sorted(anomalies, key=lambda x: x['z_score'], reverse=True)[:10],
            "mock": True
        }

    def detect_dealer_anomalies(self, dealer_name: str) -> Dict:
        """Identify anomalous periods in dealer performance."""
        query = f"""
        SELECT
            PERIOD_YEAR,
            PERIOD_MONTH,
            REVENUE
        FROM {_DB}.VW_DEALER_REVENUE_GROWTH
        WHERE DEALER_NAME = '{dealer_name}'
            AND REVENUE IS NOT NULL
        ORDER BY PERIOD_YEAR DESC, PERIOD_MONTH DESC
        """

        import sys
        try:
            msg = f"[ANOMALY] Running query for {dealer_name}..."
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

            df = athena_query(query)  # Migration note: session.sql().to_pandas() → athena_query()

            msg = f"[ANOMALY] Query returned {df.shape[0]} rows"
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

            if df.empty:
                msg = f"[ANOMALY] No data found for {dealer_name} - using mock data"
                print(msg, flush=True)
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
                return self._generate_mock_anomalies(dealer_name)

            if df.shape[0] < 3:
                msg = f"[ANOMALY] Insufficient rows ({df.shape[0]}) - using mock data"
                print(msg, flush=True)
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
                return self._generate_mock_anomalies(dealer_name)

            revenue_mean = df['REVENUE'].mean()
            revenue_std = df['REVENUE'].std()

            threshold_high = revenue_mean + (2 * revenue_std)
            threshold_low = revenue_mean - (2 * revenue_std)

            anomalies = []
            for idx, row in df.iterrows():
                revenue = float(row['REVENUE'])
                period_year = int(row['PERIOD_YEAR'])
                period_month = str(row['PERIOD_MONTH']).split('-')[1] if '-' in str(row['PERIOD_MONTH']) else 'XX'

                is_extreme = revenue > threshold_high or revenue < threshold_low

                if is_extreme:
                    if revenue < threshold_low:
                        severity = "critical"
                    else:
                        severity = "high"

                    anomalies.append({
                        "period": f"{period_year}-{period_month}",
                        "revenue": revenue,
                        "severity": severity,
                        "z_score": abs((revenue - revenue_mean) / revenue_std) if revenue_std > 0 else 0
                    })

            msg = f"[ANOMALY] SUCCESS: Found {len(anomalies)} anomalies from {len(df)} periods"
            print(msg, flush=True)
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()

            return {
                "dealer": dealer_name,
                "total_periods": len(df),
                "anomalies_count": len(anomalies),
                "anomaly_rate": (len(anomalies) / len(df) * 100) if len(df) > 0 else 0,
                "risk_level": "high" if len(anomalies) > len(df) * 0.2 else "medium" if len(anomalies) > 0 else "low",
                "anomalies": sorted(anomalies, key=lambda x: x['z_score'], reverse=True)[:10]
            }

        except Exception as e:
            error_msg = f"[ANOMALY] ERROR: {type(e).__name__}: {str(e)}"
            print(error_msg, flush=True)
            sys.stderr.write(error_msg + "\n")
            sys.stderr.flush()
            import traceback
            traceback.print_exc()
            return self._generate_mock_anomalies(dealer_name)


# ============================================================================
# COLOR SCHEME FROM CONFIGURATION
# ============================================================================
PRIMARY_COLOR = get_config_value("theme.colors.primary", "#2563eb")
PRIMARY_DARK = get_config_value("theme.colors.primary_dark", "#1e40af")
SECONDARY_COLOR = get_config_value("theme.colors.secondary", "#06b6d4")
SUCCESS_COLOR = get_config_value("theme.colors.success", "#10b981")
WARNING_COLOR = get_config_value("theme.colors.warning", "#f59e0b")
DANGER_COLOR = get_config_value("theme.colors.danger", "#ef4444")
NEUTRAL_COLOR = get_config_value("theme.colors.neutral", "#8b5cf6")
ACCENT_COLOR = get_config_value("theme.colors.accent", "#3b82f6")
ACCENT_LIGHT = get_config_value("theme.colors.accent_light", "#3b8beb")
GRID_COLOR = get_config_value("theme.colors.grid", "#e5e7eb")
TEXT_PRIMARY = get_config_value("theme.colors.text_primary", "#111827")
TEXT_MUTED = get_config_value("theme.colors.text_muted", "#6b7280")
TEXT_DARK = get_config_value("theme.colors.text_dark", "#222222")
BACKGROUND_COLOR = get_config_value("theme.colors.background", "#fbf4f9")
BORDER_COLOR = get_config_value("theme.colors.border", "#eef2f6")
SURFACE_COLOR = get_config_value("theme.colors.surface", "#ffffff")
SURFACE_LIGHT = get_config_value("theme.colors.surface_light", "#f5f5f5")
SURFACE_LIGHTER = get_config_value("theme.colors.surface_lighter", "#f3f4f6")
SURFACE_HOVER = get_config_value("theme.colors.surface_hover", "#eff6ff")
HOVER_SHADOW_COLOR = get_config_value("theme.colors.hover_shadow_color", "37, 99, 235")
HOVER_SHADOW_OPACITY = get_config_value("theme.colors.hover_shadow_opacity", "0.3")
INSIGHT_BG = get_config_value("theme.colors.insight_bg", "#f3e8ff")
INSIGHT_BORDER = get_config_value("theme.colors.insight_border", "#7c3aed")
BADGE_SUCCESS = get_config_value("theme.colors.badge_success", "#d1fae5")
BADGE_SUCCESS_TEXT = get_config_value("theme.colors.badge_success_text", "#065f46")
BADGE_DANGER = get_config_value("theme.colors.badge_danger", "#fee2e2")
BADGE_DANGER_TEXT = get_config_value("theme.colors.badge_danger_text", "#7f1d1d")

# ============================================================================
# FONT CONFIGURATION
# ============================================================================
FONT_FAMILY = get_config_value("theme.fonts.family", "Poppins, Inter, Segoe UI, sans-serif")
FONT_SIZE_LARGE = get_config_value("theme.fonts.size_large", 24)
FONT_SIZE_TITLE = get_config_value("theme.fonts.size_title", 18)
FONT_SIZE_NORMAL = get_config_value("theme.fonts.size_normal", 14)
FONT_SIZE_SMALL = get_config_value("theme.fonts.size_small", 12)
FONT_WEIGHT_BOLD = get_config_value("theme.fonts.weight_bold", 700)
FONT_WEIGHT_SEMI_BOLD = get_config_value("theme.fonts.weight_semi_bold", 600)
FONT_WEIGHT_NORMAL = get_config_value("theme.fonts.weight_normal", 400)

# ============================================================================
# BUTTON CONFIGURATION
# ============================================================================
BTN_PRIMARY_BG = get_config_value("buttons.primary.bg_color", "#2563eb")
BTN_PRIMARY_BG_HOVER = get_config_value("buttons.primary.bg_color_hover", "#1e40af")
BTN_PRIMARY_TEXT = get_config_value("buttons.primary.text_color", "#ffffff")
BTN_PRIMARY_TEXT_HOVER = get_config_value("buttons.primary.text_color_hover", "#ffffff")
BTN_PRIMARY_BORDER = get_config_value("buttons.primary.border_color", "#2563eb")
BTN_PRIMARY_BORDER_WIDTH = get_config_value("buttons.primary.border_width", "0px")

BTN_SECONDARY_BG = get_config_value("buttons.secondary.bg_color", "#ffffff")
BTN_SECONDARY_BG_HOVER = get_config_value("buttons.secondary.bg_color_hover", "#eff6ff")
BTN_SECONDARY_TEXT = get_config_value("buttons.secondary.text_color", "#222222")
BTN_SECONDARY_TEXT_HOVER = get_config_value("buttons.secondary.text_color_hover", "#1e40af")
BTN_SECONDARY_BORDER = get_config_value("buttons.secondary.border_color", "#eef2f6")
BTN_SECONDARY_BORDER_HOVER = get_config_value("buttons.secondary.border_color_hover", "#2563eb")
BTN_SECONDARY_BORDER_WIDTH = get_config_value("buttons.secondary.border_width", "1px")

BTN_PADDING_VERTICAL = get_config_value("buttons.sizing.padding_vertical", "10px")
BTN_PADDING_HORIZONTAL = get_config_value("buttons.sizing.padding_horizontal", "20px")
BTN_BORDER_RADIUS = get_config_value("buttons.sizing.border_radius", "25px")
BTN_MIN_HEIGHT = get_config_value("buttons.sizing.min_height", "auto")

BTN_FONT_WEIGHT = get_config_value("buttons.styling.font_weight", 600)
BTN_FONT_SIZE = get_config_value("buttons.styling.font_size", 14)
BTN_TRANSITION = get_config_value("buttons.styling.transition_duration", "0.3s")
BTN_TRANSITION_TIMING = get_config_value("buttons.styling.transition_timing", "ease")
BTN_SHADOW = get_config_value("buttons.styling.box_shadow", "none")
BTN_SHADOW_HOVER = get_config_value("buttons.styling.box_shadow_hover", "0 4px 12px rgba(139, 92, 246, 0.3)")

st.markdown("""
<style>
/* Hide the Material icon span inside ALL expanders */
[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {
  display: none !important;
}

/* Also hide toggle icon container if present (Streamlit versions differ) */
[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] {
  display: none !important;
}

/* Remove the empty spacing reserved for the icon */
[data-testid="stExpander"] summary {
  padding-left: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* ========== BASE TYPOGRAPHY & LAYOUT ========== */
    html, body, [data-testid="stAppViewContainer"], * {
        font-family: """ + FONT_FAMILY + """ !important;
    }
    span[data-testid="stIconMaterial"] { font-family: "Material Icons" !important; }
    [data-testid="stMetricValue"] {
        font-size: """ + str(FONT_SIZE_LARGE) + """px;
        font-weight: """ + str(FONT_WEIGHT_BOLD) + """;
    }
    [data-testid="stMetricLabel"] {
        font-size: """ + str(FONT_SIZE_NORMAL) + """px;
    }
    .kpi-card {
        margin-bottom: 18px;
        margin-top: 18px;
        background: linear-gradient(135deg, """ + SURFACE_LIGHT + """ 0%, """ + SURFACE_COLOR + """ 100%);
        border-radius: 8px;
        padding: 1.5rem;
        border-left: 4px solid """ + ACCENT_LIGHT + """;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: """ + str(FONT_SIZE_NORMAL) + """px;
        padding: 0.75rem 1.5rem;
        border-radius: 6px 6px 0 0;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """;
    }
    .insight-box {
        background-color: """ + INSIGHT_BG + """;
        border-left: 4px solid """ + INSIGHT_BORDER + """;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .severity-high {
        color: """ + DANGER_COLOR + """;
        font-weight: """ + str(FONT_WEIGHT_BOLD) + """;
    }
    .severity-medium {
        color: """ + WARNING_COLOR + """;
        font-weight: """ + str(FONT_WEIGHT_BOLD) + """;
    }
    .severity-low {
        color: """ + SUCCESS_COLOR + """;
        font-weight: """ + str(FONT_WEIGHT_BOLD) + """;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: """ + str(FONT_SIZE_SMALL) + """px;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """;
    }
    .badge-positive {
        background-color: """ + BADGE_SUCCESS + """;
        color: """ + BADGE_SUCCESS_TEXT + """;
    }
    .badge-negative {
        background-color: """ + BADGE_DANGER + """;
        color: """ + BADGE_DANGER_TEXT + """;
    }
    .header-nav-button {
        background-color: """ + PRIMARY_COLOR + """;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-size: """ + str(FONT_SIZE_NORMAL) + """px;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .header-nav-button:hover {
        background-color: """ + PRIMARY_DARK + """;
        transform: none;
        box-shadow: none;
    }
    .attention-card, .priority-card {
        background-color: """ + SURFACE_COLOR + """;
        border-radius: 12px;
        border: 1px solid """ + GRID_COLOR + """;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* UNIFIED BUTTON STYLING - All buttons use this */
    button, [data-testid="stButton"] button, div.stButton > button {
        border-radius: """ + str(BTN_BORDER_RADIUS) + """ !important;
        padding: """ + str(BTN_PADDING_VERTICAL) + """ """ + str(BTN_PADDING_HORIZONTAL) + """ !important;
        font-weight: """ + str(BTN_FONT_WEIGHT) + """ !important;
        font-size: """ + str(BTN_FONT_SIZE) + """px !important;
        transition: all """ + str(BTN_TRANSITION) + """ """ + str(BTN_TRANSITION_TIMING) + """ !important;
        height: """ + str(BTN_MIN_HEIGHT) + """ !important;
        box-shadow: """ + str(BTN_SHADOW) + """ !important;
    }

    /* Primary buttons (selected) - Blue background */
    button[type="primary"],
    [data-testid="stButton"] button[type="primary"],
    div.stButton > button[type="primary"] {
        background-color: """ + BTN_PRIMARY_BG + """ !important;
        color: """ + BTN_PRIMARY_TEXT + """ !important;
        border: """ + str(BTN_PRIMARY_BORDER_WIDTH) + """ solid """ + BTN_PRIMARY_BORDER + """ !important;
    }

    /* Form submit (Create PO / Send etc.) — light grey */
    button[kind="primaryFormSubmit"],
    [data-testid="baseButton-primaryFormSubmit"],
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stForm"] button:first-of-type {
        background-color: #e2e8f0 !important;
        background: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
        color: #1e293b !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.08) !important;
    }
    button[kind="primaryFormSubmit"]:hover,
    [data-testid="baseButton-primaryFormSubmit"]:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stForm"] button:first-of-type:hover {
        background-color: #cbd5e1 !important;
        background: #cbd5e1 !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        box-shadow: 0 2px 6px rgba(15,23,42,0.12) !important;
    }

    button[type="primary"]:hover,
    [data-testid="stButton"] button[type="primary"]:hover,
    div.stButton > button[type="primary"]:hover,
    button[type="primary"]:active,
    [data-testid="stButton"] button[type="primary"]:active,
    div.stButton > button[type="primary"]:active,
    button[type="primary"]:focus,
    [data-testid="stButton"] button[type="primary"]:focus,
    div.stButton > button[type="primary"]:focus {
        background-color: """ + BTN_PRIMARY_BG_HOVER + """ !important;
        color: """ + BTN_PRIMARY_TEXT_HOVER + """ !important;
        box-shadow: """ + str(BTN_SHADOW_HOVER) + """ !important;
    }

    /* Secondary buttons - Turn blue when selected/active */
    button[type="secondary"],
    [data-testid="stButton"] button[type="secondary"],
    div.stButton > button[type="secondary"] {
        background-color: """ + BTN_SECONDARY_BG + """ !important;
        color: """ + BTN_SECONDARY_TEXT + """ !important;
        border: """ + str(BTN_SECONDARY_BORDER_WIDTH) + """ solid """ + BTN_SECONDARY_BORDER + """ !important;
    }

    button[type="secondary"]:hover,
    [data-testid="stButton"] button[type="secondary"]:hover,
    div.stButton > button[type="secondary"]:hover {
        border-color: """ + BTN_SECONDARY_BORDER_HOVER + """ !important;
        background-color: """ + BTN_SECONDARY_BG_HOVER + """ !important;
        color: """ + BTN_SECONDARY_TEXT_HOVER + """ !important;
        box-shadow: """ + str(BTN_SHADOW_HOVER) + """ !important;
    }

    button[type="secondary"]:active,
    [data-testid="stButton"] button[type="secondary"]:active,
    div.stButton > button[type="secondary"]:active,
    button[type="secondary"]:focus,
    [data-testid="stButton"] button[type="secondary"]:focus,
    div.stButton > button[type="secondary"]:focus {
        background-color: """ + BTN_PRIMARY_BG + """ !important;
        color: """ + BTN_PRIMARY_TEXT + """ !important;
        border-color: """ + BTN_PRIMARY_BG + """ !important;
        box-shadow: """ + str(BTN_SHADOW_HOVER) + """ !important;
    }

    /* Styling for the section wrappers */
    .stColumn > div[data-testid="stVerticalBlock"] > div.section-container {
        background-color: """ + SURFACE_COLOR + """;
        border-radius: 12px;
        border: 1px solid """ + GRID_COLOR + """;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        min-height: 550px;
        display: flex;
        flex-direction: column;
    }

    /* Radio button styling for metric and time period filters */
    [data-testid="stRadio"] {
        display: flex !important;
        gap: 8px !important;
    }

    /* Unselected radio button labels - white background with dark text */
    [data-testid="stRadio"] > div > label {
        padding: 8px 16px !important;
        border-radius: 20px !important;
        border: none !important;
        background-color: """ + SURFACE_COLOR + """ !important;
        color: """ + TEXT_DARK + """ !important;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """ !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        margin-right: 0px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }

    /* Ensure all text elements in unselected state are dark */
    [data-testid="stRadio"] > div > label div,
    [data-testid="stRadio"] > div > label span,
    [data-testid="stRadio"] > div > label p {
        color: """ + TEXT_DARK + """ !important;
    }

    /* Selected radio button - blue background */
    [data-testid="stRadio"] > div > label:has(> input:checked) {
        background-color: """ + PRIMARY_COLOR + """ !important;
        box-shadow: 0 2px 8px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    /* Force ALL nested text elements to white when checked */
    [data-testid="stRadio"] > div > label:has(> input:checked),
    [data-testid="stRadio"] > div > label:has(> input:checked) div,
    [data-testid="stRadio"] > div > label:has(> input:checked) span,
    [data-testid="stRadio"] > div > label:has(> input:checked) p,
    [data-testid="stRadio"] > div > label:has(> input:checked) * {
        color: white !important;
    }

    /* Hover state */
    [data-testid="stRadio"] > div > label:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.15) !important;
    }

    /* ===== All Pill / Toggle Buttons ===== */
    div.stButton > button {
        border-radius: 999px !important;
        padding: 0.55rem 1.3rem !important;
        border: none !important;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """ !important;
        transition: all 0.18s ease !important;
        background: """ + GRID_COLOR + """ !important;
        color: """ + TEXT_PRIMARY + """ !important;
    }

    /* Hover - turn blue */
    div.stButton > button:hover {
        background: """ + PRIMARY_COLOR + """ !important;
        color: white !important;
        transform: translateY(-1px) !important;
    }

    /* Active / pressed - blue background */
    div.stButton > button:active,
    div.stButton > button:focus {
        background: """ + PRIMARY_COLOR + """ !important;
        color: white !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    /* ===== RECENT ANALYSIS QUESTION BUTTONS ===== */
    .recent-analysis-section button {
        border-radius: 999px !important;
        padding: 0.55rem 1.3rem !important;
        border: none !important;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """ !important;
        transition: all 0.18s ease !important;
        background: """ + SURFACE_LIGHTER + """ !important;
        color: """ + TEXT_DARK + """ !important;
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .recent-analysis-section button > div {
        justify-content: flex-start !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .recent-analysis-section button > div > span {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .recent-analysis-section button > div > span > div {
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .recent-analysis-section button > div > span > div > p {
        text-align: left !important;
        margin: 0 !important;
        display: block !important;
    }

    .recent-analysis-section button * {
        text-align: left !important;
    }

    .recent-analysis-section button div,
    .recent-analysis-section button span,
    .recent-analysis-section button p {
        text-align: left !important;
        justify-content: flex-start !important;
    }

    .recent-analysis-section [class*="st-emotion"],
    .recent-analysis-section [class*="e27ue8x"],
    .recent-analysis-section [class*="eg78z5t0"] {
        justify-content: flex-start !important;
        text-align: left !important;
    }

    .recent-analysis-section [data-testid="stMarkdownContainer"] {
        text-align: left !important;
        justify-content: flex-start !important;
        display: block !important;
    }

    .recent-analysis-section [data-testid="stMarkdownContainer"][class*="st-emotion"],
    .recent-analysis-section [data-testid="stMarkdownContainer"][class*="eg78z"] {
        text-align: left !important;
        display: block !important;
    }

    .recent-analysis-section button p,
    .recent-analysis-section [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        display: block !important;
        margin: 0 !important;
        justify-content: flex-start !important;
    }

    .recent-analysis-section button:hover {
        background: """ + PRIMARY_COLOR + """ !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    .recent-analysis-section button:active,
    .recent-analysis-section button:focus {
        background: """ + PRIMARY_DARK + """ !important;
        color: white !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    .recent-analysis-section [data-testid="stButton"] {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    .recent-analysis-section [data-testid="stButton"] button {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }

    .recent-analysis-section [data-testid="stMarkdownContainer"] {
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    .recent-analysis-section [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        margin: 0 !important;
    }

    .recent-analysis-section button {
        text-align: left !important;
    }

    .recent-analysis-section button:is([class*="e27ue8x"]) {
        justify-content: flex-start !important;
        text-align: left !important;
    }

    /* ===== SUGGESTED QUESTIONS BUTTONS ===== */
    .suggested-questions-section button {
        border-radius: 999px !important;
        padding: 0.55rem 1.3rem !important;
        border: none !important;
        font-weight: """ + str(FONT_WEIGHT_SEMI_BOLD) + """ !important;
        transition: all 0.18s ease !important;
        background: """ + SURFACE_LIGHTER + """ !important;
        color: """ + TEXT_DARK + """ !important;
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .suggested-questions-section button > div {
        justify-content: flex-start !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .suggested-questions-section button > div > span {
        text-align: left !important;
        justify-content: flex-start !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .suggested-questions-section button > div > span > div {
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
    }

    .suggested-questions-section button > div > span > div > p {
        text-align: left !important;
        margin: 0 !important;
        display: block !important;
    }

    .suggested-questions-section button * {
        text-align: left !important;
    }

    .suggested-questions-section button div,
    .suggested-questions-section button span,
    .suggested-questions-section button p {
        text-align: left !important;
        justify-content: flex-start !important;
    }

    .suggested-questions-section [class*="st-emotion"],
    .suggested-questions-section [class*="e27ue8x"],
    .suggested-questions-section [class*="eg78z5t0"],
    .suggested-questions-section [class*="1c9yjad"]{
        justify-content: flex-start !important;
        text-align: left !important;
    }

    .suggested-questions-section [data-testid="stMarkdownContainer"] {
        text-align: left !important;
        justify-content: flex-start !important;
        display: block !important;
    }

    .suggested-questions-section [data-testid="stMarkdownContainer"][class*="st-emotion"],
    .suggested-questions-section [data-testid="stMarkdownContainer"][class*="eg78z"],
    .suggested-questions-section [data-testid="stMarkdownContainer"][class*="1c9yjad"] {
        text-align: left !important;
        display: block !important;
    }

    .suggested-questions-section button p,
    .suggested-questions-section [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        display: block !important;
        margin: 0 !important;
        justify-content: flex-start !important;
    }

    .suggested-questions-section button:hover {
        background: """ + PRIMARY_COLOR + """ !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    .suggested-questions-section button:active,
    .suggested-questions-section button:focus {
        background: """ + PRIMARY_DARK + """ !important;
        color: white !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(""" + HOVER_SHADOW_COLOR + """, """ + HOVER_SHADOW_OPACITY + """) !important;
    }

    .suggested-questions-section [data-testid="stButton"] {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    .suggested-questions-section [data-testid="stButton"] button {
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }

    .suggested-questions-section [data-testid="stMarkdownContainer"] {
        text-align: left !important;
        width: 100% !important;
        display: flex !important;
        justify-content: flex-start !important;
    }

    .suggested-questions-section [data-testid="stMarkdownContainer"] p {
        text-align: left !important;
        margin: 0 !important;
    }

    .suggested-questions-section button {
        text-align: left !important;
    }

    .suggested-questions-section button:is([class*="e27ue8x"]) {
        justify-content: flex-start !important;
        text-align: left !important;
    }

    /* Analysis content styling */
    .analysis-content {
        font-size: """ + str(FONT_SIZE_NORMAL) + """px;
        color: """ + TEXT_PRIMARY + """;
        line-height: 1.6;
        text-align: left;
        padding: 12px 14px;
        margin: 0;
        display: block;
        word-break: break-word;
        font-weight: """ + str(FONT_WEIGHT_NORMAL) + """;
    }

    /* Make the descriptive/prescriptive/predictive boxes consistent */
    .insight-box, .section-container {
        padding: 16px;
        border-radius: 8px;
        background-color: """ + SURFACE_COLOR + """;
        box-shadow: 0 1px 6px rgba(0,0,0,0.06);
        max-width: 100%;
    }

    /* Optional: make the three D/P/P expanders visually tighter & aligned */
    .stExpander > div[data-testid="stExpanderContent"] .analysis-content {
        margin-top: 6px;
    }

    /* Ensure code block / SQL container uses monospace and wraps */
    .sql-block {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, 'Courier New', monospace;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: """ + str(FONT_SIZE_SMALL) + """px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'show_insights' not in st.session_state:
    st.session_state.show_insights = True

# ============================================================================
# GENIE OPTIMIZATION INITIALIZATION
# ============================================================================
if 'genie_cache' not in st.session_state:
    st.session_state.genie_cache = None
    st.session_state.genie_cache_initialized = False

if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None

if 'detector' not in st.session_state:
    st.session_state.detector = None

# ============================================================================
# DECISION SUPPORT INSTRUCTION FOR CORTEX ANALYST
# ============================================================================

DECISION_SUPPORT_INSTRUCTION = f"""
You are a Dealer Performance Analyst AI with programmatic access to the {_DB} database.

Purpose: Produce accurate, auditable, and runnable analyses that map directly to views present in the semantic model.

RESPONSE FORMAT (MANDATORY):
- Every response MUST contain three named sections in this exact order and format. Use XML-style tags when returning runnable SQL.
  1) <DESCRIPTIVE> ... </DESCRIPTIVE>
  2) <PRESCRIPTIVE> ... </PRESCRIPTIVE>
  3) <PREDICTIVE> ... </PREDICTIVE>

DESCRIPTIVE: Only facts, metrics, and short observations derived from the query results. Cite exact numbers, dealer identifiers (dealer_name  or DEALER_NAME), and time periods. If a metric is not available, explicitly state which table or column is missing and why the analysis cannot be completed.

PRESCRIPTIVE: Provide 3–5 specific, prioritized actions. For each action include: (1) the exact finding with numeric evidence (value + comparison), (2) the actor (dealer, regional ops, procurement), and (3) a concrete next step with expected measurable outcome (e.g., reduce DIO by X days -> free Y working capital).

PREDICTIVE: Give a short forecast (30–90 days) tied to current metrics, list assumptions, and give a confidence level (Low/Medium/High). Quantify the likely impact where possible (e.g., estimated lost revenue $ or % if no action taken).

STRICT ANALYST RULES FOR SQL GENERATION (APPLY ALWAYS):
1. ALWAYS use fully-qualified view names in the form: {_DB}.VW_<VIEW_NAME>. Do not use unqualified table names or synonyms.
2. ONLY reference tables that appear in the provided semantic model. If a required table/column is not present, state that explicitly and propose an alternative query using available tables.
3. NEVER use SELECT * in production queries. Explicitly select needed columns and apply tight filters and limits when returning large result sets.
4. AGGREGATION SAFETY:
    - When a view can have multiple rows per dealer (orders, lead times, stock by category), pre-aggregate that view in a CTE using the appropriate aggregation (AVG, SUM, COUNT) before joining to other aggregated tables.
    - Do not mix raw row-level columns and aggregated expressions in the same SELECT without grouping or pre-aggregation.
    - Prefer window functions for ranking/percentile calculations, but still pre-aggregate when joining to other aggregates.
5. ALIASES & NAMING:
    - Always use explicit aliases introduced with `AS` and choose descriptive alias names (e.g., avg_lead_days, stock_avail_pct).
    - Ensure alias uniqueness: do not reuse the same alias for multiple expressions. If you need multiple measures of the same base metric, append a descriptive suffix (e.g., avg_lead_time_30d, avg_lead_time_90d).
    - Avoid ambiguous aliases in ORDER BY / GROUP BY / HAVING; reference the full alias name rather than positional indexes.
6. JOIN KEYS & PREFERRED JOINS:
    - Inventory & backorder views: prefer joining on  (VW_STOCK_AVAILABILITY_DEALER.Dealer_name and VW_BACKORDER_INCIDENCE.DEALER_NAME).
    - Sales and operational metrics: join on `dealer_name` when available (VW_SALES_PER_PRODUCT_CATEGORY, VW_ORDER_LEAD_TIME, VW_GROSS_PROFIT_MARGIN, VW_DEALER_CONTRIBUTION_MARGIN).
    - Revenue growth and sales volume: join on `dealer_name` (VW_DEALER_REVENUE_GROWTH, VW_SALES_VOLUME).
    - Financial efficiency (CCC): prefer `DEALER_NAME` only when `dealer_name`/`DEALER_NAME` is not available across the participating views.
    - Explicitly state which join key is used in the SQL comment and only use a second key when a deterministic mapping is provided in the semantic model.
7. WINDOW & CTE PATTERN:
    - When producing leaderboards, top-N, or YoY comparisons, compute base aggregates in CTEs and then use window functions for ratios and rankings in a final SELECT.
8. DEFAULT LIMITS & SAFETY:
    - For exploratory responses that return raw rows, apply `LIMIT 1000` unless the user explicitly requests a larger export.
    - For time-bounded analyses, always include FROM/TO filters when dates exist; use the provided `ORDER_DATE`, `PERIOD_YEAR`, `CREATED_AT` columns where applicable.
9. RESULTS & EXPLANATIONS:
    - When returning SQL, wrap the complete runnable statement with <SQL>...</SQL> tags.
    - Add a one-line plain-text rationale above the SQL explaining why this query answers the question.
    - If returning a numerical recommendation, cite the exact SQL column and value used to compute it.
10. ERROR HANDLING:
    - If a GROUP BY error would occur because a column is selected but not grouped, rewrite the query to pre-aggregate that column in a CTE.
    - If you must use `DEALER_NAME` as a join key, call that out and explain potential duplication risks.

SEMANTIC MODEL USAGE GUIDELINES:
- Use the semantic model's `dimensions`, `time_dimensions`, and `facts` metadata to map natural-language column references to actual column expressions.
- Prefer the canonical metric names declared under `facts` (e.g., STOCK_AVAILABILITY_PCT, BACKORDER_INCIDENCE_PCT, ORDER_LEAD_TIME_DAYS, GROSS_PROFIT_MARGIN_PCT).
- Use synonyms mapping to interpret user queries but always output the actual column names used in the SQL.

EXAMPLES (Follow these patterns exactly):
1) Inventory health (pre-aggregate then join):
    -- Rationale: pre-aggregate stock and backorder by dealer to avoid GROUP BY conflicts
    <SQL>
    WITH stock AS (
      SELECT dealer_name, DEALER_NAME, AVG(STOCK_AVAILABILITY_PCT) AS avg_stock_avail
      FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER
      GROUP BY dealer_name, DEALER_NAME
    ),
    backorder AS (
      SELECT dealer_name, AVG(BACKORDER_INCIDENCE_PCT) AS avg_backorder
      FROM {_DB}.VW_BACKORDER_INCIDENCE
      GROUP BY dealer_name
    )
    SELECT s.dealer_name, s.DEALER_NAME, s.avg_stock_avail, COALESCE(b.avg_backorder,0) AS avg_backorder
    FROM stock s
    LEFT JOIN backorder b ON s.dealer_name = b.dealer_name
    WHERE s.avg_stock_avail < 85 OR COALESCE(b.avg_backorder,0) > 15
    ORDER BY s.avg_stock_avail ASC
    LIMIT 1000;
    </SQL>

2) Order lead time (pre-aggregate, unique aliases):
    -- Rationale: compute average lead time per dealer and rank
    <SQL>
    WITH avg_lead AS (
      SELECT dealer_name, ROUND(AVG(AVG_ORDER_LEAD_TIME_DAYS),1) AS avg_lead_days
      FROM {_DB}.VW_ORDER_LEAD_TIME
      WHERE AVG_ORDER_LEAD_TIME_DAYS IS NOT NULL
      GROUP BY dealer_name
    )
    SELECT al.dealer_name, al.avg_lead_days
    FROM avg_lead al
    ORDER BY al.avg_lead_days DESC
    LIMIT 500;
    </SQL>

FINAL NOTES FOR THE ANALYST:
- Always prefer returning a verified query template from `verified_queries` when available. If using a template, adapt only the WHERE filters and date range; do not alter join keys or aggregation logic.
- If the model returns SQL, ensure orthogonal safety steps before execution: (1) validate table names against the semantic model; (2) correct column synonyms; (3) ensure aliases are unique; (4) enforce a safe LIMIT if returning raw rows.
- When recommending actions, always back them with the exact SQL snippet or value used to compute the recommendation.

If these constraints cannot be satisfied, respond with a clear explanation of the missing semantic element and suggest an alternative using available tables/columns.
"""

# ============================================================================
# SNOWFLAKE CONFIGURATION & CONNECTION — REMOVED
# Migration note: get_snowflake_connection() → get_aws_session() from config_loader.py
# Migration note: load_semantic_model() → load_yaml_model() in ai_service.py
# ============================================================================

# ============================================================================
# LOCAL SEMANTIC MODEL HELPERS
# ============================================================================

_local_semantic_cache = None
def _load_local_semantic_model():
    """Load semantic model from local YAML file as a fallback for column resolution."""
    global _local_semantic_cache
    if _local_semantic_cache is not None:
        return _local_semantic_cache
    try:
        with open('DEALER_SEMANTIC_MODEL_V4.yml', 'r', encoding='utf-8') as f:
            _local_semantic_cache = yaml.safe_load(f)
            return _local_semantic_cache
    except Exception:
        _local_semantic_cache = None
        return None

def get_expr_column(table_name, logical_column_name):
    """Return the physical column/expression for a logical column name from the semantic model.

    If no mapping is found, returns the logical_column_name unchanged so callers can continue.
    """
    model = _load_local_semantic_model()
    if not model:
        return logical_column_name
    for tbl in model.get('tables', []):
        if tbl.get('name') == table_name:
            for dim in tbl.get('dimensions', []):
                if dim.get('name') == logical_column_name:
                    return dim.get('expr', logical_column_name)
    return logical_column_name


def dealer_filter_clause(view_name, filters):
    """Return a SQL filter clause for the dealer name based on semantic model resolution.

    Example: returns " AND dealer_name_NAME = 'Dealer X'" or " AND DEALER_NAME = 'Dealer X'"
    """
    if not filters or 'dealer' not in filters or filters['dealer'] == 'All Dealers':
        return ''
    col = get_expr_column(view_name, 'DEALER_NAME')
    return f" AND {col} = '{filters['dealer']}'"


# ============================================================================
# DATA LOADING & CACHING
# Migration note: All fetch_* functions, validate_view_schemas, fetch_dealer_health_scores,
# fetch_at_risk_dealers, fetch_at_risk_dealers_list, fetch_order_fulfillment, fetch_avg_tat,
# fetch_revenue_metrics, fetch_sales_vs_target, fetch_strategic_insights,
# generate_kpi_alerts, fetch_attention_items, fetch_dealers,
# fetch_cash_conversion_cycle, fetch_repair_turnaround_time, fetch_revenue_growth,
# fetch_gross_profit_margin, fetch_sales_per_product_category, fetch_order_lead_time,
# fetch_stock_availability, fetch_sales_volume, fetch_contribution_margin,
# fetch_backorder_incidence, fetch_journey_counts, fetch_transaction_lineage,
# fetch_regions, fetch_products, generate_dynamic_insights
# are all in data_service.py — skipped here.
# Mock data generators (generate_mock_*) also skipped — callers are in data_service.py.
# Source lines 3692–4893 skipped.
# ============================================================================


def lineage_filter_clause(filters):
    """Return additional SQL filter clauses for transaction lineage queries.

    Supports filtering by transaction_id (exact match), paid flag (Y/N),
    warranty status, and invoice_status. The caller is responsible for
    prefixing with AND/WHERE as needed.
    """
    if not filters:
        return ''
    clause = ''
    if filters.get('transaction_id'):
        t = str(filters['transaction_id']).replace("'", "''")
        clause += f" AND TRANSACTION_ID = '{t}'"
    if filters.get('paid') in ['Y', 'N']:
        clause += f" AND PAID_FLAG = '{filters['paid']}'"
    if filters.get('warranty_status'):
        ws = str(filters['warranty_status']).upper().replace("'", "''")
        clause += f" AND UPPER(WARRANTY_STATUS) = '{ws}'"
    if filters.get('invoice_status') and filters['invoice_status'] != 'All':
        inv = str(filters['invoice_status']).replace("'", "''")
        clause += f" AND UPPER(INVOICE_STATUS) = UPPER('{inv}')"
    return clause


# ============================================================================
# HEADER SECTION WITH LOGO & NAVIGATION
# ============================================================================
def render_header(subtitle = ""):
    """Render the header with logo and navigation bar"""
    header_col1, header_col2, header_col3 = st.columns([1.5, 3, 1.5])

    # Get title and subtitle from config if not provided
    if not subtitle:
        subtitle = get_config_value("app.subtitle", "Dealer Performance Analytics")
    app_title = get_config_value("app.title", "DealerPulse")

    with header_col1:

        st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            justify-content: flex-start;
            height: 100%;
            border-radius: 8px;
            gap: 10px;
        ">
            <div style="font-size: {FONT_SIZE_LARGE + 16}px; color: #111;"></div>
            <div>
                <div style="font-size: {FONT_SIZE_LARGE}px; font-weight: 700; color: #111; margin: 0;">
                    {app_title}
                </div>
                <div style="font-size: {FONT_SIZE_TITLE}px; color: #6b7280; font-weight: 500; margin: 2px 0 0 0;">
                    {subtitle}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with header_col2:
        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([0.22, 0.15, 0.25, 0.22])

        with nav_col1:
            dashboard_active = st.session_state.get('current_page', 'Dashboard') == 'Dashboard'
            if st.button("Dashboard", key="header_nav_dashboard", width='stretch'):
                st.session_state.current_page = 'Dashboard'
                st.rerun()

        with nav_col2:
            genie_active = st.session_state.get('current_page', 'Dashboard') == 'Genie'
            if st.button("Genie", key="header_nav_genie", width='stretch'):
                st.session_state.current_page = 'Genie'
                st.rerun()

        with nav_col3:
            dlc_active = st.session_state.get('current_page', 'Dashboard') == 'Dealer Life Cycle'
            if st.button("Dealer Health", key="header_nav_dlc", width='stretch'):
                st.session_state.current_page = 'Dealer Life Cycle'
                st.rerun()

        with nav_col4:
            agent_active = st.session_state.get('current_page', 'Dashboard') == 'AI Agents'
            if st.button("AI Agents", key="header_nav_agent_ai", width='stretch'):
                st.session_state.current_page = 'AI Agents'
                st.rerun()

    with header_col3:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-end; gap: 12px; height: 100%;">
            <img src="data:image/png;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/4gHYSUNDX1BST0ZJTEUAAQEAAAHIAAAAAAQwAABtbnRyUkdCIFhZWiAH4AABAAEAAAAAAABhY3NwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQAA9tYAAQAAAADTLQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlkZXNjAAAA8AAAACRyWFlaAAABFAAAABRnWFlaAAABKAAAABRiWFlaAAABPAAAABR3dHB0AAABUAAAABRyVFJDAAABZAAAAChnVFJDAAABZAAAAChiVFJDAAABZAAAAChjcHJ0AAABjAAAADxtbHVjAAAAAAAAAAEAAAAMZW5VUwAAAAgAAAAcAHMAUgBHAEJYWVogAAAAAAAAb6IAADj1AAADkFhZWiAAAAAAAABimQAAt4UAABjaWFlaIAAAAAAAACSgAAAPhAAAts9YWVogAAAAAAAA9tYAAQAAAADTLXBhcmEAAAAAAAQAAAACZmYAAPKnAAANWQAAE9AAAApbAAAAAAAAAABtbHVjAAAAAAAAAAEAAAAMZW5VUwAAACAAAAAcAEcAbwBvAGcAbABlACAASQBuAGMALgAgADIAMAAxADb/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAAkADsDAREAAhEBAxEB/8QAGwAAAgIDAQAAAAAAAAAAAAAABggFBwMJCgT/xAAlEAACAgMBAAIBBQEBAAAAAAAEBQMGAQIHCAkUEgARExUWFyT/xAAdAQABBAMBAQAAAAAAAAAAAAAHAwQFBgIICQAB/8QAKBEAAgMAAQQCAgMAAwEAAAAAAgMBBAUGBxESEwAUCCEVIiMWMTIk/9oADAMBAAIRAxEAPwDqC0+Wao0e8+oOa+jeWY4ndfOySd0kUwdC0ukfX9dToAloVTnIpdNlENsmruoNa7BIKwkJQvjHLLRSJXWe2BFHVmpRv8nzOR5X8Jd48gnIVGh92NjsYgsKhFSpyB2YdUbXiRZJ13m5kKCu35uuX4ZbXIePdI+V9LuXzz7A6naC6GhdZxksEuETNdli1Y2Vq3twX18uaG3T02C6qCtLNRQqldfqU4nLbflCsHOvItD9M9C82EV21dhu4Va45xODqeWdhutcLE1M1uhjfHNgzEQUsEJe4C6GqO5jdS6tvsUNpaBsh5W+qFjO4jQ5NocbKva2LwVsbEHV9ti7WMPOLpujNA0BIiUrWNV8nB1Z8x+0Phhi/iNm8n618j6T8a6qK08fg/H36vOefM4f9TMwdRDpROCilPKno0LAMNI2bTNmgFeUbAwlpZDYfclG+RHn1k8KMfcVgqbBAiQjNYLBz9Q6GsrZdZRbppSFFcidzrq6ORK8ZMK+VEaQsA0EBdwzyDy6w/vNMUeomfZ4K3nFioyuiuLRsZ6XjacuyN2KKa0PJdcSl7GVzgyUvwW+CkZgf7UbkH4x8ly/yHq/j7mbNbS0NFtNmbyW7Qbk0rOU7BLkF7UOgu1psWGfVraSTQq3YJ9jPNYNCWdgpfz38kXaOw37ky6weObCu5V2mAMivdP5Zf4u1r6NowIlFC/6lHVKpEHTv45dYpXsNkOrjOtg7TGHKptQ2EYULx/qRta+hkrscOsLytoVlW08q/G0FGGlIB/KRUqwFPtPYrA2TrNrB5GaigGQF76mfixwPhHHOZ2szrlmWuY8CY9enxHmHGz4DZ5DNVYOf/xA9nZN+55BJhnnlV9Wpq2IBFa4EvqlYKuQe2Kd6ovfrWgWCrWHlKbzDYj0I1ySd6siBledAnt7UwMBkyUKl4QzTCUWVjIvmY2mGDdjqERLNiLBBdj4j1SDQ1OV1f4vPqI4xZYiH6E0NGNIQffVBRWu5wTVkgoyyQBziCWQHs/rBEP+rn4rXOEcU6M7aeUavIL/AFcyq2iWXk5GnkRxlj87jlw0npZ+7eHVFTuQBWF7KWdDhqk+ERDSUlP/ADcN6Z96cUufbvN/rn0Z5UsyDojikJqd0e4UP0zztuQnrNWsOpUm9k47TbSnEmns0QE2+7S06aYDIk+nPJttBqUuknWzjXMsx1/lPSTjLKCNFmac5l3Ro6jIXXq2GWRsVjqUymBtQAJGmmZMJn7I/wDkQ1+Vv4ock6B8yzuJ8T67cn19LR4xV5N79PBzAyq329PYzU55UrNvauzPfIKwyyegfZdgBikXj5s1tz/Nr8jPhT0LZOAe3qfzrtU9EbiL7bIsULqNZ2ag4cNirtVHtFQXJ6ardIzIHKyNrQIpioyYgGGtdYQmjibjj0I6ZdQON1eR8Du6eEOgg2U4a5uhVU5Zmp1TQqXWvuA5FgCQ2U6UiMhLFzZWSyPnWXXrqd095La45z6jmbxZ7gXclSVZ9tqWCDU3M+3SUmkxL65i9UOzYI4KFs+qwWCHRJ5Z5N5L7j585Z1zlgDy0Ua/wBbxYVLm0H7xWiUss83V+FaI1n1V2lmT2ONsnsOoEe4eHABv1iTIfwLn075Hx/R4tuafHtZYL0MqyVZ8LKTUyPEWJek5gSJFlDFWEEQAcpaEmAH5AO5XG+RZ/KsLM5DkNJmfq1Rs1/YMC1f9iW6u4Yk4F9V4NrPETMBco4BhjEHKR/IV5Pcd0+Qvx63E4fYrfzaaGqi9uuaytOSqkVXk97OOyjursEXdUJiFHCYNLsWQOeSrajA7k6jxLfr62dQuKO3eoXD3Bh2bmaQ1R27qqzjplXTfYyUXXgMqDsiDGZORYamivz8YX49Mfxm6y0env4zdb6TuoGXicpWew7gGFb1qKdpWne47XrxoYGfYcNx8noGhwChbayrlN1iFSw7XtHvSnIvZ3qX5A1Vi41z9JRKF5KSQx8rs/oyrXVNyK12iMsSN83q8C+puNHZ+7kgaVCQvg/r51VEUv8ADTTTZWHK25Lkcz5T1AVYxs9NGhxJIxlWuR1bqci3agwh7qorqOh7JcQygljKyVQTY9sd1BMn0q5r0J6QfjVczOdclv8AIeR9aNBhcwyel+xgXubY2QSXFnUtdlnapFn1hoKaGiuyyLK7nI7ub9Mii5YCjvNfMPZ3AfM3tbzeb5VB6zos6OpfJajeKXeTeWdTrpJhVR6MVzJpuTTzbJKNDWqLbqVoGzFezr4p2q0DZ9FBDpCcby+Z4HGubccPiq9aFaKbCal6lfPK1a5GdTRPLbJUzsyA1qNyl4NB8rEmqXNiBGCD1V5d0J6kdWOgfVNHWGxw2bfF7udf2uPb3Hkcw4fpqQna4snllMVbiMoHM1uRYu9L6bc5dk11LdmM42MmH8ueX+jyexfPnSPNnmr035KpdZ1UsvR+nY9ntdphmkMsc1lqVAzbpyLTb6u+HjJTCDti3THeU4FmUIj1WzMv0jxfjGjPMePaXG+Ncm4lTq+lnI/5j316RxEwVmpQm4RWrlWwImkRaTmSTFtIEeuWfHvV7q5xYOh3UvivVTqt0m60butN2r0tng0Z2nupIwJeVtckjEBePia+a01X2tppoVYCvYqJdoTbCp8HeAfGtP1/l3tPsfofi/bUfUlpnQbRxarkrrJVXVobb12z2kSNfWzFUbK0zM7FMoXCwww7fcLzsBDjM8smuG2B02LYy+abPIcbbRqKPQtYtUl2ar7Tpr2rQwusaYbaltgkrCBGfYfdYx5TMfJTqR+VS+Ecv6C8F6Zc84BocPtI4zkc82FWsrYoZNIdTIx3FZ1UXCq44VMtd604zYPoT2ssmFgBfNpXxSUpt5h8KHn94Uncb2AunSOg3HHSRSabJXEAcS4HZ2+ifxg7qwMK69qZqSTrFDuFiMjXbbWTGcmHopx3Yq8Yp5TMu+vW0di1Kc06rwvNa801a4DVIIdJu9I+oYD+4yMxE9+/zSf86uoPFOV9aNHkOJyXE2ONY3D8KlO9naVO7jgFZdzSuTGhWayqQ1mX2i+fbPqYJrPsQTEck3q1t0f5ffknvxfl2jNLMDZj0VSo8swk4ASvnVKBXVjHRr0wkh31rSA8rQmylTsotCl4rYBBEKa61GDL66cQTl9F+l+cHLNBNRlVdi5fgTFjHad9jbf8Znqgu9qysPGqAqmQaaGWZJaJMw4NcwdqdauqWifEs91pdtlelnkQEtacygtVT+U0GyM/UrMODtGTRg1A5daAY+BA+7nyV53QeTvN/IPO9bZTOl3LqiOkJeTwfU3fvzCi3drf6g/zE/10L20NXDYdb9ov+tHMiB+2V9f7EnPvmXJbPMeUbXJbShQ3WuE8a4l5xWrLAK9OtLPEPaVeolCSb4B7SCWeAeXjHQrhvGa3DeL4nGarSerIpDXKwQ+E2bJmdi5Z9fkfqGxbc9wq8z9Qshfmfj5SQ9CD6SE9gf8APQJnEu68MAgA2zTBqINsny7kGxozWUKcgnSHUWMnfI0BMq3c/ARsLTUPf9DLYVuKuBbx0lZmUrSaW3iXXCfcUm0arXjWNkBCxOfATJMt9bRfC5+XeoVMlSq2UL7GRCYpgmTHhEQMsEJZEd/KYjvMQcB5DIeUfPCJt20gvYieMcCMoivqyRiBEEkK8Qg5xLZXynAtlOyTlVFlfAjywxASctmg3bIymQs2uqS55SbJM4BIsOmgwNdQhSs22SvW68LvN85rj6Qq+7xNyCCbFVj1lEZF/GwPjHc/GGmJCTYkyEVwlTPJI+Psnylnh3gDifWwQKPg3uw9JS/+7/OqxiYUTTGV8RFfnH3eEt1WVsIkO9nhiMBHVh6ZJMYnrTINz7dkTQjMVZFKYk7m5f6/TQBjVfEpE6ZBNplmv6RWE3hFqgrqiTY9qGjLtH1wfjRA1vHHj+vtMolgf3mGxMLFbPOSn0TIlJlPYQExnwR5du7ijFHaO8MDmJKeuCygLL84RkyOooK9prV4gK+TCyXwmNvxbrwnMjIbdlIMEzZJB2EY2ixrsNvH8Xe5fYc0qtEDWnYsUy+wI1O9KFUzBqFssf8A0qXaJy5dK1WH1hcC4TYlcx6VZawH2ukZKoDY9cyz/bybBCwoD/I5XAH4QRLBkj5SYeXxSrj8qHHqT+/NqWqtvrL0prO3Hxxzy+sA6cWJNC0LFAkuVvqRrmg0IHXSL82wx9jbP66GORlgsYFCy7knvh3Sjm2plK1uYBS4Jl+1vu1+VvRlg5EMIlMoUIYdm2bkSsqylRKjk1qK4TJJkhvk3VriWZeZk8a+7znc8YheNxNB6rFtiPAhv3lj9OiC2icWCaz3KEDZ9TxiA+JtdvD/AMiHycsl5Hu/oaXyZ5nGOCcrPLXEmYtpu7jaLXMo8l9t++Ta7u5HhnxiNgfNaF4B0E2wHPkBO2xkhUoc+6a9KlNHp7mP5jyk1sQ3lu8o6lBET+ijOpR67UIIhnupY02sWQ+zSsjEBAzv8A6mdV2qLqFpo4bxQGLeriOC4Lmg+Y/Yzo3Z86svESjs1hW1LYMyvMrHMnO4vzH5G89+PKHFz3z9zdLR1Mmguz1tFpsfbbieLpJro2uNqOzM5sB2N5ydx9DCsr1WhMoSMFWtxEFEE+V8z5JzXRnS5HqPvuiTiumZhdOks5jumjTX4orL7CEFIB7HSMMsMa2SZJu4nwvjXCc6MzjeWjPTMBNh0RLLt5gRMQ69cZ5Pss7kcjBn6kwZLrrSrxXDJfqr/LT8HlNuqj8wleis9edHhQbFGAqXS1kYINo5c13cgkYMmaaCDSwVywotppdNI9XKFysztg1WdBA5dSuVgBtipZQthQC2OQ1QGcoRagQMwESKa1qtYgYmZlFhDe3rcsibJu07JmqvbrPYsZNi0vU0wCHvqyRgBkQjFmrZryUxEQ+u9Xf2JYIybJosTAks3DEFStDhkILYMixwQRYIY9pZpySypIoIIYotN5JJJZNdNI9Nt9tsa65zhJSmvYCkKY5rCgQUoCYwyKYgRAAiSIpmYiIiJmZmIiO8/FWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+A1jV9Tc7TQV+21KkAya7QZn/yZ1zsGNNs7Y++uYG2Guo1huuudf4RmVWs4UUmud5sGR7fw6v6rclEQVmncvsiYKB+4ujW7x2/matdazYavv38iVbqGUT2HwmPKWFpWu+SGtcp0Fz3Hy+my9Z7T3/0WxlmrXUzt27A2pbWMx3L2RPjC52XxhyToBwkXonoHV/RG7IjOi+o9b6Psq52ynCgmMkHzxTlwXM+SWfIwcc80mLBQ7AXqJDJOTPLnSWfNnq852c1ZzxnOx+NQoe7LuNlw7TULCEIKN7WPV2akEciI/W0aweZQIjHeB+Ve1wXG0mBHJtLZ5NLS7LpbOpKcxpAJHITg5C8nFt+AQRT9nOsnADJGU9iL4zVAoHNeYJcUzl1LpPP68t30kzVqJXkdXUBSkRa/hNumQiAiQTERRa5xNuPrJPppjbO2+Nf3xVdHR1NZ/wB7WvX9Ky2Jj7ehZsW3MgZnvEPsGwyESmf1BTAzM/qO/wAtebm5eTX+jkUKGbVVMT9TPrV6iFyUR2mUVgWAkURH7kYkoj/ue3w1mmhHhlIIljgHgjkmnnmk1ihhhi1zvLLLLvnXSOOPTXbeSTfbGumuM7bZxjGc/pgIkRCIjJEUwIiMTJEUz2gRiO8zMzMRERHeZ/UfH5EIjJFMCIxJERTECIxHeZmZ/UREfuZn9RH7n5HiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/FWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+A1jV9Tc7TQV+21KkAya7QZn/yZ1zsGNNs7Y++uYG2Guo1huuudf4RmVWs4UUmud5sGR7fw6v6rclEQVmncvsiYKB+4ujW7x2/matdazYavv38iVbqGUT2HwmPKWFpWu+SGtcp0Fz3Hy+my9Z7T3/0WxlmrXUzt27A2pbWMx3L2RPjC52XxhyToBwkXonoHV/RG7IjOi+o9b6Psq52ynCgmMkHzxTlwXM+SWfIwcc80mLBQ7AXqJDJOTPLnSWfNnq852c1ZzxnOx+NQoe7LuNlw7TULCEIKN7WPV2akEciI/W0aweZQIjHeB+Ve1wXG0mBHJtLZ5NLS7LpbOpKcxpAJHITg5C8nFt+AQRT9nOsnADJGU9iL4zVAoHNeYJcUzl1LpPP68t30kzVqJXkdXUBSkRa/hNumQiAiQTERRa5xNuPrJPppjbO2+Nf3xVdHR1NZ/wB7WvX9Ky2Jj7ehZsW3MgZnvEPsGwyESmf1BTAzM/qO/wAtebm5eTX+jkUKGbVVMT9TPrV6iFyUR2mUVgWAkURH7kYkoj/ue3w1mmhHhlIIljgHgjkmnnmk1ihhhi1zvLLLLvnXSOOPTXbeSTfbGumuM7bZxjGc/pgIkRCIjJEUwIiMTJEUz2gRiO8zMzMRERHeZ/UfH5EIjJFMCIxJERTECIxHeZmZ/UREfuZn9RH7n5HiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUWVjIvmY2mGDdjqERLNiLBBdj4j1SDQ1OV1f4vPqI4xZYiH6E0NGNIQffVBRWu5wTVkgoyyQBziCWQHs/rBEP+rn4rXOEcU6M7aeUavIL/AFcyq2iWXk5GnkRxlj87jlw0npZ+7eHVFTuQBWF7KWdDhqk+ERDSUlP/ADcN6Z96cUufbvN/rn0Z5UsyDojikJqd0e4UP0zztuQnrNWsOpUm9k47TbSnEmns0QE2+7S06aYDIk+nPJttBqUuknWzjXMsx1/lPSTjLKCNFmac5l3Ro6jIXXq2GWRsVjqUymBtQAJGmmZMJn7I/wDkQ1+Vv4ock6B8yzuJ8T67cn19LR4xV5N79PBzAyq329PYzU55UrNvauzPfIKwyyegfZdgBikXj5s1tz/Nr8jPhT0LZOAe3qfzrtU9EbiL7bIsULqNZ2ag4cNirtVHtFQXJ6ardIzIHKyNrQIpioyYgGGtdYQmjibjj0I6ZdQON1eR8Du6eEOgg2U4a5uhVU5Zmp1TQqXWvuA5FgCQ2U6UiMhLFzZWSyPnWXXrqd095La45z6jmbxZ7gXclSVZ9tqWCDU3M+3SUmkxL65i9UOzYI4KFs+qwWCHRJ5Z5N5L7j585Z1zlgDy0Ua/wBbxYVLm0H7xWiUss83V+FaI1n1V2lmT2ONsnsOoEe4eHABv1iTIfwLn075Hx/R4tuafHtZYL0MqyVZ8LKTUyPEWJek5gSJFlDFWEEQAcpaEmAH5AO5XG+RZ/KsLM5DkNJmfq1Rs1/YMC1f9iW6u4Yk4F9V4NrPETMBco4BhjEHKR/IV5Pcd0+Qvx63E4fYrfzaaGqi9uuaytOSqkVXk97OOyjursEXdUJiFHCYNLsWQOeSrajA7k6jxLfr62dQuKO3eoXD3Bh2bmaQ1R27qqzjplXTfYyUXXgMqDsiDGZORYamivz8YX49Mfxm6y0env4zdb6TuoGXicpWew7gGFb1qKdpWne47XrxoYGfYcNx8noGhwChbayrlN1iFSw7XtHvSnIvZ3qX5A1Vi41z9JRKF5KSQx8rs/oyrXVNyK12iMsSN83q8C+puNHZ+7kgaVCQvg/r51VEUv8ADTTTZWHK25Lkcz5T1AVYxs9NGhxJIxlWuR1bqci3agwh7qorqOh7JcQygljKyVQTY9sd1BMn0q5r0J6QfjVczOdclv8AIeR9aNBhcwyel+xgXubY2QSXFnUtdlnapFn1hoKaGiuyyLK7nI7ub9Mii5YCjvNfMPZ3AfM3tbzeb5VB6zos6OpfJajeKXeTeWdTrpJhVR6MVzJpuTTzbJKNDWqLbqVoGzFezr4p2q0DZ9FBDpCcby+Z4HGubccPiq9aFaKbCal6lfPK1a5GdTRPLbJUzsyA1qNyl4NB8rEmqXNiBGCD1V5d0J6kdWOgfVNHWGxw2bfF7udf2uPb3Hkcw4fpqQna4snllMVbiMoHM1uRYu9L6bc5dk11LdmM42MmH8ueX+jyexfPnSPNnmr035KpdZ1UsvR+nY9ntdphmkMsc1lqVAzbpyLTb6u+HjJTCDti3THeU4FmUIj1WzMv0jxfjGjPMePaXG+Ncm4lTq+lnI/5j316RxEwVmpQm4RWrlWwImkRaTmSTFtIEeuWfHvV7q5xYOh3UvivVTqt0m60butN2r0tng0Z2nupIwJeVtckjEBePia+a01X2tppoVYCvYqJdoTbCp8HeAfGtP1/l3tPsfofi/bUfUlpnQbRxarkrrJVXVobb12z2kSNfWzFUbK0zM7FMoXCwww7fcLzsBDjM8smuG2B02LYy+abPIcbbRqKPQtYtUl2ar7Tpr2rQwusaYbaltgkrCBGfYfdYx5TMfJTqR+VS+Ecv6C8F6Zc84BocPtI4zkc82FWsrYoZNIdTIx3FZ1UXCq44VMtd604zYPoT2ssmFgBfNpXxSUpt5h8KHn94Uncb2AunSOg3HHSRSabJXEAcS4HZ2+ifxg7qwMK69qZqSTrFDuFiMjXbbWTGcmHopx3Yq8Yp5TMu+vW0di1Kc06rwvNa801a4DVIIdJu9I+oYD+4yMxE9+/zSf86uoPFOV9aNHkOJyXE2ONY3D8KlO9naVO7jgFZdzSuTGhWayqQ1mX2i+fbPqYJrPsQTEck3q1t0f5ffknvxfl2jNLMDZj0VSo8swk4ASvnVKBXVjHRr0wkh31rSA8rQmylTsotCl4rYBBEKa61GDL66cQTl9F+l+cHLNBNRlVdi5fgTFjHad9jbf8Znqgu9qysPGqAqmQaaGWZJaJMw4NcwdqdauqWifEs91pdtlelnkQEtacygtVT+U0GyM/UrMODtGTRg1A5daAY+BA+7nyV53QeTvN/IPO9bZTOl3LqiOkJeTwfU3fvzCi3drf6g/zE/10L20NXDYdb9ov+tHMiB+2V9f7EnPvmXJbPMeUbXJbShQ3WuE8a4l5xWrLAK9OtLPEPaVeolCSb4B7SCWeAeXjHQrhvGa3DeL4nGarSerIpDXKwQ+E2bJmdi5Z9fkfqGxbc9wq8z9Qshfmfj5SQ9CD6SE9gf8APQJnEu68MAgA2zTBqINsny7kGxozWUKcgnSHUWMnfI0BMq3c/ARsLTUPf9DLYVuKuBbx0lZmUrSaW3iXXCfcUm0arXjWNkBCxOfATJMt9bRfC5+XeoVMlSq2UL7GRCYpgmTHhEQMsEJZEd/KYjvMQcB5DIeUfPCJt20gvYieMcCMoivqyRiBEEkK8Qg5xLZXynAtlOyTlVFlfAjywxASctmg3bIymQs2uqS55SbJM4BIsOmgwNdQhSs22SvW68LvN85rj6Qq+7xNyCCbFVj1lEZF/GwPjHc/GGmJCTYkyEVwlTPJI+Psnylnh3gDifWwQKPg3uw9JS/+7/OqxiYUTTGV8RFfnH3eEt1WVsIkO9nhiMBHVh6ZJMYnrTINz7dkTQjMVZFKYk7m5f6/TQBjVfEpE6ZBNplmv6RWE3hFqgrqiTY9qGjLtH1wfjRA1vHHj+vtMolgf3mGxMLFbPOSn0TIlJlPYQExnwR5du7ijFHaO8MDmJKeuCygLL84RkyOooK9prV4gK+TCyXwmNvxbrwnMjIbdlIMEzZJB2EY2ixrsNvH8Xe5fYc0qtEDWnYsUy+wI1O9KFUzBqFssf8A0qXaJy5dK1WH1hcC4TYlcx6VZawH2ukZKoDY9cyz/bybBCwoD/I5XAH4QRLBkj5SYeXxSrj8qHHqT+/NqWqtvrL0prO3Hxxzy+sA6cWJNC0LFAkuVvqRrmg0IHXSL82wx9jbP66GORlgsYFCy7knvh3Sjm2plK1uYBS4Jl+1vu1+VvRlg5EMIlMoUIYdm2bkSsqylRKjk1qK4TJJkhvk3VriWZeZk8a+7znc8YheNxNB6rFtiPAhv3lj9OiC2icWCaz3KEDZ9TxiA+JtdvD/AMiHycsl5Hu/oaXyZ5nGOCcrPLXEmYtpu7jaLXMo8l9t++Ta7u5HhnxiNgfNaF4B0E2wHPkBO2xkhUoc+6a9KlNHp7mP5jyk1sQ3lu8o6lBET+ijOpR67UIIhnupY02sWQ+zSsjEBAzv8A6mdV2qLqFpo4bxQGLeriOC4Lmg+Y/Yzo3Z86svESjs1hW1LYMyvMrHMnO4vzH5G89+PKHFz3z9zdLR1Mmguz1tFpsfbbieLpJro2uNqOzM5sB2N5ydx9DCsr1WhMoSMFWtxEFEE+V8z5JzXRnS5HqPvuiTiumZhdOks5jumjTX4orL7CEFIB7HSMMsMa2SZJu4nwvjXCc6MzjeWjPTMBNh0RLLt5gRMQ69cZ5Pss7kcjBn6kwZLrrSrxXDJfqr/LT8HlNuqj8wleis9edHhQbFGAqXS1kYINo5c13cgkYMmaaCDSwVywotppdNI9XKFysztg1WdBA5dSuVgBtipZQthQC2OQ1QGcoRagQMwESKa1qtYgYmZlFhDe3rcsibJu07JmqvbrPYsZNi0vU0wCHvqyRgBkQjFmrZryUxEQ+u9Xf2JYIybJosTAks3DEFStDhkILYMixwQRYIY9pZpySypIoIIYotN5JJJZNdNI9Nt9tsa65zhJSmvYCkKY5rCgQUoCYwyKYgRAAiSIpmYiIiJmZmIiO8/FWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+A1jV9Tc7TQV+21KkAya7QZn/yZ1zsGNNs7Y++uYG2Guo1huuudf4RmVWs4UUmud5sGR7fw6v6rclEQVmncvsiYKB+4ujW7x2/matdazYavv38iVbqGUT2HwmPKWFpWu+SGtcp0Fz3Hy+my9Z7T3/0WxlmrXUzt27A2pbWMx3L2RPjC52XxhyToBwkXonoHV/RG7IjOi+o9b6Psq52ynCgmMkHzxTlwXM+SWfIwcc80mLBQ7AXqJDJOTPLnSWfNnq852c1ZzxnOx+NQoe7LuNlw7TULCEIKN7WPV2akEciI/W0aweZQIjHeB+Ve1wXG0mBHJtLZ5NLS7LpbOpKcxpAJHITg5C8nFt+AQRT9nOsnADJGU9iL4zVAoHNeYJcUzl1LpPP68t30kzVqJXkdXUBSkRa/hNumQiAiQTERRa5xNuPrJPppjbO2+Nf3xVdHR1NZ/wB7WvX9Ky2Jj7ehZsW3MgZnvEPsGwyESmf1BTAzM/qO/wAtebm5eTX+jkUKGbVVMT9TPrV6iFyUR2mUVgWAkURH7kYkoj/ue3w1mmhHhlIIljgHgjkmnnmk1ihhhi1zvLLLLvnXSOOPTXbeSTfbGumuM7bZxjGc/pgIkRCIjJEUwIiMTJEUz2gRiO8zMzMRERHeZ/UfH5EIjJFMCIxJERTECIxHeZmZ/UREfuZn9RH7n5HiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfPHiPEh+dcAOFZudxpjNMCMBCc7CDkyBEFa4hm3/ACGgMilEmnx+8URMUkG+2sum2mFDrvX39iWr7EIT5rMexkEMEJ8hj+xAUGI/9yEwURMTE/EgsIZ2hb0smQlkeDALuAnIEceJT3ATiQkv+oOJGZ7xMfPWGYIxEFYLyhjgDhoDAjQ54iRDBCYtZxihSYNt4SBiId9JoJ4d94pYt9ZI9ttNsZzgYGozWwCWxZEDFmMgYGEyJgYlEEJCUSJCUSIzmYiIiJmZmYiO8/GWuUhZue1aVLGSY1pitYCMSREZnMCIjETMzMxEREzM9o+YfA==" alt="YASH logo" style='height:64px; width:auto; object-fit:contain;' />
            <div style="display: flex; flex-direction: column; gap: 8px;">
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_filters(session):
    """Render filter controls in a single row, styled to match the provided mock (44px pills)."""

    st.markdown(f"""
<style>
    :root{{
        --primary-blue: {PRIMARY_COLOR};
        --border-gray: {BORDER_COLOR};
        --text-dark: {TEXT_DARK};
        --muted-rail: {SURFACE_LIGHTER};
        --control-shadow: rgba(16,24,40,0.06);
        --active-shadow: rgba(37,99,235,0.18);
        --pill-h: 44px;
        --pill-r: 12px;
        --font-14: {FONT_SIZE_NORMAL}px;
        --font-13: {FONT_SIZE_NORMAL}px;
    }}

    /* Row spacing */
    [data-testid="stHorizontalBlock"] {{ gap: 12px !important; align-items: center; flex-wrap: nowrap !important; }}
    [data-testid="stColumn"] {{ padding: 0 6px !important; }}

    /* Hide labels */
    .stDateInput > div > label, .stSelectbox > div > label {{ display:none !important; }}

    /* -------------------- DATE PILL -------------------- */
    .stDateInput > div > div > div {{
        background: #ffffff !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--pill-r) !important;
        padding: 0 12px !important;
        height: var(--pill-h) !important;
        display:flex !important;
        align-items:center !important;
        color: var(--text-dark) !important;
    }}
    .stDateInput input {{
        font-size: var(--font-14) !important;
        padding: 0 !important;
        height: calc(var(--pill-h) - 2px) !important;
        border: none !important;
        box-shadow: none !important;
    }}
    .stDateInput > div > div > div::before {{
        content: "" !important;
        display:inline-block !important; width:16px !important; height:16px !important; margin-right:8px !important;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%236b7280' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><rect x='3' y='4' width='18' height='18' rx='2'/><line x1='16' y1='2' x2='16' y2='6'/><line x1='8' y1='2' x2='8' y2='6'/><line x1='3' y1='10' x2='21' y2='10'/></svg>") !important;
        background-repeat: no-repeat !important; background-position: center !important;
    }}

    /* -------------------- SELECT PILLS (Dealer/Region/Product) -------------------- */
    /* base pill styling */
    .stSelectbox > div > div > div {{
        background: #ffffff !important;
        border: 1px solid var(--border-gray) !important;
        border-radius: var(--pill-r) !important;
        padding: 0 18px !important;
        height: var(--pill-h) !important;
        display:flex !important;
        align-items:center !important;
        font-size: var(--font-13) !important;
        color: var(--text-dark) !important;
        position: relative !important;
        font-weight: 400 !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }}
    .stSelectbox > div > div > div > div {{
        width: 100% !important;
        text-align: left !important;
        display: flex !important;
        align-items: center !important;
    }}
    /* keep room for our caret; allow full visible labels */
    .stSelectbox > div > div > div > div {{
        overflow: visible !important; text-overflow: clip !important; white-space: nowrap !important; padding-right: 22px !important;
    }}

    /* Hide the small inline dropdown button to avoid duplicate caret; keep our ::after caret */
    .stSelectbox>div>div>button, .stSelectbox [role="button"] {{ display: none !important; }}

    /* HIDE BaseWeb / internal carets & icons (fixes the double arrow) */
    /* These target the built-in icon containers across Streamlit versions */
    .stSelectbox [data-baseweb="select"] svg,
    .stSelectbox [data-baseweb="select"] img,
    .stSelectbox [data-baseweb="popover"] svg,
    .stSelectbox [role="button"] svg,
    .stSelectbox [class*="select-container"] svg {{
        display:none !important;
    }}

    /* Hide ::before/::after on inner child containers so only our parent ::after shows */
    .stSelectbox > div > div > div > div::after,
    .stSelectbox > div > div > div > div::before,
    .stSelectbox > div > div > div > div > div::after,
    .stSelectbox > div > div > div > div > div::before {{
        display:none !important; content:none !important; background-image:none !important; width:0 !important; height:0 !important;
    }}

    /* OUR single caret */
    .stSelectbox > div > div > div::after {{
        content: "" !important;
        width: 14px; height: 14px; display:inline-block;
        position:absolute; right: 12px; top: 50%; transform: translateY(-50%);
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='14' height='14' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 12 15 18 9'/></svg>");
        background-repeat: no-repeat; background-position: center;
        pointer-events:none;
    }}

    /* Hide known dropdown indicator classes (react-select/baseweb etc.) and any inner buttons that render another caret */
    .stSelectbox [class*="dropdown-indicator"],
    .stSelectbox [class*="indicator"],
    .stSelectbox [class*="indicators"],
    .stSelectbox .react-select__indicator,
    .stSelectbox .react-select__indicators,
    .stSelectbox .Select__indicator,
    .stSelectbox .Select__indicators,
    .stSelectbox [class*="control"] svg,
    .stSelectbox [class*="control"]::after,
    .stSelectbox button,
    .stSelectbox > div > div > div > button,
    .stSelectbox > div > div > div::before,
    /* Hide tiny adjacent sibling containers that Streamlit may render as a separate caret button */
    .stSelectbox > div > div > div + div,
    .stSelectbox > div > div + div {{
        display: none !important; content: none !important; background-image: none !important; width: 0 !important; height: 0 !important; border:0 !important; padding:0 !important; margin:0 !important;
    }}

    /* Final fallback: hide any tiny inline svgs inside the select container (but keep our ::after) */
    .stSelectbox > div > div > div svg {{ display:none !important; }}

    /* Additional Chrome-friendly fallbacks: hide tiny rounded indicator buttons or containers (common in different Streamlit/React versions) */
    .stSelectbox > div > div > div > div[role="button"],
    .stSelectbox > div > div > div > div[class*="indicator"],
    .stSelectbox > div > div > div > div[class*="dropdown-indicator"],
    .stSelectbox > div > div > div > span[class*="indicator"],
    .stSelectbox > div > div > div > span[class*="dropdown-indicator"],
    .stSelectbox > div > div > div > button[aria-hidden="true"],
    .stSelectbox > div > div > div > button[class*="indicator"] {{
        display:none !important; width:0 !important; height:0 !important; padding:0 !important; margin:0 !important; border:0 !important; background:transparent !important;
    }}

    /* Hide any small circular marker elements inside the radio labels (red dot) — target inline styles and common classnames */
    /* -------------------- SEGMENTED CONTROLS (Revenue/Units & Last30/QTD/YTD) -------------------- */
    /* Keep radio groups in a single row */
    div[role="radiogroup"] {{
        display: inline-flex !important;
        gap: 8px !important;
        align-items: center !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
    }}

    /* Hide the native radio bubble if present */
    div[role="radiogroup"] input[type="radio"] {{
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }}

    /* Base pill style for each option */
    div[role="radiogroup"] label {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 18px !important;
        height: var(--pill-h) !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        font-size: var(--font-14) !important;
        border: 1px solid var(--border-gray) !important;
        background: #ffffff !important;
        color: var(--text-dark) !important;
        cursor: pointer !important;
        transition: all 120ms ease !important;
        box-shadow: 0 4px 10px var(--control-shadow) !important;
        margin: 0 !important;
    }}

    /* Active (selected) pill state – use same blue as Dashboard button (#2563eb) */
    /* Streamlit nests the real "checked" state inside several wrappers, so we style both
       the label AND its inner content when something is selected. */
    div[role="radiogroup"] label[aria-checked="true"],
    div[role="radiogroup"] label[data-selected="true"],
    div[role="radiogroup"] [role="radio"][aria-checked="true"],
    div[role="radiogroup"] label > div[aria-checked="true"],
    div[role="radiogroup"] label > span[aria-checked="true"],
    div[role="radiogroup"] [aria-checked="true"] {{
        background-color: #2563eb !important;
        background: #2563eb !important;
        border-color: #2563eb !important;
        color: #ffffff !important;
    }}

    /* Force all text inside the selected pill to white (Streamlit may set nested colors) */
    div[role="radiogroup"] label[aria-checked="true"],
    div[role="radiogroup"] label[data-selected="true"],
    div[role="radiogroup"] [role="radio"][aria-checked="true"],
    div[role="radiogroup"] label > div[aria-checked="true"],
    div[role="radiogroup"] label > span[aria-checked="true"] {{
        color: #ffffff !important;
    }}

    /* Catch any nested elements (spans/divs) inside the active pill and force them to white too */
    div[role="radiogroup"] label[aria-checked="true"] *,
    div[role="radiogroup"] label[data-selected="true"] *,
    div[role="radiogroup"] [role="radio"][aria-checked="true"] *,
    div[role="radiogroup"] label > div[aria-checked="true"] *,
    div[role="radiogroup"] label > span[aria-checked="true"] * {{
        color: #ffffff !important;
    }}

    /* Remove any tiny leading icon / dot containers if they exist */
    div[role="radiogroup"] label > div:first-child {{
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }}

</style>
    """, unsafe_allow_html=True)

    # ---------- layout (wider slots for radios to avoid wrap) ----------
    # Adjusted columns: [date, dealer, product, (spacer), time_period]
    cols = st.columns([3.5, 1.2, 1.2, 0.3, 2.1])


    if 'time_period' not in st.session_state:
        st.session_state['time_period'] = get_config_value("filters.default_time_period", "Last 30 Days")

    def compute_date_range_for_period(period):
        today = datetime.now().date()
        if period == 'Last 30 Days':
            return (today - timedelta(days=29), today)
        elif period == 'QTD':
            month = today.month
            q_start_month = ((month - 1)//3)*3 + 1
            q_start = datetime(today.year, q_start_month, 1).date()
            return (q_start, today)
        elif period == 'YTD':
            return (datetime(today.year, 1, 1).date(), today)
        return (today - timedelta(days=29), today)

    with cols[4]:
        time_period = st.radio(
            label="Time Period",
            options=['Last 30 Days', 'QTD', 'YTD'],
            index=['Last 30 Days', 'QTD', 'YTD'].index(st.session_state.get('time_period', 'Last 30 Days')),
            key='time_period_radio',
            horizontal=True,
            label_visibility='collapsed'
        )
        # Check if time period changed
        if st.session_state.get('time_period') != time_period:
            st.session_state['time_period'] = time_period
            st.session_state['time_period_changed'] = True
        else:
            st.session_state['time_period'] = time_period

    with cols[0]:
        # Always recompute date range based on current time period
        current_time_period = st.session_state.get('time_period', 'Last 30 Days')
        initial_date_range = compute_date_range_for_period(current_time_period)

        # Create dynamic key based on time period to force recompute
        date_picker_key = f'date_range_picker_{current_time_period}'
        date_range = st.date_input(
            "Date Range",
            value=initial_date_range,
            key=date_picker_key,
            label_visibility="collapsed"
        )

    # Dealer and Product filters side by side
    with cols[1]:
        # Migration note: fetch_dealers(session) → fetch_dealers()
        dealer_list = fetch_dealers() if session else []
        if not dealer_list:
            pass  # silently handle
            dealer = st.selectbox("Dealer", ["All Dealers"], key='dealer_sel', label_visibility='collapsed')
        else:
            options = ['All Dealers'] + [d for d in dealer_list if d != 'All Dealers']
            dealer = st.selectbox("Dealer", options, index=0, key='dealer_sel', label_visibility='collapsed')
    with cols[2]:
        # Migration note: fetch_products(session) → fetch_products()
        products = fetch_products() if session else []
        if not products:
            pass  # silently handle
            product = st.selectbox("Product", ["Product"], key='product_sel', label_visibility='collapsed')
        else:
            product = st.selectbox("Product", ["Product"] + products, index=0, key='product_sel', label_visibility='collapsed')
    # Spacer
    with cols[3]:
        st.markdown("", unsafe_allow_html=True)

    # Normalize returned date tuple safely
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        f_date, t_date = date_range
    else:
        f_date, t_date = datetime(2026, 1, 1), datetime(2026, 2, 1)

    return {
        "from_date": f_date,
        "to_date": t_date,
        "dealer": dealer,
        "product": product,
        "metric": "Revenue",
        "time_period": st.session_state.get('time_period', 'Last 30 Days'),
    }


# safe_number, safe_int, abbr_currency → SKIPPED: in utils.py


def _pick_chart_columns(df):
    """Pick best categorical and numeric columns for charting."""
    if df is None or df.empty:
        return None, None

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()

    if cat_cols and num_cols:
        return cat_cols[0], num_cols[0]
    return None, None


# _cortex_complete_prescriptive → SKIPPED: in ai_service.py as _bedrock_complete_prescriptive()


def _parse_descriptive_prescriptive(text: str):
    """Split analyst response into (descriptive, prescriptive) sections if clearly marked; else (None, None)."""
    if not text or not text.strip():
        return None, None
    text = text.strip()
    # Look for Prescriptive section (various markdown/plain forms)
    pres_markers = ("**Prescriptive**", "**Prescriptive**:", "Prescriptive:", "Prescriptive**", "\nPrescriptive:")
    idx = -1
    for m in pres_markers:
        i = text.find(m)
        if i >= 0:
            idx = i
            break
    if idx >= 0:
        descriptive = text[:idx].strip()
        prescriptive = text[idx:].strip()
        # Strip the marker line from prescriptive
        for m in pres_markers:
            if prescriptive.startswith(m):
                prescriptive = prescriptive[len(m):].strip().lstrip(":\n ")
                break
        # Strip Descriptive marker from first part for display
        for d in ("**Descriptive**", "**Descriptive**:", "Descriptive:", "Descriptive**"):
            if descriptive.startswith(d):
                descriptive = descriptive[len(d):].strip().lstrip(":\n ")
                break
        return descriptive or None, prescriptive or None
    return None, None


def _generate_descriptive_predictive_for_quick(question: str, vendors_df, metrics: dict, session=None):
    """
    Generate question- and data-specific descriptive and predictive text for Genie quick layout.
    Returns (descriptive_text, predictive_html). Used so custom/suggested/KPI/quick analyses
    show narrative analysis in addition to visualization and table.
    """
    if vendors_df is None or vendors_df.empty:
        return None, None

    descriptive = None
    predictive = None

    # Debug: Check if session is available
    if session is None:
        pass  # silently handle

    # Build data summary for LLM (limit size for token budget)
    head = vendors_df.head(25)
    data_str = head.to_string(index=False, max_colwidth=35)
    if len(data_str) > 12000:
        data_str = data_str[:12000] + "\n... (truncated)"

    cols_list = list(vendors_df.columns)
    metrics_str = ""
    if metrics:
        metrics_str = "Aggregate metrics: " + ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:8]) + "."

    # Enhanced prompt for more detailed output
    prompt = (
        "You are a dealer business analyst. The user asked a specific business question and the system returned data. "
        "Based on this data, provide detailed analysis in two sections:\n\n"
        "**Descriptive** (4–6 sentences): What the data actually shows. Be specific:\n"
        "- Mention the number of dealers/records in the result\n"
        "- Cite specific dealer names, values, and metrics from the data\n"
        "- Highlight top performers, bottom performers, or anomalies\n"
        "- Reference column names and exact figures where relevant\n"
        "- Directly address the user's question with concrete findings\n\n"
        "**Prescriptive** (5–7 bullet points): Specific actions based on the data insights:\n"
        "- Use • for bullets\n"
        "- Be concrete and actionable (not generic)\n"
        "- Include responsible parties (e.g., 'dealer X', 'procurement team')\n"
        "- Reference specific metrics or thresholds from the data\n"
        "- Provide expected business outcomes\n\n"
        f"User question: {question}\n\n"
        f"Columns available: {cols_list}\n"
        f"{metrics_str}\n\n"
        f"Data sample:\n{data_str}\n\n"
        "Format exactly as shown above, with clear section headers. Be detailed and business-focused, not generic."
    )

    try:
        # Migration note: session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?,?)") → bedrock_complete()
        text = bedrock_complete(prompt, model_id=get_config()["bedrock"]["primary_model"])
        if text and isinstance(text, str) and len(text.strip()) > 50:
            # Parse Descriptive / Predictive sections
            desc_part, pres_part = _parse_descriptive_prescriptive(text)
            if not desc_part and "Descriptive:" in text:
                idx = text.find("Descriptive:")
                end = text.find("Predictive:") if "Predictive:" in text else len(text)
                desc_part = text[idx + len("Descriptive:"):end].strip().strip(":\n ")
            if not pres_part and "Predictive:" in text:
                idx = text.find("Predictive:")
                pres_part = text[idx + len("Predictive:"):].strip().strip(":\n ")
            if desc_part or pres_part:
                descriptive = desc_part or descriptive
                predictive = pres_part or predictive
                if descriptive and predictive:
                    # Format predictive as HTML list
                    pres_lines = predictive.split("\n")
                    pres_html = "<ul style='margin:0;padding-left:24px;'>"
                    for line in pres_lines:
                        line = line.strip()
                        if line.startswith("•"):
                            line = line.lstrip("• ").strip()
                        if line:
                            pres_html += f"<li style='margin-bottom:8px;line-height:1.6;'>{html.escape(line)}</li>"
                    pres_html += "</ul>"
                    return descriptive, pres_html
    except Exception:
        pass  # silently handle

    # Enhanced rule-based fallback: build detailed descriptive + prescriptive from data
    upper = {str(c).upper(): c for c in vendors_df.columns}
    x_col, y_col = _pick_chart_columns(vendors_df)
    n_dealers = len(vendors_df)

    desc_parts = []

    # Data overview
    desc_parts.append(f"The analysis covers {n_dealers} dealer{'s' if n_dealers != 1 else ''} with the following key dimensions: {', '.join(str(c) for c in cols_list[:5])}.")

    # Top performers analysis
    if x_col and y_col:
        try:
            numeric = pd.to_numeric(vendors_df[y_col], errors="coerce")
            if numeric.notna().any():
                top3 = vendors_df.nlargest(3, y_col)
                bottom3 = vendors_df.nsmallest(3, y_col)

                # Detailed top performers
                top_names = []
                for i in range(len(top3)):
                    name = str(top3.iloc[i].get(x_col, ""))[:40]
                    value = top3.iloc[i].get(y_col)
                    if name and value:
                        top_names.append(f"{name} ({value})")
                if top_names:
                    desc_parts.append(f"Top performers: {', '.join(top_names)}.")

                # Add variance/range
                max_val = numeric.max()
                min_val = numeric.min()
                mean_val = numeric.mean()
                desc_parts.append(f"Range spans from {min_val:.1f} to {max_val:.1f} across dealers, with an average of {mean_val:.1f}.")

                # Identify outliers
                above_mean = len(vendors_df[numeric > mean_val])
                below_mean = n_dealers - above_mean
                desc_parts.append(f"Performance split: {above_mean} dealers exceed average, {below_mean} fall below.")
        except Exception:
            desc_parts.append(f"Data contains {n_dealers} records with key metrics for analysis.")

    descriptive = " ".join(desc_parts)

    # Generate prescriptive bulletpoints
    bullets = []
    q_lower = (question or "").lower()

    # Context-aware recommendations
    if "stock" in q_lower or "inventory" in q_lower or "backorder" in q_lower:
        bullets.append("• Monitor dealers with low stock availability (<70%) to prevent revenue loss from stock-outs.")
        bullets.append("• Implement safety stock reviews for high-backorder dealers (>10% incidence).")
        bullets.append("• Prioritize inventory replenishment for dealers in the bottom quartile.")
        if "BACKORDER_INCIDENCE_PCT" in [c.upper() for c in vendors_df.columns]:
            bullets.append("• Track backorder trends weekly and escalate dealers exceeding 15% threshold.")

    if "cash" in q_lower or "cycle" in q_lower or "ccc" in q_lower:
        bullets.append("• Target dealers with CCC >60 days for immediate working capital optimization.")
        bullets.append("• Focus on Days Sales Outstanding (DSO) reduction through accelerated collections.")
        bullets.append("• Review inventory days (DIO) for dealers with long cash cycles.")
        bullets.append("• Negotiate payment terms with suppliers to improve Days Payable Outstanding (DPO).")

    if "service" in q_lower or "turnaround" in q_lower or "repair" in q_lower:
        bullets.append("• Dealers exceeding 48-hour turnaround need process review and resource allocation.")
        bullets.append("• Share best practices from top-performing dealers with turnaround challenges.")
        bullets.append("• Implement SLA tracking to ensure consistent service delivery across network.")
        bullets.append("• Consider technician training or equipment upgrades for underperformers.")

    if "lead" in q_lower or "fulfillment" in q_lower or "delivery" in q_lower:
        bullets.append("• Dealers with lead times >7 days require logistics optimization review.")
        bullets.append("• Analyze root causes: supplier delays, transportation, or order processing.")
        bullets.append("• Establish tiered delivery SLAs based on order size and urgency.")
        bullets.append("• Coordinate with fulfillment partners to reduce variability in lead times.")

    if "revenue" in q_lower or "growth" in q_lower or "profit" in q_lower or "margin" in q_lower:
        bullets.append("• Identify high-growth dealers and document their sales strategies for replication.")
        bullets.append("• Low-growth dealers need targeted intervention with pricing or product mix reviews.")
        bullets.append("• Monitor trend direction: accelerating or declining growth informs resource allocation.")
        bullets.append("• Cross-reference growth with margin metrics to ensure profitable expansion.")

    if not bullets:
        bullets = [
            "• Review the data table and chart below to identify top/bottom performers.",
            "• Investigate outliers or anomalies that deviate significantly from network average.",
            "• Develop targeted action plans for dealers underperforming their peer group.",
            "• Monitor key metrics weekly to track progress toward goals.",
            "• Share insights with relevant teams (sales, operations, finance) for alignment."
        ]

    predictive_html = "<ul style='margin:0;padding-left:24px;'>"
    for bullet in bullets[:7]:
        # Remove bullet character if present and wrap in list item
        bullet_text = bullet.lstrip("• ").strip()
        predictive_html += f"<li style='margin-bottom:8px;line-height:1.6;'>{bullet_text}</li>"
    predictive_html += "</ul>"

    return descriptive, predictive_html


# ============================================================================
# STRATEGIC INSIGHTS SECTION
# ============================================================================
def render_insights(session):
    """Render dynamic strategic insights section with hide/show button"""

    # Check if insights are enabled in config
    if not get_config_value("sections.strategic_insights.enabled", True):
        return  # Skip rendering completely if disabled

    if st.session_state.get('show_insights', True):
        # Show insights box with hide button
        # Fetch insights data
        # Migration note: generate_dynamic_insights(session) → generate_dynamic_insights()
        insights_df = generate_dynamic_insights()

        # Build insights box
        insights_html = '<div style="background-color: #f3e8ff; border-radius: 12px; padding: 0; margin-bottom: 1.5rem; border-left: 4px solid #7c3aed; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden;">'

        # Header section with gradient
        insights_html += '<div style="padding: 1.25rem 1.5rem; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e9d5ff; background: linear-gradient(135deg, #faf5ff 0%, #f5edff 100%);">'
        insights_html += f'<div style="display: flex; align-items: center; gap: 10px;"><span style="font-size: {FONT_SIZE_LARGE}px;"></span><span style="font-size: {FONT_SIZE_TITLE}px; font-weight: {FONT_WEIGHT_BOLD}; color: {TEXT_DARK};">Strategic Insights</span></div>'
        insights_html += '</div>'

        # Content section
        insights_html += '<div style="padding: 1.5rem;">'
        insights_html += f'<div style="color: #666; font-weight: 600; margin-bottom: 1rem; font-size: {FONT_SIZE_SMALL}px; text-transform: uppercase; letter-spacing: 0.5px;">Key findings:</div>'

        if not insights_df.empty:
            # Add insights as bullet points
            for idx, row in insights_df.iterrows():
                insights_html += f'<div style="margin-bottom: 0.9rem; color: #333; font-size: {FONT_SIZE_NORMAL}px; line-height: 1.6; display: flex; gap: 8px;"><span style="color: #7c3aed; font-weight: 600; flex-shrink: 0;">•</span><span>{row["INSIGHT_TEXT"]}</span></div>'
        else:
            insights_html += f'<div style="color: #666; font-size: {FONT_SIZE_NORMAL}px; font-style: italic; text-align: center; padding: 0.5rem;">No insights available. Ensure views have data.</div>'

        # Footer
        insights_html += '<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #e9d5ff;">'
        insights_html += f'<small style="color: #999; font-size: {FONT_SIZE_SMALL}px;">Based on current KPI data</small>'
        insights_html += '</div></div>'

        insights_html += '</div>'

        # Only show one insights block and one hide button
        if st.session_state.get('show_insights', True):
            # Add hide button inside the purple block, top right (Streamlit native button only)
            insights_html = insights_html[:-6]  # Remove closing header div
            insights_html += '</div>'
            st.markdown(insights_html, unsafe_allow_html=True)
            if st.button("Hide Insights", key="hide_insights_btn", type="secondary"):
                st.session_state.show_insights = False
                st.rerun()
        else:
            # Place the unhide button in the header actions row, right-aligned
            actions_col = st.columns([4, 1.7])[1]
            with actions_col:
                if st.button("Show Strategic Insights", key="show_insights_btn", help="Show Strategic Insights"):
                    st.session_state.show_insights = True
                    st.rerun()


# ============================================================================
# KPI METRICS SECTION
# ============================================================================
def render_kpi_metrics(session, filters):
    """Render KPI metric cards for 10 new KPIs with dynamic deltas"""

    # Check if KPI metrics are enabled in config
    if not get_config_value("kpi_metrics.enabled", True):
        return  # Skip rendering if disabled

    # Validate view schemas first
    # Migration note: validate_view_schemas(session) → validate_view_schemas()
    if not validate_view_schemas():
        st.error("""
        **Required Action**: You need to update the Athena views to match the new schema.

        The views must include these columns:
        - DEALER_NAME (string)
        - PERIOD_YEAR (integer)
        - PERIOD_MONTH (integer)
        - PERIOD_START_DATE (date - optional)

        Please execute the updated view creation SQL provided by your data team.
        """)
        return

    # Fetch current period metrics
    # Migration note: fetch_*(session, filters) → fetch_*(filters)
    ccc = fetch_cash_conversion_cycle(filters)
    repair_tat = fetch_repair_turnaround_time(filters)
    revenue_growth = fetch_revenue_growth(filters)
    gross_margin = fetch_gross_profit_margin(filters)
    sales_per_product = fetch_sales_per_product_category(filters)
    lead_time = fetch_order_lead_time(filters)
    stock_avail = fetch_stock_availability(filters)
    backorder = fetch_backorder_incidence(filters)
    sales_vol = fetch_sales_volume(filters)
    contrib_margin = fetch_contribution_margin(filters)

    # Helper to compute previous period range
    def previous_period(from_date, to_date):
        if isinstance(from_date, datetime):
            from_date = from_date.date()
        if isinstance(to_date, datetime):
            to_date = to_date.date()
        length = (to_date - from_date).days + 1
        prev_to = from_date - timedelta(days=1)
        prev_from = prev_to - timedelta(days=length - 1)
        return prev_from, prev_to

    prev_from, prev_to = previous_period(filters['from_date'], filters['to_date'])
    prev_filters = filters.copy()
    prev_filters['from_date'] = prev_from
    prev_filters['to_date'] = prev_to

    # Fetch previous period metrics
    prev_ccc = fetch_cash_conversion_cycle(prev_filters)
    prev_repair_tat = fetch_repair_turnaround_time(prev_filters)
    prev_revenue_growth = fetch_revenue_growth(prev_filters)
    prev_gross_margin = fetch_gross_profit_margin(prev_filters)
    prev_sales_per_product = fetch_sales_per_product_category(prev_filters)
    prev_lead_time = fetch_order_lead_time(prev_filters)
    prev_stock_avail = fetch_stock_availability(prev_filters)
    prev_backorder = fetch_backorder_incidence(prev_filters)
    prev_sales_vol = fetch_sales_volume(prev_filters)
    prev_contrib_margin = fetch_contribution_margin(prev_filters)

    # Helper to compute delta
    def compute_delta(curr, prev, higher_is_good=True, unit='', fmt='{:+.1f}'):
        try:
            curr_v = float(curr)
            prev_v = float(prev)
        except Exception:
            return {'delta_text': 'N/A', 'color': '#6b7280'}

        # Validate that both values exist and are meaningful
        if prev_v == 0 or prev_v != prev_v or curr_v != curr_v:  # NaN check
            return {'delta_text': 'N/A', 'color': '#6b7280'}

        delta = curr_v - prev_v

        # Color logic: Green if metric improved, Red if deteriorated, Gray if no change
        color = '#22c55e' if (delta > 0 and higher_is_good) or (delta < 0 and not higher_is_good) else '#ef4444' if delta != 0 else '#6b7280'

        if abs(delta) >= 1 and fmt == '{:+.1f}':
            fmt = '{:+.0f}'
        delta_text = (fmt.format(delta)) + (unit if unit else '')
        return {'delta_text': delta_text, 'color': color}

    # Compute deltas
    delta_ccc = compute_delta(ccc, prev_ccc, higher_is_good=False, unit=' days', fmt='{:+.0f}')
    delta_repair_tat = compute_delta(repair_tat, prev_repair_tat, higher_is_good=False, unit=' hrs', fmt='{:+.0f}')
    delta_revenue_growth = compute_delta(revenue_growth, prev_revenue_growth, higher_is_good=True, unit='%', fmt='{:+.1f}')
    delta_gross_margin = compute_delta(gross_margin, prev_gross_margin, higher_is_good=True, unit='%', fmt='{:+.1f}')

    # For sales_per_product, calculate percentage change instead of absolute amount
    try:
        curr_val = float(sales_per_product)
        prev_val = float(prev_sales_per_product)
        if prev_val != 0:
            pct_change = ((curr_val - prev_val) / prev_val) * 100
            color = '#22c55e' if pct_change > 0 else '#ef4444' if pct_change < 0 else '#6b7280'
            delta_sales_per_product = {'delta_text': f'{pct_change:+.1f}%', 'color': color}
        else:
            delta_sales_per_product = {'delta_text': 'N/A', 'color': '#6b7280'}
    except Exception as e:
        delta_sales_per_product = {'delta_text': 'N/A', 'color': '#6b7280'}

    delta_lead_time = compute_delta(lead_time, prev_lead_time, higher_is_good=False, unit=' days', fmt='{:+.0f}')
    delta_stock_avail = compute_delta(stock_avail, prev_stock_avail, higher_is_good=True, unit='%', fmt='{:+.1f}')
    delta_backorder = compute_delta(backorder, prev_backorder, higher_is_good=False, unit='%', fmt='{:+.1f}')
    delta_sales_vol = compute_delta(sales_vol, prev_sales_vol, higher_is_good=True, unit='', fmt='{:+.0f}')
    delta_contrib_margin = compute_delta(contrib_margin, prev_contrib_margin, higher_is_good=True, unit='%', fmt='{:+.1f}')

    # Helper functions
    def safe_int(val, suffix=''):
        try:
            if val is None or (isinstance(val, float) and val != val):
                return 'N/A'
            return f"{int(round(val))}{suffix}"
        except Exception:
            return 'N/A'

    def safe_float_one(val, suffix=''):
        try:
            if val is None or (isinstance(val, float) and val != val):
                return 'N/A'
            return f"{float(val):.1f}{suffix}"
        except Exception:
            return 'N/A'

    def safe_currency(val):
        try:
            if val is None or (isinstance(val, float) and val != val):
                return 'N/A'
            v = float(val)
            if abs(v) >= 1_000_000_000:
                return f"{v/1_000_000_000:.1f}B"
            elif abs(v) >= 1_000_000:
                return f"{v/1_000_000:.1f}M"
            elif abs(v) >= 1_000:
                return f"{v/1_000:.1f}K"
            else:
                return f"{int(round(v)):,}"
        except Exception:
            return 'N/A'

    # Build KPI list with 10 items
    kpis = [
        {'title': 'CASH CONVERSION CYCLE', 'value': safe_int(ccc, ' days'), 'delta': delta_ccc['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_ccc['color']},
        {'title': 'AVG REPAIR TAT', 'value': safe_int(repair_tat, ' hrs'), 'delta': delta_repair_tat['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_repair_tat['color']},
        {'title': 'REVENUE GROWTH', 'value': safe_float_one(revenue_growth, '%'), 'delta': delta_revenue_growth['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_revenue_growth['color']},
        {'title': 'GROSS PROFIT MARGIN', 'value': safe_float_one(gross_margin, '%'), 'delta': delta_gross_margin['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_gross_margin['color']},
        {'title': 'SALES', 'value': safe_currency(sales_per_product), 'delta': delta_sales_per_product['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_sales_per_product['color']},
        {'title': 'ORDER LEAD TIME', 'value': safe_int(lead_time, ' days'), 'delta': delta_lead_time['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_lead_time['color']},
        {'title': 'STOCK AVAILABILITY %', 'value': safe_float_one(stock_avail, '%'), 'delta': delta_stock_avail['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_stock_avail['color']},
        {'title': 'BACKORDER INCIDENCE %', 'value': safe_float_one(backorder, '%'), 'delta': delta_backorder['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_backorder['color']},
        {'title': 'TOTAL UNITS SOLD', 'value': safe_int(sales_vol), 'delta': delta_sales_vol['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_sales_vol['color']},
        {'title': 'CONTRIBUTION MARGIN %', 'value': safe_float_one(contrib_margin, '%'), 'delta': delta_contrib_margin['delta_text'], 'delta_text': 'vs previous period', 'delta_color': delta_contrib_margin['color']}
    ]

    # Generate alerts based on all dealers' KPI thresholds
    # Migration note: generate_kpi_alerts(session, filters) → generate_kpi_alerts(filters)
    st.session_state.kpi_alerts_df = generate_kpi_alerts(filters)

    # Read layout configuration from ui_config.yaml
    items_per_row = get_config_value("kpi_metrics.items_per_row", 5)
    num_rows = get_config_value("kpi_metrics.rows", 2)

    # Display KPIs in configured layout
    for row in range(num_rows):
        kpi_cols = st.columns(items_per_row, gap='small')
        for col_idx, col in enumerate(kpi_cols):
            kpi_index = row * items_per_row + col_idx
            if kpi_index < len(kpis):
                item = kpis[kpi_index]
                with col:
                    st.markdown(f"""
                    <div style="background:#ffffff;border-radius:12px;padding:18px;border:1px solid #eef2f6;box-shadow:0 1px 3px rgba(16,24,40,0.04);min-height:92px;margin-bottom:18px;">
                        <div style="font-size:{FONT_SIZE_SMALL}px;font-weight:700;color:#6b7280;margin-bottom:6px;letter-spacing:0.6px;">{item['title']}</div>
                        <div style="display:flex;align-items:baseline;justify-content:space-between;gap:12px;">
                            <div style="font-size:{FONT_SIZE_LARGE + 10}px;font-weight:800;color:#111;">{item['value']}</div>
                            <div style="text-align:right;">
                                <div style="font-size:{FONT_SIZE_NORMAL}px;color:{item['delta_color']};font-weight:700;">{item['delta']}</div>
                                <div style="font-size:{FONT_SIZE_SMALL}px;color:#6b7280;margin-top:4px;">{item['delta_text']}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)


# ============================================================================
# ATTENTION SECTION (full width, 8 cards in 4-up / 4-below grid)
# ============================================================================
def render_attention_and_priority(session, filters):
    """
    Render Dealer Health block.
    ONE compact card per dealer — all 8 KPIs fetched with the SAME filters
    (including from_date/to_date) as render_dealer_health_scorecard.
    """
    import re as _re
    import html

    # Migration note: fetch_attention_items(session) → fetch_attention_items()
    attention_df = fetch_attention_items()

    if attention_df.empty:
        dealer_list = []
    else:
        def parse_dealer_name(desc):
            m = _re.match(r'\[([^\]]+)\]', str(desc))
            return m.group(1).strip() if m else None

        attention_df = attention_df.copy()
        attention_df['_DEALER'] = attention_df['ISSUE_DESCRIPTION'].apply(parse_dealer_name)
        dealer_list = sorted(attention_df['_DEALER'].dropna().unique().tolist())

    # ── Scoring functions — kept in sync with render_dealer_health_scorecard ──
    def s_revenue_growth(v):
        if v < 0:   return 1
        if v < 2:   return 2
        if v < 4:   return 3
        if v < 6:   return 4
        if v < 8:   return 5
        if v < 12:  return 6
        return 7

    def s_sales_volume(v, avg_vol):
        if avg_vol is None or avg_vol == 0: return 4
        r = (v / avg_vol) * 100
        if r < 50:  return 1
        if r < 65:  return 2
        if r < 80:  return 3
        if r < 90:  return 4
        if r < 100: return 5
        if r < 115: return 6
        return 7

    def s_stock_avail(v):
        if v < 40:  return 1
        if v < 55:  return 2
        if v < 65:  return 3
        if v < 75:  return 4
        if v < 85:  return 5
        if v < 93:  return 6
        return 7

    def s_repair_tat(v):
        if v > 72:  return 1
        if v > 60:  return 2
        if v > 48:  return 3
        if v > 36:  return 4
        if v > 24:  return 5
        if v > 12:  return 6
        return 7

    def s_backorder(v):
        if v > 20:  return 1
        if v > 15:  return 2
        if v > 10:  return 3
        if v > 7:   return 4
        if v > 4:   return 5
        if v > 2:   return 6
        return 7

    def s_ccc(v):
        if v > 60:  return 1
        if v > 45:  return 2
        if v > 30:  return 3
        if v > 20:  return 4
        if v > 12:  return 5
        if v > 6:   return 6
        return 7

    def s_gross_margin(v):
        if v < 10:  return 1
        if v < 20:  return 2
        if v < 30:  return 3
        if v < 40:  return 4
        if v < 55:  return 5
        if v < 65:  return 6
        return 7

    def s_contrib_margin(v):
        if v < 5:   return 1
        if v < 15:  return 2
        if v < 25:  return 3
        if v < 35:  return 4
        if v < 45:  return 5
        if v < 55:  return 6
        return 7

    def safe_score(val, fn, *args):
        """Returns 0 for missing data (excluded from avg), score otherwise."""
        if val is None:
            return 0
        try:
            import math
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return 0
            return fn(fval, *args)
        except Exception:
            return 0

    def score_to_severity(score: float) -> str:
        if score <= 2: return 'Critical'
        if score <= 4: return 'Average'
        return 'Good'   # Good + Excellent both go to Good tab

    def score_color(s: float):
        if s <= 2: return '#dc2626', '#fef2f2'
        if s <= 4: return '#f59e0b', '#fffbeb'
        if s < 6:  return '#3b82f6', '#eff6ff'
        return '#16a34a', '#f0fdf4'

    def score_label(s: float) -> str:
        if s == 0: return 'No Data'
        if s <= 2: return 'Critical'
        if s <= 4: return 'Average'
        if s < 6:  return 'Good'
        return 'Excellent'

    # ── BATCH KPI fetch: 8 queries total for ALL dealers (was 8 × N) ──────────
    # This replaces the old per-dealer loop that ran 160+ queries for 20 dealers.
    # Each query returns one row per dealer; we build a lookup dict keyed by
    # DEALER_NAME for O(1) access when building dealer_summaries.

    fd_str = str(filters.get('from_date', ''))
    td_str = str(filters.get('to_date', ''))

    def _batch(sql):
        """Run a batch SQL, return {dealer_name: value} dict. Silent on error."""
        try:
            # Migration note: session.sql(sql).to_pandas() → athena_query(sql)
            df = athena_query(sql)
            df.columns = df.columns.str.upper()
            if df.empty or 'DEALER_NAME' not in df.columns:
                return {}
            val_col = [c for c in df.columns if c != 'DEALER_NAME'][0]
            return {str(r['DEALER_NAME']): r[val_col] for _, r in df.iterrows()}
        except Exception:
            return {}

    # Revenue growth per dealer
    # Migration note: DATE_TRUNC('MONTH', val::DATE) → date_trunc('month', cast(val as date))
    _rg_map = _batch(f"""
        SELECT DEALER_NAME, AVG(REVENUE_GROWTH_MOM_PERCENT) AS V
        FROM {_DB}.VW_DEALER_REVENUE_GROWTH
        WHERE REVENUE_GROWTH_MOM_PERCENT IS NOT NULL
          AND PERIOD_MONTH >= date_trunc('month', cast('{fd_str}' as date))
          AND PERIOD_MONTH <= date_trunc('month', cast('{td_str}' as date))
        GROUP BY DEALER_NAME
    """)

    # Sales volume per dealer
    _sv_map = _batch(f"""
        SELECT DEALER_NAME, SUM(UNITS_SOLD) AS V
        FROM {_DB}.VW_SALES_VOLUME
        WHERE UNITS_SOLD IS NOT NULL
          AND PERIOD_START_DATE >= '{fd_str}' AND PERIOD_START_DATE <= '{td_str}'
        GROUP BY DEALER_NAME
    """)
    _avg_sales_vol = (sum(_sv_map.values()) / len(_sv_map)) if _sv_map else None

    # Stock availability per dealer
    _sa_map = _batch(f"""
        SELECT DEALER_NAME, AVG(STOCK_AVAILABILITY_PCT) AS V
        FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER
        WHERE STOCK_AVAILABILITY_PCT IS NOT NULL
          AND PERIOD_START_DATE >= '{fd_str}' AND PERIOD_START_DATE <= '{td_str}'
        GROUP BY DEALER_NAME
    """)

    # Repair TAT per dealer
    _tat_map = _batch(f"""
        SELECT DEALER_NAME, AVG(AVG_TURNAROUND_HOURS) AS V
        FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME
        WHERE AVG_TURNAROUND_HOURS IS NOT NULL
          AND PERIOD_START_DATE >= '{fd_str}' AND PERIOD_START_DATE <= '{td_str}'
        GROUP BY DEALER_NAME
    """)

    # Backorder per dealer
    _bo_map = _batch(f"""
        SELECT DEALER_NAME, AVG(BACKORDER_INCIDENCE_PCT) AS V
        FROM {_DB}.VW_BACKORDER_INCIDENCE
        WHERE BACKORDER_INCIDENCE_PCT IS NOT NULL
          AND PERIOD_START_DATE >= '{fd_str}' AND PERIOD_START_DATE <= '{td_str}'
        GROUP BY DEALER_NAME
    """)

    # CCC per dealer
    _ccc_map = _batch(f"""
        SELECT DEALER_NAME, AVG(CCC) AS V
        FROM {_DB}.VW_CASH_CONVERSION_CYCLE
        WHERE CCC IS NOT NULL
          AND PERIOD_MONTH >= date_trunc('month', cast('{fd_str}' as date))
          AND PERIOD_MONTH <= date_trunc('month', cast('{td_str}' as date))
        GROUP BY DEALER_NAME
    """)

    # Gross margin per dealer
    _gm_map = _batch(f"""
        SELECT DEALER_NAME, AVG(GROSS_PROFIT_MARGIN_PCT) AS V
        FROM {_DB}.VW_GROSS_PROFIT_MARGIN
        WHERE GROSS_PROFIT_MARGIN_PCT IS NOT NULL
          AND PERIOD_MONTH >= date_trunc('month', cast('{fd_str}' as date))
          AND PERIOD_MONTH <= date_trunc('month', cast('{td_str}' as date))
        GROUP BY DEALER_NAME
    """)

    # Contribution margin per dealer
    _cm_map = _batch(f"""
        SELECT DEALER_NAME, AVG(CONTRIBUTION_MARGIN_PCT) AS V
        FROM {_DB}.VW_DEALER_CONTRIBUTION_MARGIN
        WHERE CONTRIBUTION_MARGIN_PCT IS NOT NULL
          AND PERIOD_START_DATE >= '{fd_str}' AND PERIOD_START_DATE <= '{td_str}'
        GROUP BY DEALER_NAME
    """)

    def _v(m, dealer):
        """Safe lookup from batch map — returns None if missing or NaN."""
        import math as _m
        val = m.get(dealer)
        if val is None:
            return None
        try:
            f = float(val)
            return None if (_m.isnan(f) or _m.isinf(f)) else f
        except Exception:
            return None

    # ── Build ONE summary per dealer (pure Python now — no more DB calls) ────
    dealer_summaries = []

    for dealer in dealer_list:
        revenue_growth = _v(_rg_map, dealer)
        sales_vol      = _v(_sv_map, dealer)
        stock_avail    = _v(_sa_map, dealer)
        repair_tat     = _v(_tat_map, dealer)
        backorder      = _v(_bo_map, dealer)
        ccc            = _v(_ccc_map, dealer)
        gross_margin   = _v(_gm_map, dealer)
        contrib_margin = _v(_cm_map, dealer)

        all_scores = [
            safe_score(revenue_growth, s_revenue_growth),
            safe_score(sales_vol,      s_sales_volume, _avg_sales_vol),
            safe_score(stock_avail,    s_stock_avail),
            safe_score(repair_tat,     s_repair_tat),
            safe_score(backorder,      s_backorder),
            safe_score(ccc,            s_ccc),
            safe_score(gross_margin,   s_gross_margin),
            safe_score(contrib_margin, s_contrib_margin),
        ]

        # Exclude 0s (missing data) from average — same logic as scorecard
        valid_scores = [s for s in all_scores if s > 0]
        avg_score    = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0
        severity_tab = score_to_severity(avg_score)
        ov_color, ov_bg = score_color(avg_score)
        ov_label     = score_label(avg_score)

        # Flagged issues from attention_df for card footer
        issue_lines = []
        if not attention_df.empty:
            for _, row in attention_df[attention_df['_DEALER'] == dealer].iterrows():
                clean = _re.sub(r'^\[[^\]]+\]\s*', '',
                                str(row['ISSUE_DESCRIPTION'])).strip()
                issue_lines.append(clean)

        dealer_summaries.append({
            'dealer':       dealer,
            'avg_score':    avg_score,
            'severity_tab': severity_tab,
            'ov_color':     ov_color,
            'ov_bg':        ov_bg,
            'ov_label':     ov_label,
            'issue_lines':  issue_lines,
        })

    # ── Session-state defaults ────────────────────────────────────────────────
    if 'attention_page' not in st.session_state:
        st.session_state.attention_page = 0
    if 'attention_severity_tab' not in st.session_state:
        st.session_state.attention_severity_tab = "Good"
    if st.session_state.attention_severity_tab not in ["Critical", "Average", "Good"]:
        st.session_state.attention_severity_tab = "Good"

    def _count(tab):
        return sum(1 for d in dealer_summaries if d['severity_tab'] == tab)

    critical_count = _count('Critical')
    average_count  = _count('Average')
    good_count     = _count('Good')
    total_count    = len(dealer_summaries)

    with st.container(border=True):

        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:0.75rem;'>
            <div style='font-size:18px;font-weight:900;color:#1a1a1a;letter-spacing:.2px;'>
                Dealer Health
                <span style='font-weight:700;color:#6b7280;'>({total_count:,})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        current_tab = st.session_state.attention_severity_tab

        tab_cols = st.columns([1, 1, 1], gap="small")
        with tab_cols[0]:
            if st.button(f"Critical ({critical_count})", key="na_btn_critical", width='stretch'):
                st.session_state.attention_severity_tab = 'Critical'
                st.session_state.attention_page = 0
                st.rerun()
        with tab_cols[1]:
            if st.button(f"Average ({average_count})", key="na_btn_average", width='stretch'):
                st.session_state.attention_severity_tab = 'Average'
                st.session_state.attention_page = 0
                st.rerun()
        with tab_cols[2]:
            if st.button(f"Good ({good_count})", key="na_btn_good", width='stretch'):
                st.session_state.attention_severity_tab = 'Good'
                st.session_state.attention_page = 0
                st.rerun()

        st.markdown(f"""
        <style>
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_critical'] { background: #fef2f2 !important; color: #dc2626 !important; border-color: #fecaca !important; font-weight: 800 !important; }" if current_tab == 'Critical' else ""}
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_average']  { background: #fffbeb !important; color: #f59e0b !important; border-color: #fde68a !important; font-weight: 800 !important; }" if current_tab == 'Average'  else ""}
        {"div[data-testid='stButton'] button[data-testid='baseButton-na_btn_good']     { background: #f0fdf4 !important; color: #16a34a !important; border-color: #bbf7d0 !important; font-weight: 800 !important; }" if current_tab == 'Good'     else ""}
        button[data-testid^="baseButton-dealer_pill_"] {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #1a1a1a !important;
            font-weight: 900 !important;
            font-size: 13px !important;
            padding: 0 !important;
            margin: 0 !important;
            cursor: pointer !important;
            text-align: left !important;
            text-decoration: underline !important;
        }}
        button[data-testid^="baseButton-dealer_pill_"] p {{
            font-weight: 900 !important;
            font-size: 13px !important;
        }}
        button[data-testid^="baseButton-dealer_pill_"]:hover {{
            color: #2563eb !important;
            text-decoration: underline !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

        if current_tab == 'Good':
            border_color, card_bg, tab_class = "#fecaca", "#fef2f2", "critical"
        elif current_tab == 'Average':
            border_color, card_bg, tab_class = "#fde68a", "#fffbeb", "average"
        else:
            border_color, card_bg, tab_class = "#bbf7d0", "#f0fdf4", "good"

        st.markdown(f"""
        <style>
        [class*="st-key-na_bg_{tab_class}_"] {{
            background: {card_bg} !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,.05) !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        tab_dealers = [d for d in dealer_summaries if d['severity_tab'] == current_tab]

        items_per_page = 8
        total_items    = len(tab_dealers)
        total_pages    = max(1, (total_items + items_per_page - 1) // items_per_page)

        if total_items == 0:
            st.markdown(
                f"<div style='text-align:center;padding:40px;color:#9ca3af;font-size:14px;'>"
                f"No <b>{current_tab}</b> dealers found.</div>",
                unsafe_allow_html=True
            )
        else:
            start_idx  = st.session_state.attention_page * items_per_page
            page_items = tab_dealers[start_idx: start_idx + items_per_page]

            def render_dealer_card(d: dict, col_obj, g_idx: int):
                dealer      = d['dealer']
                avg_score   = d['avg_score']
                ov_color    = d['ov_color']
                ov_label    = d['ov_label']
                issue_lines = d['issue_lines']
                bar_pct     = int((avg_score / 7) * 100)

                with col_obj:
                    with st.container(border=True,
                                      key=f"na_bg_{tab_class}_{g_idx}"):

                        top_l, top_r = st.columns([3, 2], gap="small")
                        with top_l:
                            pill_key = (
                                f"dealer_pill_{g_idx}_{start_idx}_"
                                f"{str(dealer).replace(' ', '_')[:25]}"
                            )
                            if st.button(f"{dealer}", key=pill_key):
                                st.session_state['current_page']       = 'Dealer Life Cycle'
                                st.session_state['dlc_prefill_dealer'] = dealer
                                st.session_state['scroll_to_top']      = True
                                st.rerun()
                        with top_r:
                            st.markdown(
                                f"<div style='text-align:right;padding-top:6px;'>"
                                f"<span style='color:{ov_color};font-size:12px;"
                                f"font-weight:800;'>{ov_label}</span>"
                                f"</div>",
                                unsafe_allow_html=True
                            )

                        st.markdown(
                            f"<div style='display:flex;align-items:center;"
                            f"gap:8px;margin:2px 0 8px 0;'>"
                            f"<div style='font-size:20px;font-weight:800;"
                            f"color:{ov_color};line-height:1;'>{avg_score:.1f}"
                            f"<span style='font-size:10px;color:#9ca3af;"
                            f"font-weight:400;'> /7</span></div>"
                            f"<div style='flex:1;background:#e5e7eb;"
                            f"border-radius:4px;height:6px;'>"
                            f"<div style='width:{bar_pct}%;background:{ov_color};"
                            f"border-radius:4px;height:6px;'></div>"
                            f"</div></div>",
                            unsafe_allow_html=True
                        )

                        issues_html = "".join(
                            f"<div style='font-size:11px;color:#64748b;"
                            f"white-space:nowrap;overflow:hidden;"
                            f"text-overflow:ellipsis;margin-bottom:2px;'>"
                            f"{html.escape(line)}</div>"
                            for line in issue_lines[:2]
                        )
                        if len(issue_lines) > 2:
                            issues_html += (
                                f"<div style='font-size:10px;color:#9ca3af;'>"
                                f"+{len(issue_lines)-2} more issue(s)</div>"
                            )
                        if issues_html:
                            st.markdown(issues_html, unsafe_allow_html=True)

            chunk1 = page_items[:4]
            cols1  = st.columns(4, gap="medium")
            for c_idx, d in enumerate(chunk1):
                render_dealer_card(d, cols1[c_idx], c_idx)

            chunk2 = page_items[4:8]
            if chunk2:
                st.write("")
                cols2 = st.columns(4, gap="medium")
                for c_idx, d in enumerate(chunk2):
                    render_dealer_card(d, cols2[c_idx], c_idx + 4)

        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        pag_cols = st.columns([1, 1, 1], gap="small")

        with pag_cols[0]:
            if st.session_state.attention_page > 0:
                if st.button("← Prev", key="na_prev_bottom", width='stretch'):
                    st.session_state.attention_page -= 1
                    st.rerun()
            else:
                st.markdown(
                    "<div style='text-align:center;color:#d1d5db;"
                    "font-size:14px;padding:10px;'>← Prev</div>",
                    unsafe_allow_html=True
                )
        with pag_cols[1]:
            st.markdown(
                f"<div style='text-align:center;font-weight:500;color:#6b7280;"
                f"font-size:14px;padding:10px;'>"
                f"{st.session_state.attention_page + 1} of {total_pages}</div>",
                unsafe_allow_html=True
            )
        with pag_cols[2]:
            if st.session_state.attention_page < total_pages - 1:
                if st.button("Next →", key="na_next_bottom", width='stretch'):
                    st.session_state.attention_page += 1
                    st.rerun()
            else:
                st.markdown(
                    "<div style='text-align:center;color:#d1d5db;"
                    "font-size:14px;padding:10px;'>Next →</div>",
                    unsafe_allow_html=True
                )


# ============================================================================
# GRAPH DATA FETCH FUNCTIONS (skipped — all in data_service.py)
# ============================================================================
# fetch_revenue_trend, fetch_profit_margin_by_dealer, fetch_sales_by_product_category,
# fetch_cash_conversion_cycle_trend, fetch_order_lead_time_distribution
# → imported from data_service

# ============================================================================
# GRAPHS/CHARTS SECTION
# ============================================================================
# -----------------------------
# Design tokens (consistent UI)
# -----------------------------
PRIMARY = "#2563EB"   # blue
SECONDARY = "#06B6D4" # cyan
SUCCESS = "#10B981"   # teal/green
INFO = "#6366F1"      # indigo
NEUTRAL = "#8B5CF6"   # purple
ACCENT = "#3B82F6"    # bright blue
GRID    = "#E5E7EB"   # light grey for axes/grid
TEXT    = "#111827"   # primary text
MUTED   = "#6B7280"   # muted/secondary text

# -----------------------------
# Helpers
# -----------------------------
def _apply_pro_theme(fig, title: str = "", height: int = 360, bg_color: str = "#FBF4F9"):
    """
    Uniform modern chart theme across the dashboard (light background, app-aligned).

    - Rounded, minimal grid
    - Muted axes / strong data colors
    - Compact margins and horizontal legends
    - Background color matches app theme
    """
    import colorsys
    bg_hex = bg_color.lstrip('#')
    bg_rgb = tuple(int(bg_hex[i:i+2], 16)/255 for i in (0, 2, 4))
    hsv = colorsys.rgb_to_hsv(*bg_rgb)
    lighter_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1] * 0.5, min(hsv[2] + 0.15, 1.0))
    plot_bg = f"rgba({int(lighter_rgb[0]*255)},{int(lighter_rgb[1]*255)},{int(lighter_rgb[2]*255)},0.3)"

    fig.update_layout(
        template="simple_white",
        height=height,
        margin=dict(l=16, r=16, t=42 if title else 18, b=24),
        title=dict(
            text=title,
            x=0.0,
            xanchor="left",
            font=dict(size=15, color=TEXT, family="Inter, Segoe UI, sans-serif"),
        ),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12.5, color=TEXT),
        paper_bgcolor="rgba(248,250,252,0)",
        plot_bgcolor=plot_bg,
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#020617",
            font=dict(color="white", size=12),
            bordercolor="#020617",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
            font=dict(color=MUTED, size=11.5),
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11, weight='bold'),
        title_font=dict(color=MUTED, size=12, weight='bold'),
        linecolor=GRID,
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=False,
        gridcolor="rgba(226,232,240,0.55)",
        gridwidth=0.6,
        zeroline=False,
        tickfont=dict(color=MUTED, size=11, weight='bold'),
        title_font=dict(color=MUTED, size=12, weight='bold'),
        linecolor="rgba(226,232,240,0.9)",
        automargin=True,
    )
    return fig


def _hover_money(name="Value"):
    return f"{name}: $%{{y:,.0f}}<extra></extra>"


def _hover_pct(name="Value"):
    return f"{name}: %{{y:.1f}}%<extra></extra>"


def _hover_days(name="Value"):
    return f"{name}: %{{x:.1f}} days<extra></extra>"


def _chart_card_open(title=None):
    """Open a chart card container - returns the container object"""
    return st.container(border=True)


def _chart_card_close():
    """Close a chart card - no-op now since we use context managers"""
    pass


def _safe_has_cols(df, cols):
    return (df is not None) and (not df.empty) and all(c in df.columns for c in cols)


def render_visualizations(sla_lead_time_days: float = 7.0):
    # Migration note: session parameter removed; fetch functions use athena_query() internally
    """
    Render the analytics section with dealer-focused visualizations.
    Layout:
      - Revenue vs Gross Profit Margin (combo chart)
      - Sales mix by product category (grouped bars)
      - Cash Conversion Cycle by dealer (single, color-coded bar per dealer)
      - Order lead time by dealer (SLA-aware horizontal bars)
    Designed to be readable at a glance for non-analyst users.
    """

    # Dashboard header
    st.markdown("###  Analytics")
    st.caption(
        "Dealer performance at a glance – clean, focused visualizations designed for quick scanning."
    )

    # Card CSS (PowerBI / modern BI–style containers)
    st.markdown(
        """
        <style>
        /* CSS for chart containers removed - now using st.container(border=True) */
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------
    # Row 1 (2 charts)
    # ---------------------------------
    col1, col2 = st.columns(2)

    # ========== CHART 1: Revenue vs Gross Profit Margin (Modern Combo) ==========
    with col1:
        # Migration note: fetch_profit_margin_by_dealer(session) → fetch_profit_margin_by_dealer()
        profit_df = fetch_profit_margin_by_dealer()

        if _safe_has_cols(profit_df, ["dealer_name", "total_revenue", "gross_profit_margin_pct"]):
            profit_df = profit_df.sort_values("total_revenue", ascending=False)

            fig = go.Figure()

            fig.add_bar(
                x=profit_df["dealer_name"],
                y=profit_df["total_revenue"],
                name="Revenue",
                marker=dict(
                    color=PRIMARY,
                    opacity=0.88,
                    line=dict(color="rgba(15,23,42,0.08)", width=0.8),
                ),
                hovertemplate="Dealer: %{x}<br>" + _hover_money("Revenue"),
            )

            fig.add_scatter(
                x=profit_df["dealer_name"],
                y=profit_df["gross_profit_margin_pct"],
                name="GPM %",
                mode="lines+markers",
                line_shape="spline",
                yaxis="y2",
                line=dict(color=SECONDARY, width=3),
                marker=dict(
                    size=8,
                    color=SECONDARY,
                    line=dict(color="white", width=2),
                    symbol="circle",
                ),
                hovertemplate="Dealer: %{x}<br>" + _hover_pct("GPM"),
            )

            avg_gpm = float(profit_df["gross_profit_margin_pct"].mean())
            fig.add_hline(
                y=avg_gpm,
                line_dash="dash",
                line_color=SECONDARY,
                opacity=0.3,
                annotation_text="Avg GPM",
                annotation_position="top left",
                annotation_font_size=11,
                yref="y2",
            )

            fig.update_layout(
                bargap=0.12,
                yaxis=dict(title="Revenue", tickprefix="$", tickformat="~s"),
                yaxis2=dict(
                    title="GPM %",
                    overlaying="y",
                    side="right",
                    ticksuffix="%",
                    showgrid=False,
                ),
            )

            bg_color = st.session_state.get('bg_color', '#FBF4F9')
            _apply_pro_theme(fig, "", height=360, bg_color=bg_color)

            with _chart_card_open():
                st.markdown("**Revenue vs Gross Profit Margin**")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No profit margin data available.")

    # ========== CHART 2: Sales by Product Category (Donut Chart) ==========
    with col2:
        # Migration note: fetch_sales_by_product_category(session) → fetch_sales_by_product_category()
        sales_df = fetch_sales_by_product_category()

        if _safe_has_cols(sales_df, ["product_category", "total_revenue", "total_quantity"]):
            sales_df = sales_df.sort_values("total_revenue", ascending=False)

            colors_palette = [PRIMARY, SECONDARY, INFO, NEUTRAL, ACCENT, "#6366F1", "#8B5CF6", "#06B6D4"]

            fig = go.Figure(data=[go.Pie(
                labels=sales_df["product_category"],
                values=sales_df["total_revenue"],
                hole=0.45,
                marker=dict(
                    colors=colors_palette[:len(sales_df)],
                    line=dict(color="white", width=2),
                ),
                textposition="auto",
                textinfo="label+percent",
                textfont=dict(size=11, color="white", family="Inter, sans-serif"),
                hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
                showlegend=True,
            )])

            fig.update_layout(
                height=360,
                margin=dict(l=16, r=16, t=42, b=24),
                paper_bgcolor="rgba(248,250,252,0)",
                font=dict(family="Inter, Segoe UI, sans-serif", size=11, color=TEXT),
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    bgcolor="rgba(255,255,255,0)",
                    bordercolor="rgba(148,163,184,0)",
                    font=dict(color=MUTED, size=10.5),
                ),
                showlegend=True,
            )

            with _chart_card_open():
                st.markdown("**Sales Mix by Product Category**")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No product sales data available.")

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # ---------------------------------
    # Row 2 (2 charts)
    # ---------------------------------
    col3, col4 = st.columns(2)

    # ========== CHART 3: Cash Conversion Cycle by Dealer (Modern Bar) ==========
    with col3:
        # Migration note: fetch_cash_conversion_cycle_trend(session) → fetch_cash_conversion_cycle_trend()
        ccc_df = fetch_cash_conversion_cycle_trend()

        if _safe_has_cols(ccc_df, ["dealer_name", "dso", "dio", "dpo"]):
            ccc_df = ccc_df.copy()
            ccc_df["ccc"] = ccc_df["dso"] + ccc_df["dio"] - ccc_df["dpo"]

            ccc_df = ccc_df.sort_values("ccc", ascending=False)

            ccc_target = float(ccc_df["ccc"].median()) if not ccc_df["ccc"].empty else 45.0
            colors = []
            for v in ccc_df["ccc"]:
                if v <= ccc_target * 0.8:
                    colors.append(SECONDARY)  # cyan - healthy
                elif v <= ccc_target * 1.2:
                    colors.append(ACCENT)     # bright blue - watch
                else:
                    colors.append(PRIMARY)    # darker blue - at risk

            fig = go.Figure()

            fig.add_bar(
                x=ccc_df["ccc"],
                y=ccc_df["dealer_name"],
                orientation="h",
                marker=dict(
                    color=colors,
                    opacity=0.88,
                    line=dict(color="rgba(15,23,42,0.06)", width=0.6),
                    cornerradius="10px",
                ),
                hovertemplate=(
                    "Dealer: %{y}<br>"
                    "CCC: %{x:.1f} days<br>"
                    "DSO: %{customdata[0]:.1f} days<br>"
                    "DIO: %{customdata[1]:.1f} days<br>"
                    "DPO: %{customdata[2]:.1f} days<extra></extra>"
                ),
                customdata=ccc_df[["dso", "dio", "dpo"]].to_numpy(),
            )

            fig.add_vrect(
                x0=0,
                x1=ccc_target,
                fillcolor="rgba(6,182,212,0.05)",
                layer="below",
                line_width=0,
            )
            fig.add_vline(
                x=ccc_target,
                line_dash="dash",
                line_color=SECONDARY,
                opacity=0.7,
                annotation_text=f"Target: {ccc_target:.0f}d",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(255,255,255,0.9)",
                annotation_bordercolor=SECONDARY,
                annotation_borderwidth=1,
            )

            fig.update_layout(
                xaxis=dict(title="Cash Conversion Cycle (Days)"),
                yaxis=dict(title="Dealer"),
                bargap=0.14,
                margin=dict(l=16, r=16, t=55, b=24),
            )

            _apply_pro_theme(fig, "", height=360)

            with _chart_card_open():
                st.markdown("**Cash Conversion Cycle by Dealer**")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No CCC data available.")

    # ========== CHART 4: Order Lead Time by Dealer ==========
    with col4:
        # Migration note: fetch_order_lead_time_distribution(session) → fetch_order_lead_time_distribution()
        lead_time_df = fetch_order_lead_time_distribution()

        if _safe_has_cols(lead_time_df, ["dealer_name", "avg_lead_time"]):
            lead_time_df = lead_time_df.sort_values("avg_lead_time", ascending=False)
            lead_time_df = lead_time_df.copy()

            max_dealers_display = 8
            if len(lead_time_df) > max_dealers_display:
                top = lead_time_df.head(max_dealers_display)
                others = lead_time_df.iloc[max_dealers_display:]
                if not others.empty:
                    others_row = {
                        "dealer_name": "Others (avg)",
                        "avg_lead_time": float(others["avg_lead_time"].mean()),
                    }
                    lead_time_df = pd.concat([top, pd.DataFrame([others_row])], ignore_index=True)
            else:
                pass

            lead_time_df["delta_vs_sla"] = lead_time_df["avg_lead_time"] - sla_lead_time_days

            colors = []
            for delta in lead_time_df["delta_vs_sla"]:
                if delta <= 0:
                    colors.append(SECONDARY)
                elif delta <= 2:
                    colors.append(INFO)
                else:
                    colors.append(PRIMARY)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=lead_time_df["avg_lead_time"],
                x=lead_time_df["dealer_name"],
                orientation="v",
                marker=dict(
                    color=colors,
                    opacity=0.85,
                    line=dict(color="rgba(15,23,42,0.15)", width=1),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Avg Lead Time: %{x:.1f} days<br>"
                    "Vs SLA: %{customdata:.1f} days<extra></extra>"
                ),
                customdata=lead_time_df["delta_vs_sla"],
            ))

            fig.add_hline(
                y=sla_lead_time_days,
                line_dash="dash",
                line_color=SECONDARY,
                opacity=0.75,
                annotation_text=f"SLA: {sla_lead_time_days:g}d",
                annotation_position="top right",
                annotation_font_size=10,
                annotation_bgcolor="rgba(255,255,255,0.92)",
                annotation_bordercolor=SECONDARY,
                annotation_borderwidth=1,
            )

            fig.update_layout(
                xaxis=dict(title="Dealer", rangemode="tozero", automargin=True),
                yaxis=dict(title="Days", automargin=True),
                showlegend=False,
                margin=dict(l=100, r=16, t=45, b=24),
                hovermode="closest",
                bargap=0.5,
                height=380,
            )

            bg_color = st.session_state.get('bg_color', '#FBF4F9')
            _apply_pro_theme(fig, "", height=360, bg_color=bg_color)

            with _chart_card_open():
                st.markdown("**Order Lead Time by Dealer**")
                st.plotly_chart(fig, width='stretch')
        else:
            st.info("No lead time data available.")

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)


# ============================================================================
# GENIE PAGE HELPER FUNCTIONS
# ============================================================================

def run_df(sqlstr: str) -> pd.DataFrame:
    # Migration note: get_active_session().sql() → athena_query()
    """Execute SQL query and return pandas DataFrame."""
    try:
        if not sqlstr or not sqlstr.strip():
            return pd.DataFrame()
        return athena_query(sqlstr)
    except Exception as e:
        st.error(f"SQL Error: {str(e)}")
        return pd.DataFrame()


def _pick_chart_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Pick a likely categorical X and numeric Y for charts."""
    if df is None or df.empty:
        return None, None

    cat_candidates = [c for c in df.columns if str(c).upper() in ("DEALER_NAME", "VENDOR_NAME", "CATEGORY", "STATUS")]
    if not cat_candidates:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
                cat_candidates.append(c)
                break

    num_candidates = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_candidates.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().sum() > max(3, len(df) // 3):
                num_candidates.append(c)

    x = cat_candidates[0] if cat_candidates else None
    y = num_candidates[0] if num_candidates else None
    if x == y:
        y = num_candidates[1] if len(num_candidates) > 1 else None
    return x, y


def _generate_prescriptive_from_data(content_blocks, run_df_fn) -> Optional[str]:
    """Rule-based fallback prescriptive actions."""
    return (
        "<ul>"
        "<li>Filter to dealers below median performance and compare against top quartile.</li>"
        "<li>Investigate the largest deltas by category and prioritize the biggest contributors.</li>"
        "<li>Set an owner + due date for each action and track improvement weekly.</li>"
        "</ul>"
    )


def _has_comparison_columns(df: pd.DataFrame):
    """
    Check if DataFrame has period comparison columns (current vs previous).
    Returns: (cat_col, curr_col, prev_col, curr_label, prev_label) or (None, None, None, None, None)
    """
    if df is None or df.empty:
        return None, None, None, None, None

    upper = {str(c).upper(): c for c in df.columns}

    curr_patterns = ["CURRENT", "THIS_PERIOD", "CURRENT_PERIOD"]
    prev_patterns = ["PREVIOUS", "PRIOR", "PRIOR_PERIOD", "LAST_PERIOD"]

    curr_col = None
    prev_col = None

    for pattern in curr_patterns:
        curr_col = next((upper[k] for k in upper if k.startswith(pattern) or pattern in k), None)
        if curr_col:
            break

    for pattern in prev_patterns:
        prev_col = next((upper[k] for k in upper if k.startswith(pattern) or pattern in k), None)
        if prev_col:
            break

    if not (curr_col and prev_col):
        return None, None, None, None, None

    cat_col = None
    for col in df.columns:
        u = str(col).upper()
        if u in ("CURRENT_PERIOD", "PRIOR_PERIOD", str(curr_col).upper(), str(prev_col).upper()):
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            cat_col = col
            break

    return cat_col, curr_col, prev_col, "Current Period", "Previous Period"


def alt_bar_comparison(
    df: pd.DataFrame,
    cat_col,
    curr_col,
    prev_col,
    curr_label="Current",
    prev_label="Previous",
    title="Comparison",
    height=300
):
    """Create an Altair bar chart comparing two periods side-by-side."""
    try:
        import altair as alt

        if df is None or df.empty or not cat_col or not curr_col or not prev_col:
            return

        tidy = df[[cat_col, curr_col, prev_col]].copy()
        tidy[curr_col] = pd.to_numeric(tidy[curr_col], errors="coerce").fillna(0)
        tidy[prev_col] = pd.to_numeric(tidy[prev_col], errors="coerce").fillna(0)

        df_melted = tidy.melt(
            id_vars=[cat_col],
            value_vars=[curr_col, prev_col],
            var_name="Period",
            value_name="Value"
        )
        df_melted["Period"] = df_melted["Period"].map({curr_col: curr_label, prev_col: prev_label})

        chart = alt.Chart(df_melted).mark_bar().encode(
            x=alt.X(f"{cat_col}:N", sort="-y"),
            y=alt.Y("Value:Q"),
            color=alt.Color("Period:N", scale=alt.Scale(scheme="set2")),
            tooltip=[alt.Tooltip(f"{cat_col}:N"), "Period:N", alt.Tooltip("Value:Q", format=",.2f")]
        ).properties(height=height, title=title).interactive()

        st.altair_chart(chart, width='stretch')
    except Exception:
        return


def alt_bar(df: pd.DataFrame, x: str, y: str, color: str = "#5046e5", height: int = 300, horizontal: bool = False):
    """Create a simple Altair bar chart."""
    try:
        import altair as alt

        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return

        df_chart = df.copy()
        df_chart[y] = pd.to_numeric(df_chart[y], errors="coerce").fillna(0)

        if horizontal:
            chart = alt.Chart(df_chart).mark_bar().encode(
                y=alt.Y(f"{x}:N", sort="-x"),
                x=alt.X(f"{y}:Q"),
                color=alt.value(color),
                tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.2f")]
            ).properties(height=height).interactive()
        else:
            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X(f"{x}:N", sort="-y"),
                y=alt.Y(f"{y}:Q"),
                color=alt.value(color),
                tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.2f")]
            ).properties(height=height).interactive()

        st.altair_chart(chart, width='stretch')
    except Exception:
        return


def alt_bar_multi(df: pd.DataFrame, x: str, y: str, color: str = "#5046e5", height: int = 300, charts_per_row: int = 2):
    """
    Create 1-3 vertical bar charts showing performance tiers for large datasets.

    Args:
        df: DataFrame to chart
        x: Column for x-axis (typically dealer_name)
        y: Column for y-axis (metric)
        color: Bar color
        height: Height of each chart
        charts_per_row: Number of charts per row (2-3 recommended)
    """
    try:
        import altair as alt

        if df is None or df.empty or x not in df.columns or y not in df.columns:
            return

        df_chart = df.copy()
        df_chart[y] = pd.to_numeric(df_chart[y], errors="coerce").fillna(0)

        df_chart = df_chart.sort_values(by=y, ascending=False).reset_index(drop=True)

        num_items = len(df_chart)

        if num_items <= 15:
            num_charts = 1
        elif num_items <= 30:
            num_charts = 2
        else:
            num_charts = 3

        if num_charts == 1:
            chart = alt.Chart(df_chart).mark_bar().encode(
                x=alt.X(f"{x}:N", sort="-y"),
                y=alt.Y(f"{y}:Q"),
                color=alt.value(color),
                tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.2f")]
            ).properties(height=height, width=400).interactive()
            st.altair_chart(chart, width='stretch')
            return

        items_per_chart = num_items // num_charts
        num_rows = (num_charts + charts_per_row - 1) // charts_per_row

        for row_idx in range(num_rows):
            charts_in_row = min(charts_per_row, num_charts - row_idx * charts_per_row)
            cols = st.columns(charts_in_row)

            for col_idx in range(charts_in_row):
                chart_idx = row_idx * charts_per_row + col_idx
                if chart_idx >= num_charts:
                    break

                start_idx = chart_idx * items_per_chart
                end_idx = (chart_idx + 1) * items_per_chart if chart_idx < num_charts - 1 else num_items
                chart_data = df_chart.iloc[start_idx:end_idx]

                if chart_idx == 0:
                    tier_label = "Top Performers"
                elif chart_idx == 1:
                    tier_label = "Mid-Tier"
                else:
                    tier_label = "Bottom Tier"

                with cols[col_idx]:
                    st.markdown(f"**{tier_label}**", help=f"{len(chart_data)} dealers")
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X(f"{x}:N", sort="-y"),
                        y=alt.Y(f"{y}:Q"),
                        color=alt.value(color),
                        tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.2f")]
                    ).properties(height=height, width=250).interactive()
                    st.altair_chart(chart, width='stretch')

    except Exception as e:
        print(f"[CHART ERROR] {str(e)[:100]}")
        return


# ============================================================================
# QUICK ANALYSIS GENERATORS - For Genie Page
# ============================================================================
# _norm, _has_specific_threshold, route_verified_query, route_verified_query_smart,
# _route_verified_query_legacy, add_dealer_filter_to_sql, extract_first_select_sql,
# _build_minimal_schema, is_sql_obviously_bad, compile_check_sql
# → all in ai_service.py (imported via `from ai_service import *`)


def generate_sql_with_bedrock(bedrock_model: str, question: str, yaml_content: str) -> Tuple[str, bool, Optional[str]]:
    # Migration note: generate_sql_with_cortex(session, cortex_model, ...) → generate_sql_with_bedrock(bedrock_model, ...)
    # session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?,?)") → bedrock_complete()
    # compile_check_sql(session, sql) → compile_check_sql(sql)
    """Returns (sql, ok, error_message). ok means compile check passed."""
    import time as time_module

    t0 = time_module.time()
    model = yaml.safe_load(yaml_content) if yaml_content else {}

    t1 = time_module.time()
    tables_with_columns = _build_minimal_schema(question, model)
    t2 = time_module.time()
    print(f"[TIMING] Schema building: {t2-t1:.2f}s")

    prompt = f"""You are an AWS Athena SQL expert for dealer analytics. You MUST generate EXACT, VALID SQL.

Question: {question}

==============================================================================
CRITICAL RULES - FOLLOW THESE EXACTLY OR SQL WILL FAIL
==============================================================================

1. TABLE NAMES (for FROM/JOIN):
   - VW_AVERAGE_REPAIR_TURNAROUND_TIME (alias: t)
   - VW_DEALER_CONTRIBUTION_MARGIN (alias: cm)
   - VW_GROSS_PROFIT_MARGIN (alias: gpm or m)
   - VW_CASH_CONVERSION_CYCLE (alias: ccc or c)
   - VW_ORDER_LEAD_TIME (alias: lead or l)
   - VW_BACKORDER_INCIDENCE (alias: b)
   - VW_STOCK_AVAILABILITY_DEALER (alias: s)
   - VW_DEALER_REVENUE_GROWTH (alias: rg)
   - VW_SALES_VOLUME (alias: sv)

2. EXACT COLUMN NAMES - USE THESE EXACTLY:
   VW_AVERAGE_REPAIR_TURNAROUND_TIME:
   ✓ DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, AVG_TURNAROUND_HOURS
   ✗ DO NOT use: AVG_HOURS (WRONG), TURNAROUND_TIME (WRONG)

   VW_DEALER_CONTRIBUTION_MARGIN:
   ✓ DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, CONTRIBUTION_MARGIN_PCT
   ✗ DO NOT use: MARGIN (WRONG), CONTRIB_MARGIN (WRONG)

   VW_GROSS_PROFIT_MARGIN:
   ✓ DEALER_NAME, GROSS_PROFIT_MARGIN_PCT, TOTAL_REVENUE, PERIOD_YEAR, PERIOD_MONTH
   ✗ DO NOT use: MARGIN_PCT (WRONG), GPM (WRONG - column name, not abbreviation)

   VW_CASH_CONVERSION_CYCLE:
   ✓ DEALER_NAME, DSO, DIO, DPO, CCC, PERIOD_YEAR, PERIOD_MONTH
   ✗ DO NOT use: CASH_CYCLE (WRONG), DAYS_OUTSTANDING (WRONG)

   VW_BACKORDER_INCIDENCE:
   ✓ DEALER_NAME, BACKORDER_INCIDENCE_PCT, PERIOD_YEAR, PERIOD_MONTH
   ✗ DO NOT use: BACKORDER_RATE (WRONG), BACKORDER_PERCENT (WRONG)

   VW_ORDER_LEAD_TIME:
   ✓ DEALER_NAME, AVG_ORDER_LEAD_TIME_DAYS, PERIOD_YEAR, PERIOD_MONTH
   ✗ DO NOT use: LEAD_TIME_DAYS (WRONG)

3. ALIAS RULES - MUST FOLLOW:
   ✓ CORRECT: FROM table_name AS t, JOIN table2 AS t2 ON t.DEALER_NAME = t2.DEALER_NAME
   ✗ WRONG: FROM table_name, JOIN table2 ON DEALER_NAME = DEALER_NAME

4. JOIN RULE - ALWAYS USE EXPLICIT QUALIFIED COLUMNS:
   ✓ ON t.DEALER_NAME = s.DEALER_NAME
   ✗ ON DEALER_NAME = DEALER_NAME (SQL WILL FAIL - ambiguous column)

5. SELECT RULE - QUALIFY ALL COLUMNS:
   ✓ SELECT t.DEALER_NAME, t.AVG_TURNAROUND_HOURS, cm.CONTRIBUTION_MARGIN_PCT
   ✗ SELECT DEALER_NAME, AVG_TURNAROUND_HOURS, CONTRIBUTION_MARGIN_PCT

6. MULTI-TABLE JOINS - USE CTE PATTERN:
   ✓ WITH service_agg AS (SELECT DEALER_NAME, AVG(AVG_TURNAROUND_HOURS) as avg_hours
                           FROM VW_AVERAGE_REPAIR_TURNAROUND_TIME GROUP BY DEALER_NAME),
        margin_agg AS (SELECT DEALER_NAME, AVG(CONTRIBUTION_MARGIN_PCT) as avg_margin
                       FROM VW_DEALER_CONTRIBUTION_MARGIN GROUP BY DEALER_NAME)
      SELECT s.DEALER_NAME, s.avg_hours, m.avg_margin
      FROM service_agg s LEFT JOIN margin_agg m ON s.DEALER_NAME = m.DEALER_NAME

⚠️ CRITICAL: ALWAYS USE TABLE ALIASES, EVEN FOR SINGLE TABLES

EXAMPLE - SINGLE TABLE (MUST USE ALIAS):
SELECT
  t.DEALER_NAME,
  t.AVG_TURNAROUND_HOURS,
  t.PERIOD_YEAR
FROM VW_AVERAGE_REPAIR_TURNAROUND_TIME t
ORDER BY t.DEALER_NAME
LIMIT 100;

==============================================================================
AVAILABLE TABLES AND COLUMNS:
{tables_with_columns}

Return ONLY executable Athena SQL. No markdown. No explanations. No comments. Just valid SQL.
DO NOT ABBREVIATE COLUMN NAMES - USE EXACT NAMES SHOWN ABOVE.
REMEMBER: Every SELECT must use qualified table.column format with aliases.""".strip()

    try:
        print(f"[TIMING] About to call Bedrock API...")
        t3 = time_module.time()
        # Migration note: session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?,?)") → bedrock_complete()
        raw = bedrock_complete(prompt, model_id=bedrock_model) or ""
        t4 = time_module.time()
        print(f"[TIMING] Bedrock API call: {t4-t3:.2f}s")

        sql = extract_first_select_sql(raw)
        print(f"[TIMING] SQL extracted: {len(sql)} chars")
        print(f"[SQL DEBUG] Generated SQL:\n{sql}")

        bad, reason = is_sql_obviously_bad(sql)
        if bad:
            print(f"[TIMING] SQL rejected: {reason}")
            return sql, False, f"Rejected: {reason}"

        print(f"[TIMING] About to compile check...")
        t5 = time_module.time()
        # Migration note: compile_check_sql(session, sql) → compile_check_sql(sql)
        ok, err = compile_check_sql(sql)
        t6 = time_module.time()
        print(f"[TIMING] Compile check: {t6-t5:.2f}s - Result: {'OK' if ok else 'FAILED'}")

        if ok:
            t_total = time_module.time() - t0
            print(f"[TIMING] TOTAL BEDROCK SQL GEN: {t_total:.2f}s")
            return sql, True, None

        return sql, False, f"Compile failed: {err[:100]}"

    except Exception as e:
        t_total = time_module.time() - t0
        print(f"[TIMING] EXCEPTION after {t_total:.2f}s: {str(e)[:100]}")
        return "", False, f"Generation failed: {str(e)[:100]}"


def run_quick_analysis(analysis_key: str):
    # Migration note: get_snowflake_connection() + session.sql(CORTEX.COMPLETE) → bedrock_complete()
    """
    Generate quick analysis data using verified queries from YAML.
    Returns structured data with "layout": "quick" including Genie text summary.
    """
    import time as time_module
    t_start = time_module.time()

    QUERY_MAPPING = {
        "cash_cycle": "check_dealer_ccc",
        "service_turnaround": "dealer_service_efficiency",
        "inventory_health": "dealer_inventory_health",
        "order_fulfillment": "lead_time_analysis_2026",
    }

    model = load_yaml_model("dealer_model.yml")

    if not model or "verified_queries" not in model:
        return {"layout": "cortex", "error": "Could not load semantic model or no verified_queries found in YAML"}

    query_name = QUERY_MAPPING.get(analysis_key)
    if not query_name:
        return {"layout": "cortex", "message": {"content": [{"type": "text", "text": f"Analysis type '{analysis_key}' not recognized."}]}}

    verified_query = next((vq for vq in model.get("verified_queries", []) if vq.get("name") == query_name), None)
    if not verified_query or not verified_query.get("sql"):
        return {"layout": "cortex", "error": f"Verified query '{query_name}' not found or missing SQL"}

    try:
        sql = verified_query["sql"].strip()
        if "LIMIT" not in sql.upper():
            sql = sql.rstrip(";") + "\nLIMIT 500"
        vendors_df = run_df(sql)

        if vendors_df.empty:
            return {"layout": "cortex", "error": "Query executed but returned no results"}

        metrics = {}
        upper_cols = {str(c).upper(): c for c in vendors_df.columns}

        if analysis_key == "cash_cycle":
            for k in ("CCC", "DSO", "DIO", "DPO"):
                col = upper_cols.get(k)
                if col:
                    metrics[f"avg_{k.lower()}_days"] = int(safe_number(vendors_df[col].mean(), 0))

        elif analysis_key == "service_turnaround":
            turnaround = upper_cols.get("AVG_TURNAROUND_HOURS")
            lead_time = upper_cols.get("AVG_ORDER_LEAD_TIME_DAYS")
            stock = upper_cols.get("STOCK_AVAILABILITY_PCT")
            if turnaround:
                metrics["avg_turnaround_hours"] = int(safe_number(vendors_df[turnaround].mean(), 0))
            if lead_time:
                metrics["avg_lead_time_days"] = int(safe_number(vendors_df[lead_time].mean(), 0))
            if stock:
                metrics["avg_stock_availability_pct"] = int(safe_number(vendors_df[stock].mean(), 0))

        elif analysis_key == "inventory_health":
            stock = upper_cols.get("STOCK_AVAILABILITY_PCT")
            back = upper_cols.get("BACKORDER_INCIDENCE_PCT")
            if stock:
                metrics["avg_stock_availability_pct"] = int(safe_number(vendors_df[stock].mean(), 0))
            if back:
                metrics["avg_backorder_incidence_pct"] = int(safe_number(vendors_df[back].mean(), 0))

        elif analysis_key == "order_fulfillment":
            lead = upper_cols.get("AVG_LEAD_TIME_DAYS")
            if lead:
                metrics["avg_lead_time_days"] = int(safe_number(vendors_df[lead].mean(), 0))

        text_summary = ""
        try:
            tile_question_map = {
                "cash_cycle": "Show cash conversion cycle analysis",
                "service_turnaround": "What is the average repair turnaround time?",
                "inventory_health": "Show inventory metrics and stock availability by dealer.",
                "order_fulfillment": "What is the average order lead time?",
            }
            query_text = tile_question_map.get(analysis_key, "Provide analysis")

            data_summary = f"Query returned {len(vendors_df)} rows:\n{vendors_df.head(5).to_string()}"
            if len(vendors_df) > 5:
                data_summary += f"\n... and {len(vendors_df) - 5} more rows"

            cortex_prompt = f"""Analyze this dealer question based on the data:

Question: {query_text.strip()}
Data: {data_summary}

Respond with THREE sections (ONLY these headers and content, no extra text):

**Descriptive** - What the data shows (2-3 sentences with key numbers)

**Prescriptive** - Recommendations (5-7 bullet points starting with •)

**Predictive** - Expected Impact in 12-24 months (1-2 sentences with outcomes)"""

            t_text = time_module.time()
            # Migration note: get_snowflake_connection() + session.sql(CORTEX.COMPLETE) → bedrock_complete()
            text_summary = bedrock_complete(cortex_prompt, model_id=get_config()["bedrock"]["primary_model"]) or ""
            print(f"[TEXT GEN] Generated text summary in {time_module.time() - t_text:.2f}s")
        except Exception as e:
            print(f"[TEXT GEN] Failed to generate text: {str(e)[:100]}")
            text_summary = ""

        response_time_ms = (time_module.time() - t_start) * 1000
        return {"layout": "quick", "metrics": metrics, "vendors_df": vendors_df, "sql": sql, "text_summary": text_summary, "response_time_ms": response_time_ms}

    except Exception as e:
        return {"layout": "cortex", "error": f"Could not generate quick analysis: {str(e)}"}


# _genie_debug, call_cortex_analyst → in ai_service.py as _genie_debug, call_bedrock_analyst
# extract_dealer_name_from_question, extract_dealer_name_from_context → in ai_service.py


# ============================================================================
# PREDICTIVE ANALYSIS HELPER FUNCTIONS
# ============================================================================

def generate_dealer_predictions(dealer_name: str) -> Dict:
    # Migration note: session parameter removed; DealerPerformanceForecaster/AnomalyDetector use athena_query() internally
    """Generate comprehensive predictions for a dealer."""
    try:
        forecaster = DealerPerformanceForecaster()
        detector = AnomalyDetector()

        predictions = {
            "dealer": dealer_name,
            "forecast": forecaster.forecast_revenue(dealer_name, weeks=8),
            "anomalies": detector.detect_dealer_anomalies(dealer_name),
            "timestamp": datetime.now().isoformat()
        }

        return predictions
    except Exception as e:
        print(f"[PREDICTIONS] Error generating predictions: {str(e)}")
        return {"error": str(e)}


def render_prediction_sidebar():
    # Migration note: session parameter removed
    """Render predictive analysis sidebar controls."""
    with st.sidebar:
        st.markdown("### Predictive Analysis")

        forecast_weeks = st.slider(
            "Forecast Horizon (weeks)",
            1, 26, 8,
            help="Number of weeks to forecast dealer performance"
        )

        show_anomalies = st.checkbox("Detect Anomalies", value=True)
        show_scenarios = st.checkbox("What-If Scenarios", value=False)

        if st.button("Run Predictions", width='stretch', type="primary"):
            st.session_state['run_predictions'] = True
            st.session_state['forecast_weeks'] = forecast_weeks

        st.divider()

        cache_stats = st.session_state.genie_cache.stats()
        with st.expander("⚙️ Cache Stats"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cached", f"{cache_stats['size']}/{cache_stats['max_size']}")
            with col2:
                st.metric("Hit Rate", f"{cache_stats['hit_rate']:.0%}")

            if st.button("🔄 Clear Cache", width='stretch'):
                st.session_state.genie_cache.cache.clear()
                st.success("Cache cleared!")


def render_prediction_dashboard(dealer_name: str, forecast_weeks: int = 8):
    # Migration note: session parameter removed; DealerPerformanceForecaster uses athena_query() internally
    """Render prediction dashboard for a dealer."""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Revenue Forecast")

        forecaster = DealerPerformanceForecaster()
        forecast = forecaster.forecast_revenue(dealer_name, weeks=forecast_weeks)

        if forecast.get("success"):
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                trend_emoji = "▲" if forecast.get('trend') == 'improving' else "▼"
                st.metric(f"{trend_emoji} Trend", forecast.get('trend', 'unknown').upper())
            with metric_col2:
                change = forecast.get('change_percent', 0)
                st.metric("Expected Change", f"{change:+.1f}%")
            with metric_col3:
                st.metric("Confidence", forecast.get('confidence', 'unknown').upper())

            # Chart data
            try:
                import altair as alt

                forecast_df = pd.DataFrame({
                    'week': range(1, len(forecast['forecast_values']) + 1),
                    'forecast': forecast['forecast_values'],
                })

                chart = alt.Chart(forecast_df).mark_line(point=True, color='#5046e5').encode(
                    x=alt.X('week:O', title='Week'),
                    y=alt.Y('forecast:Q', title='Revenue ($)'),
                    tooltip=['week', 'forecast']
                ).properties(height=250)

                st.altair_chart(chart, width='stretch')
            except Exception as e:
                st.info(f"Chart rendering: {str(e)[:50]}")
        else:
            st.warning(forecast.get('error', 'Forecast unavailable'))

    with col2:
        st.markdown("### Anomalies & Risk")

        # Migration note: AnomalyDetector(session) → AnomalyDetector() (session removed)
        detector = AnomalyDetector()
        anomalies = detector.detect_dealer_anomalies(dealer_name)

        if anomalies.get("anomalies"):
            risk_level = anomalies.get("risk_level", "unknown")
            risk_color = "#dc2626" if risk_level == "high" else "#f59e0b" if risk_level == "medium" else "#10b981"

            st.markdown(f"""
            <div style="padding:12px;background:{risk_color}22;border-left:4px solid {risk_color};border-radius:6px;">
                <strong>Risk Level:</strong> {risk_level.upper()}<br>
                <strong>Anomalies Found:</strong> {anomalies.get("anomalies_count", 0)} / {anomalies.get("total_periods", 0)}
            </div>
            """, unsafe_allow_html=True)

            for anomaly in anomalies['anomalies'][:3]:
                with st.expander(f"Period {anomaly.get('period')} ({anomaly.get('severity').upper()})"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write(f"**Revenue:** ${anomaly.get('revenue', 0):,.0f}")
                        st.write(f"**Margin:** {anomaly.get('margin', 0):.1f}%")
                    with col_b:
                        st.write(f"**Stock:** {anomaly.get('stock', 0):.1f}%")
                        st.write(f"**Backorder:** {anomaly.get('backorder', 0):.1f}%")
        else:
            st.info("✅ No anomalies detected")


# generate_forecast_prediction_text → in ai_service.py


# ============================================================================
# GENIE PAGE - DEALER ASSISTANT (render)
# ============================================================================
def render_genie_page():
    # Migration note: session parameter removed; GenieLongTermMemory/DynamoChatPersistence/call_bedrock_analyst use boto3 internally
    """Render Genie page - Dealer Genie assistant with predictive analytics."""

    defaults = {
        "selected_analysis":    None,
        "show_analysis":        False,
        "analyst_response":     None,
        "genie_messages":       [],
        "saved_insights":       [],
        "recent_analyses":      [],
        "sidebar_expanded":     True,
        "genie_input_version":  0,
        "last_custom_query":    "",
        "run_predictions":      False,
        "forecast_weeks":       8,
        "context_turns":        6,
        "genie_memory":         None,
        "genie_memory_built":   False,
        "genie_session_id":         None,
        "chat_persistence":         None,
        "chat_persist_init":        False,
        "chat_turn_index":          0,
        "restore_offered":          False,
        "restore_dismissed":        False,
        "_all_sessions_cache":      [],
        "show_chats_panel":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if (not st.session_state.get("genie_memory_built")
            and st.session_state.get("genie_cache") is not None):
        try:
            # Migration note: GenieLongTermMemory(session) → GenieLongTermMemory()
            st.session_state.genie_memory = GenieLongTermMemory()
        except Exception as e:
            print(f"[MEMORY INIT] {str(e)[:100]}", file=__import__('sys').stderr)
            st.session_state.genie_memory = None
        finally:
            st.session_state.genie_memory_built = True

    if not st.session_state.get("chat_persist_init"):
        try:
            # Migration note: GenieChatPersistence(session) → DynamoChatPersistence()
            cp = DynamoChatPersistence()
            st.session_state.chat_persistence  = cp
            cp.purge_old_sessions(keep_days=3)
        except Exception as e:
            print(f"[CHAT PERSIST INIT] {str(e)[:100]}", file=__import__('sys').stderr)
            st.session_state.chat_persistence  = None
        finally:
            st.session_state.chat_persist_init = True

    if not st.session_state.get("genie_session_id"):
        import uuid as _uuid
        st.session_state.genie_session_id = str(_uuid.uuid4())
        st.session_state.genie_session_label = (
            "Chat on " + datetime.now().strftime("%b %d %H:%M")
        )

    st.markdown("""
    <style>
    .streamlit-expanderHeader svg { display: none !important; }
    .streamlit-expanderHeader { padding: 0.75rem 1rem !important; }
    </style>
    """, unsafe_allow_html=True)

    _CASH_SVG  = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><rect x="2" y="5" width="20" height="14" rx="2" stroke="white" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="2.5" stroke="white" stroke-width="1.5" fill="none"/><path d="M7 12h-2" stroke="white" stroke-width="1.5" stroke-linecap="round"/><path d="M19 12h2" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg>"""
    _BOX_SVG   = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M21 16V8c0-1-1-2-2-2H5c-1 0-2 1-2 2v8c0 1 1 1 2 2h14c1 0 2-1 2-2z" stroke="white" stroke-width="1.5" fill="none"/><path d="M3.27 6.96L12 12.88l8.73-5.92" stroke="white" stroke-width="1.5" fill="none"/></svg>"""
    _CLOCK_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="9" stroke="white" stroke-width="1.5" fill="none"/><path d="M12 6v6l4 2" stroke="white" stroke-width="1.5" stroke-linecap="round"/></svg>"""
    _TRUCK_SVG = """<svg width="24" height="24" viewBox="0 0 24 24" fill="none"><path d="M1 6h14v10H1z" stroke="white" stroke-width="1.5" fill="none"/><path d="M15 8h6l2 4v4h-8v-8z" stroke="white" stroke-width="1.5" fill="none"/><circle cx="5" cy="19" r="2" stroke="white" stroke-width="1.5" fill="none"/><circle cx="18" cy="19" r="2" stroke="white" stroke-width="1.5" fill="none"/></svg>"""

    QUICK_ANALYSES = {
        "cash_cycle":         {"title": "Cash Conversion Cycle",       "icon_svg": _CASH_SVG,  "desc": "Track CCC trends and inventory efficiency",                   "question": "Show cash conversion cycle analysis"},
        "service_turnaround": {"title": "Service Turnaround Analysis", "icon_svg": _CLOCK_SVG, "desc": "Review repair turnaround times and service efficiency",        "question": "What is the average repair turnaround time?"},
        "inventory_health":   {"title": "Inventory & Stock Health",    "icon_svg": _BOX_SVG,   "desc": "Monitor stock availability, backorders, and inventory", "question": "Show inventory metrics and stock availability by dealer."},
        "order_fulfillment":  {"title": "Order Lead Time Analysis",    "icon_svg": _TRUCK_SVG, "desc": "Analyze delivery performance and lead time trends",            "question": "What is the average order lead time?"},
    }

    def parse_analysis_sections(text: str):
        import re as _re2
        di = _re2.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})descriptive(?:\*{0,2})(?:\s*[-:])?',  text, _re2.IGNORECASE)
        pi = _re2.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})prescriptive(?:\*{0,2})(?:\s*[-:])?', text, _re2.IGNORECASE)
        ri = _re2.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})predictive(?:\*{0,2})(?:\s*[-:])?',   text, _re2.IGNORECASE)

        def _ext(sm, *others):
            if not sm: return ""
            s = sm.end(); e = len(text)
            for o in others:
                if o and o.start() > sm.start():
                    e = min(e, o.start())
            return text[s:e].strip().lstrip("*").strip()

        return _ext(di, pi, ri), _ext(pi, ri, di), _ext(ri, di, pi)

    def process_genie_query(query: str, analysis_type: str = "custom") -> Dict[str, Any]:
        st.session_state.genie_messages.append({
            "role": "user", "content": query,
            "timestamp": pd.Timestamp.now(), "response": None,
        })

        turns   = st.session_state.get("context_turns", 6)
        history = st.session_state.genie_messages[-(turns * 2):]

        # Migration note: call_cortex_analyst(session, query, history=history) → call_bedrock_analyst(query, history=history)
        response = call_bedrock_analyst(query, history=history)

        assistant_text = ""
        if isinstance(response, dict):
            df = response.get("vendors_df")
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols         = list(df.columns)
                cols_preview = ", ".join(cols[:4]) + ("..." if len(cols) > 4 else "")
                assistant_text = f"Returned {len(df):,} rows · columns: {cols_preview}"
            elif response.get("error"):
                assistant_text = str(response["error"])
            else:
                assistant_text = "Analysis complete."

        st.session_state.genie_messages.append({
            "role":      "assistant",
            "content":   assistant_text,
            "timestamp": pd.Timestamp.now(),
            "response":  response,
        })

        st.session_state.genie_messages = st.session_state.genie_messages[-40:]

        st.session_state.recent_analyses.insert(0, {
            "query": query, "type": analysis_type,
            "timestamp": pd.Timestamp.now(), "response": response,
        })
        st.session_state.recent_analyses = st.session_state.recent_analyses[:10]

        cp  = st.session_state.get("chat_persistence")
        sid = st.session_state.get("genie_session_id", "")
        lbl = st.session_state.get("genie_session_label", "")
        if cp and sid:
            try:
                ti = st.session_state.get("chat_turn_index", 0)

                cp.save_turn(
                    session_id=sid, turn_index=ti,
                    role="user", content=query,
                    sql_used="", source="user_input",
                    session_label=lbl,
                )
                ti += 1

                sql_used = ""
                src      = ""
                if isinstance(response, dict):
                    sql_used = (response.get("sql") or "")[:2000]
                    src      = response.get("source", "")
                cp.save_turn(
                    session_id=sid, turn_index=ti,
                    role="assistant", content=assistant_text,
                    sql_used=sql_used, source=src,
                    session_label=lbl,
                )
                ti += 1

                st.session_state.chat_turn_index = ti
            except Exception as e:
                print(f"[CHAT PERSIST] save failed: {str(e)[:100]}",
                      file=__import__('sys').stderr)

        msg_count  = len(st.session_state.get("genie_messages", []))
        memory_obj = st.session_state.get("genie_memory")
        if memory_obj and msg_count % 10 == 0:
            try:
                memory_obj.refresh()
            except Exception as e:
                print(f"[MEMORY REFRESH] {str(e)[:80]}", file=__import__('sys').stderr)

        return response

    def render_result_panel(response: dict, msg_idx: int):
        if not response:
            return
        if "error" in response:
            st.error(f"**Error:** {response['error']}")
            return

        if response.get("layout") == "quick":
            text_summary = response.get("text_summary", "").strip()
            vendors_df   = response.get("vendors_df")
            sql_used     = response.get("sql", "")
        elif "message" in response:
            blocks       = response.get("message", {}).get("content", [])
            text_summary = "\n\n".join([(b.get("text") or "") for b in blocks
                                        if b.get("type") == "text"]).strip()
            rd           = response.get("related_data") or {}
            vendors_df   = rd.get("df")
            sql_used     = rd.get("sql", "")
        else:
            return

        if not text_summary and vendors_df is not None and not vendors_df.empty:
            try:
                question_text = st.session_state.get("last_custom_query") or "Dealer analysis"
                data_summary  = f"Query returned {len(vendors_df)} rows:\n{vendors_df.head(5).to_string()}"
                if len(vendors_df) > 5:
                    data_summary += f"\n... and {len(vendors_df) - 5} more rows"
                cortex_prompt = f"""Analyze this dealer question based on the data:
Question: {question_text.strip()}
Data: {data_summary}
Respond with THREE sections (ONLY these headers, no extra text):
**Descriptive** - What the data shows (2-3 sentences with key numbers)
**Prescriptive** - Recommendations (5-7 bullet points starting with •)
**Predictive** - Expected Impact in 12-24 months (1-2 sentences)"""
                # Migration note: get_snowflake_connection() + session.sql(CORTEX.COMPLETE) → bedrock_complete()
                text_summary = bedrock_complete(cortex_prompt, model_id=get_config()["bedrock"]["primary_model"]) or ""
                response["text_summary"] = text_summary
            except Exception as e:
                print(f"[TEXT GEN] {str(e)[:100]}")

        rt  = response.get("response_time_ms", 0)
        cft = response.get("cache_fetch_time_ms", 0)
        if cft and rt and cft < rt:   badge = f"Cached · {cft/1000:.2f}s (orig {rt/1000:.2f}s)"
        elif rt and rt < 500:         badge = f"Cached · {rt/1000:.2f}s"
        elif rt and rt < 2000:        badge = f"Fast · {rt/1000:.2f}s"
        elif rt:                      badge = f"{rt/1000:.2f}s"
        else:                         badge = ""
        if badge:
            st.markdown(f"<div style='font-size:11px;color:#64748b;margin-bottom:8px;'>{badge}</div>",
                        unsafe_allow_html=True)

        desc_s, pres_s, pred_s = parse_analysis_sections(text_summary)

        if desc_s:
            st.markdown(f"""
            <div style="padding:14px;background:#e0f2fe;border-radius:10px;
                 border-left:4px solid #0284c7;margin-bottom:12px;">
                <div style="font-size:12px;font-weight:800;color:#0369a1;margin-bottom:6px;">
                    Descriptive — What the data shows</div>
                <div style="color:#0f172a;font-size:14px;line-height:1.7;">
                    {html.escape(desc_s).replace(chr(10), '<br/>')}
                </div>
            </div>""", unsafe_allow_html=True)

        if pres_s:
            with st.expander("Prescriptive — Recommendations & Actions", expanded=False):
                bullets = [
                    f"<li style='margin-bottom:10px;line-height:1.6;'>{html.escape(l)}</li>"
                    for l in [line.strip().lstrip("•").lstrip("*").strip()
                               for line in pres_s.split("\n")] if l
                ]
                if bullets:
                    st.markdown(f"<ul style='margin:0;padding-left:22px;font-size:14px;"
                                f"color:#0f172a;'>{''.join(bullets)}</ul>", unsafe_allow_html=True)

        if pred_s:
            with st.expander("Predictive — Expected Impact (12–24 months)", expanded=False):
                st.markdown(f"<div style='font-size:14px;color:#0f172a;line-height:1.7;'>"
                            f"{html.escape(pred_s).replace(chr(10), '<br/>')}</div>",
                            unsafe_allow_html=True)

        if vendors_df is not None and not vendors_df.empty:
            x_col, y_col = _pick_chart_columns(vendors_df)
            if x_col and y_col:
                st.markdown("**Chart Analysis**")
                try:
                    alt_bar_multi(vendors_df, x=x_col, y=y_col, height=300, charts_per_row=2)
                except Exception as e:
                    print(f"[CHART] {e}")
            st.markdown("**Data Results**")
            st.dataframe(vendors_df, use_container_width=True, height=280)
            st.download_button("Download CSV", vendors_df.to_csv(index=False),
                               "results.csv", key=f"dl_btn_{msg_idx}")
        elif not desc_s:
            # No data + no analysis text → show a friendly "no data" card instead of a cold error
            user_q = st.session_state.get("last_custom_query", "").strip()
            st.markdown(f"""
            <div style="padding:18px 20px;background:#fef9f0;border-radius:12px;
                 border-left:4px solid #f59e0b;margin:8px 0;">
                <div style="font-size:15px;font-weight:700;color:#92400e;margin-bottom:6px;">
                    📭 No matching data found
                </div>
                <div style="font-size:13px;color:#78350f;line-height:1.7;">
                    I couldn't find data in our dealer database for <em>"{user_q or 'your question'}"</em>.<br/>
                    This might be because the data doesn't exist yet, filters returned no results,
                    or the question is outside the dealer analytics scope.
                </div>
                <div style="font-size:12px;color:#92400e;margin-top:10px;font-weight:600;">
                    💡 Try asking:
                </div>
                <ul style="font-size:12px;color:#78350f;margin:4px 0 0 16px;line-height:1.9;">
                    <li>Which dealers have the highest gross profit margin?</li>
                    <li>Show inventory metrics and stock availability by dealer</li>
                    <li>What is the cash conversion cycle trend?</li>
                    <li>Compare service turnaround time across dealers</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        if sql_used:
            with st.expander("Query used", expanded=False):
                st.code(sql_used.strip(), language="sql")

    def render_cache_stats():
        cache = st.session_state.get("genie_cache")
        if not cache: return
        # Migration note: cache.table_initialized → DynamoDB always ready; cache.session → DynamoDB client
        init_status = "✅ Ready"
        st.markdown(f"""<div style="padding:8px 12px;background:#f0f0f0;border-radius:6px;
             margin-bottom:12px;font-size:12px;">
            <strong>Cache Status:</strong> {init_status} |
            Session: ✅ Connected |
            User: {get_current_user()}</div>""", unsafe_allow_html=True)
        if st.button("Test DB Write", key="test_db_write"):
            # Migration note: cache.session.sql(INSERT INTO Snowflake) → DynamoDB test
            try:
                cache.set("_test_question_", {"layout": "test", "text_summary": "ok"})
                st.success("✅ DynamoDB write succeeded!")
            except Exception as e:
                st.error(f"❌ FAILED: {type(e).__name__}: {str(e)[:300]}")
        stats = cache.stats(); db_stats = stats.get('db_stats', {})
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Cached Questions", f"{stats.get('memory_cache_size',0)}/{stats.get('max_size',100)}")
        with c2: st.metric("Cache Hit Rate",   f"{stats.get('hit_rate',0):.1%}")
        with c3: st.metric("Active Users",     db_stats.get('unique_users', 0))
        with c4:
            avg_t = db_stats.get('avg_response_time_ms', 0)
            st.metric("Avg Response Time" if avg_t else "Total Queries (7d)",
                      f"{avg_t/1000:.2f}s" if avg_t else db_stats.get('total_queries', 0))
        st.markdown("---")
        st.markdown("**📊 Most Frequently Asked Questions (Last 7 days)**")
        popular_df = cache.get_popular_questions(limit=5, days=7)
        if not popular_df.empty:
            if 'LAST_ASKED' in popular_df.columns:
                popular_df['LAST_ASKED'] = pd.to_datetime(
                    popular_df['LAST_ASKED']).dt.strftime('%Y-%m-%d %H:%M')
            if 'AVG_RESPONSE_TIME_MS' in popular_df.columns:
                popular_df['AVG_RESPONSE_TIME_MS'] = popular_df['AVG_RESPONSE_TIME_MS'].apply(
                    lambda x: f"{x:.0f}ms")
            st.dataframe(popular_df[['QUESTION','FREQUENCY','AVG_RESPONSE_TIME_MS','LAST_ASKED']],
                         use_container_width=True, hide_index=True)
        else:
            st.info("No query history available yet.")

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE HEADER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="margin-bottom:8px;">
        <h1 style="font-size:28px;font-weight:900;color:#1a1a1a;margin:0 0 4px 0;">
            Welcome to Dealer Genie</h1>
        <p style="font-size:16px;color:#64748b;margin:0;">
            AI-powered analysis of dealer performance and KPIs</p>
    </div>""", unsafe_allow_html=True)

    ICON_BG="#5046e5"; LAVENDER="#e8e4f7"; CARD_BORDER="#e5e7eb"
    SELECTED_BORDER="#5046e5"; TEXT_TITLE="#1a1a1a"; TEXT_DESC="#64748b"

    tile_cols   = st.columns(4, gap="medium")
    clicked_key = None
    sel         = st.session_state.get("selected_analysis")
    show        = st.session_state.get("show_analysis", False)

    for idx, (key, analysis) in enumerate(QUICK_ANALYSES.items()):
        with tile_cols[idx]:
            with st.form(f"tile_{key}", border=False):
                selected = bool(show and sel == key)
                bg       = LAVENDER if selected else "#fff"
                border   = SELECTED_BORDER if selected else CARD_BORDER
                st.markdown(f"""
                <div style="background:{bg};border:1.5px solid {border};border-radius:8px;
                     padding:20px;box-shadow:0 2px 8px rgba(0,0,0,.04);min-height:160px;">
                    <div style="width:48px;height:48px;border-radius:12px;display:flex;
                         align-items:center;justify-content:center;
                         margin-bottom:14px;background:{ICON_BG};">
                        {analysis["icon_svg"]}</div>
                    <div style="font-size:16px;font-weight:800;color:{TEXT_TITLE};margin-bottom:6px;">
                        {analysis["title"]}</div>
                    <div style="font-size:13px;color:{TEXT_DESC};line-height:1.4;">
                        {analysis["desc"]}</div>
                </div>""", unsafe_allow_html=True)
                if st.form_submit_button("Ask Genie", width='stretch'):
                    clicked_key = key

    if clicked_key is not None:
        a = QUICK_ANALYSES[clicked_key]
        st.session_state.selected_analysis = clicked_key
        st.session_state.show_analysis     = True
        st.session_state.last_custom_query = a["question"]
        with st.spinner(f"Loading {a['title']}..."):
            response = process_genie_query(a["question"], analysis_type=clicked_key)
        st.session_state.analyst_response = response
        st.rerun()

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([0.35, 0.65], gap="medium", vertical_alignment="top")

    with left_col:

        with st.container(border=True):
            st.markdown("""
            <div style="font-size:16px;font-weight:800;color:#0f172a;margin-bottom:16px;">
                Analysis Library
            </div>""", unsafe_allow_html=True)

            with st.expander("Recent analysis", expanded=True):
                recent_list = st.session_state.get('recent_analyses', [])
                if recent_list:
                    for i, item in enumerate(recent_list[:5]):
                        q         = item.get("query", "")
                        q_display = (q[:50] + "...") if len(q) > 50 else q
                        if st.button(q_display, key=f"recent_{i}",
                                     use_container_width=True, type="secondary"):
                            st.session_state.selected_analysis = "custom"
                            st.session_state.last_custom_query = q
                            st.session_state.show_analysis     = True
                            st.session_state.analyst_response  = item.get("response")
                            st.session_state.genie_messages.append({
                                "role": "assistant", "content": f"Re-showing: {q_display}",
                                "timestamp": pd.Timestamp.now(), "response": item.get("response"),
                            })
                            st.rerun()
                else:
                    st.markdown("""
                    <div style="border:2px dashed #e2e8f0;border-radius:12px;
                         padding:16px 12px;text-align:center;">
                        <div style="font-size:13px;color:#94a3b8;">
                            Run analyses to see them here.
                        </div>
                    </div>""", unsafe_allow_html=True)

        with st.expander("Suggested questions", expanded=True):
            for i, suggestion in enumerate([
                "Show gross profit margins by dealer",
                "What is the cash conversion cycle trend?",
                "Compare service efficiency across dealers",
                "Which dealers have inventory issues?",
                "Show revenue growth year-over-year",
            ]):
                if st.button(suggestion, key=f"default_suggestion_{i}",
                             use_container_width=True, type="secondary"):
                    st.session_state.selected_analysis = "custom"
                    st.session_state.last_custom_query = suggestion
                    st.session_state.show_analysis     = True
                    with st.spinner("Analyzing..."):
                        response = process_genie_query(suggestion)
                    st.session_state.analyst_response = response
                    st.rerun()

        cache      = st.session_state.get("genie_cache")
        popular_df = cache.get_popular_questions(limit=5, days=7) if cache else pd.DataFrame()
        if not popular_df.empty:
            with st.expander("Most Frequently Asked (Last 7 days)", expanded=False):
                for i, row in popular_df.iterrows():
                    question  = row.get('QUESTION', '')
                    frequency = int(row.get('FREQUENCY', 0))
                    avg_time  = row.get('AVG_RESPONSE_TIME_MS', 0)
                    q_display = (question[:80] + "...") if len(question) > 80 else question
                    col_q, col_freq = st.columns([0.75, 0.25])
                    with col_q:
                        if st.button(q_display, key=f"faq_popular_{i}",
                                     use_container_width=True, type="secondary"):
                            st.session_state.selected_analysis = "custom"
                            st.session_state.last_custom_query = question
                            st.session_state.show_analysis     = True
                            with st.spinner("Analyzing..."):
                                response = process_genie_query(question)
                            st.session_state.analyst_response = response
                            st.rerun()
                    with col_freq:
                        st.markdown(
                            f"<div style='text-align:center;font-size:11px;color:#6b7280;'>"
                            f"<div style='font-weight:600;color:#111827;'>{frequency}x</div>"
                            f"<div>{avg_time:.0f}ms</div></div>",
                            unsafe_allow_html=True,
                        )

    with right_col:
        tab_analysis, tab_forecast = st.tabs(["Chat Analysis", "Forecast & Predictions"])

        with tab_analysis:

            st.markdown("""
            <style>
            .g-user { display:flex; justify-content:flex-end; margin:6px 0; }
            .g-user-inner {
                max-width:72%; background:#2563eb; color:#fff;
                padding:10px 14px; border-radius:16px;
                border-bottom-right-radius:4px; font-size:14px; line-height:1.5;
            }
            .g-user-lbl { font-size:11px; font-weight:700; opacity:.8; margin-bottom:3px; }
            .g-ai  { display:flex; justify-content:flex-start; margin:6px 0; }
            .g-ai-inner {
                max-width:72%; background:#f1f5f9; color:#0f172a;
                padding:10px 14px; border-radius:16px;
                border-bottom-left-radius:4px; font-size:14px; line-height:1.5;
            }
            .g-ai-lbl { font-size:11px; font-weight:700; color:#64748b; margin-bottom:3px; }
            .g-card {
                background:#f8fafc; border:1px solid #e2e8f0;
                border-radius:12px; padding:16px; margin:4px 0 18px 0;
            }
            </style>
            """, unsafe_allow_html=True)

            msg_count = len(st.session_state.get("genie_messages", []))

            def _build_chat_md() -> str:
                msgs  = st.session_state.get("genie_messages", [])
                label = st.session_state.get("genie_session_label",
                                              "Chat on " + datetime.now().strftime("%b %d %H:%M"))
                lines = [f"# Genie Chat — {label}", "",
                         f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}*", "", "---", ""]
                for m in msgs:
                    role    = m.get("role", "user")
                    content = m.get("content", "") or ""
                    if not content:
                        continue
                    prefix = "**You:** " if role == "user" else "**AI Assistant:** "
                    lines.append(prefix + content)
                    lines.append("")
                return "\n".join(lines)

            hdr_left, hdr_chats, hdr_mid, hdr_dl, hdr_right = st.columns(
                [2.4, 1, 1, 1, 1], gap="small"
            )

            with hdr_left:
                st.markdown(
                    '<div style="font-size:15px;font-weight:800;color:#0f172a;'
                    'padding-top:6px;">AI Assistant</div>',
                    unsafe_allow_html=True,
                )

            with hdr_chats:
                chats_clicked = st.button(
                    "Chats",
                    use_container_width=True,
                    help="Browse and resume previous conversations",
                    key="btn_chats",
                )

            with hdr_mid:
                summarize_clicked = st.button(
                    "Summarize",
                    use_container_width=True,
                    disabled=msg_count < 2,
                    help="Compress conversation history into a summary and continue",
                    key="btn_summarize",
                )

            with hdr_dl:
                _md_content = _build_chat_md()
                _md_filename = (
                    "genie_chat_" + datetime.now().strftime("%Y%m%d_%H%M") + ".md"
                )
                st.download_button(
                    label="Export MD",
                    data=_md_content,
                    file_name=_md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                    disabled=msg_count == 0,
                    help="Download this conversation as a Markdown file",
                    key="btn_dl_chat",
                )

            with hdr_right:
                clear_clicked = st.button(
                    "Clear",
                    use_container_width=True,
                    disabled=msg_count == 0,
                    type="secondary",
                    help="Clear all messages and start fresh",
                    key="btn_clear",
                )

            if chats_clicked:
                st.session_state["show_chats_panel"] = not st.session_state.get("show_chats_panel", False)
                if st.session_state["show_chats_panel"]:
                    cp_tmp = st.session_state.get("chat_persistence")
                    if cp_tmp:
                        try:
                            st.session_state["_all_sessions_cache"] = cp_tmp.load_all_sessions()
                        except Exception:
                            st.session_state["_all_sessions_cache"] = []

            if st.session_state.get("show_chats_panel", False):
                all_sess = st.session_state.get("_all_sessions_cache", [])
                cp_tmp   = st.session_state.get("chat_persistence")

                with st.container(border=True):
                    st.markdown(
                        "<div style='display:flex;align-items:center;justify-content:space-between;"
                        "margin-bottom:12px;'><div style='font-size:14px;font-weight:800;"
                        "color:#1e40af;'> Previous Conversations</div></div>",
                        unsafe_allow_html=True,
                    )

                    import uuid as _uuid_panel
                    if st.button("New Conversation", key="btn_panel_new",
                                 use_container_width=True, type="primary"):
                        st.session_state.genie_messages       = []
                        st.session_state.analyst_response     = None
                        st.session_state.show_analysis        = False
                        st.session_state.selected_analysis    = None
                        st.session_state.last_custom_query    = ""
                        st.session_state.genie_session_id     = str(_uuid_panel.uuid4())
                        st.session_state.genie_session_label  = "Chat on " + datetime.now().strftime("%b %d %H:%M")
                        st.session_state.chat_turn_index      = 0
                        st.session_state.restore_dismissed    = True
                        st.session_state["show_chats_panel"]  = False
                        st.rerun()

                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    if not all_sess:
                        st.info("No previous conversations found in the last 2 days.", icon="💬")
                    else:
                        for s_idx, sess in enumerate(all_sess):
                            age_h   = sess.get("age_hours", 0)
                            label   = sess.get("session_label", "Previous chat")
                            n_turns = sess.get("turn_count", 0)
                            is_current = (sess.get("session_id") ==
                                          st.session_state.get("genie_session_id", ""))

                            if age_h < 1:   age_str = "< 1 hr ago"
                            elif age_h < 24: age_str = f"{int(age_h)}h ago"
                            else:            age_str = f"{int(age_h/24)}d {int(age_h%24)}h ago"

                            row_bg  = "#eff6ff" if is_current else "#f8fafc"
                            row_bdr = "#bfdbfe" if is_current else "#e2e8f0"

                            sl, sr = st.columns([5, 2], gap="small")
                            with sl:
                                current_tag = (" <span style='background:#dcfce7;color:#15803d;"
                                              "border-radius:10px;padding:1px 8px;font-size:10px;"
                                              "font-weight:700;'>Active</span>") if is_current else ""
                                st.markdown(
                                    f"<div style='background:{row_bg};border:1px solid {row_bdr};"
                                    f"border-radius:10px;padding:9px 12px;margin-bottom:4px;'>"
                                    f"<div style='font-size:13px;font-weight:700;color:#0f172a;'>"
                                    f"{html.escape(label)}{current_tag}</div>"
                                    f"<div style='font-size:11px;color:#64748b;margin-top:2px;'>"
                                    f"{n_turns} messages &nbsp;·&nbsp; {age_str}</div>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                            with sr:
                                if is_current:
                                    st.markdown(
                                        "<div style='padding:10px 0;text-align:center;font-size:12px;"
                                        "color:#64748b;'>Current</div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    if st.button(
                                        "▶ Resume",
                                        key=f"btn_panel_resume_{s_idx}",
                                        use_container_width=True,
                                        type="primary",
                                    ):
                                        with st.spinner("Loading conversation..."):
                                            msgs = cp_tmp.load_session_messages(sess["session_id"]) if cp_tmp else []
                                        st.session_state.genie_messages      = msgs
                                        st.session_state.chat_turn_index     = len(msgs)
                                        st.session_state.genie_session_id    = sess["session_id"]
                                        st.session_state.genie_session_label = sess["session_label"]
                                        st.session_state.restore_dismissed   = True
                                        st.session_state["show_chats_panel"] = False
                                        st.rerun()

                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                    if st.button("Close", key="btn_panel_close",
                                 use_container_width=True, type="secondary"):
                        st.session_state["show_chats_panel"] = False
                        st.rerun()

            if summarize_clicked:
                msgs       = st.session_state.get("genie_messages", [])
                transcript = "\n".join([
                    f"{'User' if m['role'] == 'user' else 'AI'}: {m.get('content', '')}"
                    for m in msgs if m.get("content")
                ])
                try:
                    with st.spinner("Summarizing conversation..."):
                        # Migration note: get_snowflake_connection() + session.sql(CORTEX.COMPLETE) → bedrock_complete()
                        summary = bedrock_complete(
                            f"Summarize this dealer analytics conversation in 4-5 bullet points. "
                            f"Keep key findings, dealer names, and important numbers:\n\n"
                            f"{transcript[:3000]}",
                            model_id=get_config()["bedrock"]["primary_model"],
                        ) or "Previous conversation summarized."
                except Exception:
                    summary = "Previous conversation context retained."

                st.session_state.genie_messages = [{
                    "role":      "assistant",
                    "content":   f"Conversation summary:\n{summary}",
                    "timestamp": pd.Timestamp.now(),
                    "response":  None,
                }]
                st.rerun()

            if clear_clicked:
                import uuid as _uuid
                st.session_state.genie_messages       = []
                st.session_state.analyst_response     = None
                st.session_state.show_analysis        = False
                st.session_state.selected_analysis    = None
                st.session_state.last_custom_query    = ""
                st.session_state.genie_session_id     = str(_uuid.uuid4())
                st.session_state.genie_session_label  = (
                    "Chat on " + datetime.now().strftime("%b %d %H:%M")
                )
                st.session_state.chat_turn_index      = 0
                st.session_state.restore_dismissed    = True
                st.session_state.restore_offered      = False
                st.session_state["_all_sessions_cache"] = []
                st.rerun()

            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

            with st.container(height=620, border=True):

                all_messages = st.session_state.get("genie_messages", [])

                cp = st.session_state.get("chat_persistence")
                if (
                    not all_messages
                    and cp
                    and not st.session_state.get("restore_dismissed")
                    and not st.session_state.get("restore_offered")
                ):
                    try:
                        sessions = cp.load_all_sessions()
                    except Exception:
                        sessions = []
                    st.session_state["_all_sessions_cache"] = sessions
                    st.session_state["restore_offered"] = True

                all_sessions = st.session_state.get("_all_sessions_cache", [])

                if (
                    not all_messages
                    and all_sessions
                    and not st.session_state.get("restore_dismissed")
                ):
                    st.markdown("""
                    <div style="background:#eff6ff;border:1.5px solid #bfdbfe;
                        border-radius:14px;padding:18px 20px 10px 20px;
                        margin:16px 8px 12px 8px;">
                        <div style="font-size:16px;font-weight:800;color:#1e40af;
                            margin-bottom:4px;"> Resume a previous conversation</div>
                        <div style="font-size:13px;color:#374151;margin-bottom:14px;">
                            You have chats from the last 2 days. Pick one to continue,
                            or start fresh below.
                        </div>
                    </div>""", unsafe_allow_html=True)

                    for s_idx, sess in enumerate(all_sessions):
                        age_h  = sess.get("age_hours", 0)
                        label  = sess.get("session_label", "Previous chat")
                        n_turns = sess.get("turn_count", 0)
                        if age_h < 1:
                            age_str = "less than 1 hr ago"
                        elif age_h < 24:
                            age_str = f"{int(age_h)}h ago"
                        else:
                            age_str = f"{int(age_h/24)}d {int(age_h%24)}h ago"

                        sc_left, sc_right = st.columns([5, 2], gap="small")
                        with sc_left:
                            st.markdown(
                                f'<div style="background:#fff;border:1px solid #e2e8f0;'
                                f'border-radius:10px;padding:10px 14px;margin-bottom:6px;">'
                                f'<div style="font-size:13px;font-weight:700;color:#0f172a;">'
                                f'{html.escape(label)}</div>'
                                f'<div style="font-size:11px;color:#64748b;margin-top:2px;">'
                                f'{n_turns} messages &nbsp;·&nbsp; {age_str}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                        with sc_right:
                            if st.button(
                                "▶ Resume",
                                key=f"btn_resume_{s_idx}",
                                use_container_width=True,
                                type="primary",
                            ):
                                with st.spinner("Loading conversation..."):
                                    msgs = cp.load_session_messages(sess["session_id"])
                                st.session_state.genie_messages      = msgs
                                st.session_state.chat_turn_index     = len(msgs)
                                st.session_state.genie_session_id    = sess["session_id"]
                                st.session_state.genie_session_label = sess["session_label"]
                                st.session_state.restore_dismissed   = True
                                st.rerun()

                    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
                    if st.button("Start a new conversation",
                                 key="btn_start_fresh",
                                 use_container_width=True,
                                 type="secondary"):
                        st.session_state.restore_dismissed = True
                        st.rerun()

                elif not all_messages:
                    st.markdown("""
                    <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;height:560px;text-align:center;">
                        <div style="font-size:52px;margin-bottom:16px;">💬</div>
                        <div style="font-size:18px;font-weight:800;color:#1a1a1a;
                            margin-bottom:8px;">Start a Conversation</div>
                        <div style="font-size:14px;color:#64748b;max-width:360px;">
                            Select a quick analysis above, use a suggestion on the left,
                            or type a question below.
                        </div>
                    </div>""", unsafe_allow_html=True)

                for msg_idx, msg in enumerate(all_messages):
                    role     = msg.get("role", "user")
                    content  = msg.get("content", "") or ""
                    response = msg.get("response")

                    if role == "user":
                        st.markdown(f"""
                        <div class="g-user">
                            <div class="g-user-inner">
                                <div class="g-user-lbl">You</div>
                                {html.escape(content)}
                            </div>
                        </div>""", unsafe_allow_html=True)

                    else:
                        if content:
                            st.markdown(f"""
                            <div class="g-ai">
                                <div class="g-ai-inner">
                                    <div class="g-ai-lbl">AI Assistant</div>
                                    {html.escape(content)}
                                </div>
                            </div>""", unsafe_allow_html=True)

                        if response:
                            st.markdown('<div class="g-card">', unsafe_allow_html=True)
                            render_result_panel(response, msg_idx)
                            st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(
                    '<div id="genie-bottom" style="height:4px;"></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("""
            <script>
            (function() {
                function scrollChat() {
                    var anchor = document.getElementById('genie-bottom');
                    if (!anchor) { setTimeout(scrollChat, 150); return; }
                    var el = anchor.parentElement;
                    while (el) {
                        var ov = window.getComputedStyle(el).overflowY;
                        if (ov === 'auto' || ov === 'scroll') {
                            el.scrollTop = el.scrollHeight;
                            return;
                        }
                        el = el.parentElement;
                    }
                    anchor.scrollIntoView({ behaviour:'smooth', block:'end' });
                }
                scrollChat();
                setTimeout(scrollChat, 400);
                setTimeout(scrollChat, 900);
            })();
            </script>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

            with st.form("genie_question_form", clear_on_submit=True):
                inp_col, btn_col = st.columns([0.88, 0.12])
                with inp_col:
                    user_query = st.text_input(
                        "Ask",
                        placeholder="Ask about dealer metrics, performance, trends...",
                        label_visibility="collapsed",
                        key=f"genie_chat_input_{st.session_state.genie_input_version}",
                    )
                with btn_col:
                    send_clicked = st.form_submit_button("Send →", width='stretch')

            if send_clicked and user_query:
                st.session_state.selected_analysis   = "custom"
                st.session_state.last_custom_query   = user_query.strip()
                st.session_state.show_analysis       = True
                st.session_state.genie_input_version += 1
                with st.spinner("Analyzing..."):
                    response = process_genie_query(user_query)
                st.session_state.analyst_response = response
                st.rerun()

        with tab_forecast:
            with st.container(border=True):
                st.markdown(
                    '<div style="font-size:16px;font-weight:800;color:#0f172a;'
                    'margin-bottom:20px;">📊 Revenue Forecast & Anomalies</div>',
                    unsafe_allow_html=True,
                )
                fc1, fc2, fc3 = st.columns([2, 1, 1], gap="medium")
                with fc1:
                    st.session_state.forecast_weeks = st.slider(
                        "Forecast Weeks", min_value=1, max_value=26,
                        value=st.session_state.get('forecast_weeks', 8), step=1,
                    )
                with fc2:
                    show_anomalies = st.checkbox("Show Anomalies", value=True)
                with fc3:
                    if st.button("Run Forecast", use_container_width=True):
                        st.session_state['run_predictions'] = True
                        st.rerun()
                st.divider()

                if st.session_state.get('run_predictions') or st.session_state.get('analyst_response'):
                    resp = st.session_state.get('analyst_response')
                    if resp:
                        dealer_name = extract_dealer_name_from_context(
                            resp, st.session_state.get('last_custom_query', '')
                        )
                        if dealer_name:
                            with st.spinner("⏳ Generating forecast and detecting anomalies..."):
                                if st.session_state.get('forecaster') is None:
                                    # Migration note: DealerPerformanceForecaster(session) → DealerPerformanceForecaster()
                                    st.session_state.forecaster = DealerPerformanceForecaster()
                                if st.session_state.get('detector') is None:
                                    # Migration note: AnomalyDetector(session) → AnomalyDetector()
                                    st.session_state.detector = AnomalyDetector()
                                forecast_result = st.session_state.forecaster.forecast_revenue(
                                    dealer_name, st.session_state.forecast_weeks)
                                anomaly_result = (
                                    st.session_state.detector.detect_dealer_anomalies(dealer_name)
                                    if show_anomalies else {}
                                )

                            if forecast_result.get("success"):
                                fcol1, fcol2 = st.columns(2, gap="large")
                                with fcol1:
                                    st.markdown("**Revenue Forecast**")
                                    try:
                                        import altair as alt
                                        fdata = pd.DataFrame({
                                            'Week':    range(1, len(forecast_result['forecast_values']) + 1),
                                            'Forecast':forecast_result['forecast_values'],
                                            'Upper':   forecast_result['forecast_upper_bound'],
                                            'Lower':   forecast_result['forecast_lower_bound'],
                                        })
                                        st.altair_chart(
                                            alt.Chart(fdata).mark_line(point=True).encode(
                                                x='Week:Q', y='Forecast:Q'
                                            ).properties(height=250, width=350),
                                            use_container_width=True,
                                        )
                                    except Exception:
                                        st.line_chart(forecast_result['forecast_values'])
                                    st.markdown(f"**Trend:** {forecast_result.get('trend','N/A').upper()}")
                                    st.markdown(f"**Expected Change:** {forecast_result.get('change_percent',0):.1f}%")
                                    st.markdown(f"**Confidence:** {forecast_result.get('confidence','unknown')}")
                                    st.markdown(f"**Accuracy (MAPE):** {forecast_result.get('mape',0):.1f}%")
                                with fcol2:
                                    if anomaly_result and "anomalies_count" in anomaly_result:
                                        st.markdown("**Anomalies & Risk**")
                                        rl   = anomaly_result.get('risk_level', 'low').upper()
                                        icon = {"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}.get(rl, "⚪")
                                        st.markdown(f"**Risk Level:** {icon} {rl}")
                                        st.markdown(f"**Anomalies Found:** {anomaly_result.get('anomalies_count',0)}")
                                        st.markdown(f"**Anomaly Rate:** {anomaly_result.get('anomaly_rate',0):.1f}%")
                                        if anomaly_result.get('anomalies'):
                                            st.markdown("**Anomaly Details:**")
                                            for anom in anomaly_result['anomalies'][:5]:
                                                si = {"critical":"🔴","high":"🟠","medium":"🟡"}.get(
                                                    anom.get('severity','medium'), "⚪")
                                                st.markdown(
                                                    f"{si} **{anom.get('period')}** — "
                                                    f"Revenue: ${anom.get('revenue',0):,.0f}"
                                                )
                                    else:
                                        st.info("✅ No anomalies detected")

                                with st.spinner("🔮 Generating predictive insights..."):
                                    # Migration note: generate_forecast_prediction_text(session, dealer_name, ...) → generate_forecast_prediction_text(dealer_name, ...)
                                    prediction_text = generate_forecast_prediction_text(
                                        dealer_name, forecast_result, anomaly_result)
                                if prediction_text:
                                    st.markdown(f"""
                                    <div style="padding:14px;background:#fef3c7;border-radius:10px;
                                         border-left:4px solid #f59e0b;margin-top:12px;">
                                        <div style="font-size:12px;font-weight:800;color:#92400e;
                                             margin-bottom:8px;">🔮 Prediction — What to expect</div>
                                        <div style="color:#1a1a1a;font-size:14px;line-height:1.6;">
                                            {html.escape(prediction_text).replace(chr(10),'<br/>')}
                                        </div>
                                    </div>""", unsafe_allow_html=True)
                            elif "error" in forecast_result:
                                st.warning(f"❌ {forecast_result['error']}")
                            st.session_state['run_predictions'] = False
                        else:
                            st.info("💡 Could not extract dealer name. Please ask about a specific dealer first.")
                    else:
                        st.markdown("""
                        <div style="display:flex;flex-direction:column;align-items:center;
                             justify-content:center;padding:60px 20px;text-align:center;">
                            <div style="font-size:64px;margin-bottom:16px;">📈</div>
                            <div style="font-size:16px;font-weight:800;color:#1a1a1a;
                                 margin-bottom:8px;">No Forecast Yet</div>
                            <div style="font-size:14px;color:#64748b;max-width:400px;">
                                Ask about a specific dealer in Chat Analysis, then run forecast here.
                            </div>
                        </div>""", unsafe_allow_html=True)


# ── RENDER FUNCTIONS ─────────────────────────────────────────────────────────

def render_journey_with_counts(filters=None):
    # Migration note: session parameter removed; fetch_journey_counts(session, filters) → fetch_journey_counts(filters)
    """
    Render the Dealer Journey horizontal timeline with live counts from DB.
    Replaces the static journey HTML in render_dealer_life_cycle().
    """
    # Migration note: fetch_journey_counts(session, filters) → fetch_journey_counts(filters)
    counts_df = fetch_journey_counts(filters)
    row = counts_df.iloc[0] if not counts_df.empty else {}
    if filters and filters.get('transaction_id') and filters.get('transaction_id') != 'All':
        st.caption("Journey counts are scoped to the selected Transaction ID")

    def _c(col):
        try:
            v = row.get(col, 0) or 0
            if col == 'AVG_LEAD_DAYS':
                try:
                    fv = float(v)
                    return f"{int(fv)}" if fv > 0 else "N/A"
                except Exception:
                    return "N/A"
            return f"{int(v):,}"
        except Exception:
            return "0"

    stages = [
        ("""<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#3b82f6"><path d="M19 6h-2c0-2.76-2.24-5-5-5S7 3.24 7 6H5c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm-7-3c1.66 0 3 1.34 3 3H9c0-1.66 1.34-3 3-3z"/></svg>""",
         "Order",    "ORDER_COUNT"),
        ("""<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#8b5cf6"><path d="M20 8h-2.81c-.45-.78-1.07-1.45-1.82-1.96L17 4.41 15.59 3l-2.17 2.17C13 5.06 12.51 5 12 5s-1 .06-1.41.17L8.41 3 7 4.41l1.62 1.63C7.88 6.55 7.26 7.22 6.81 8H4v2h2.09c-.05.33-.09.66-.09 1v1H4v2h2v1c0 .34.04.67.09 1H4v2h2.81c1.04 1.79 2.97 3 5.19 3s4.15-1.21 5.19-3H20v-2h-2.09c.05-.33.09-.66.09-1v-1h2v-2h-2v-1c0-.34-.04-.67-.09-1H20V8z"/></svg>""",
         "Delivery", "DELIVERY_COUNT"),
        ("""<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#f59e0b"><path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/></svg>""",
         "Invoice",  "INVOICE_COUNT"),
        ("""<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#10b981"><path d="M11.8 10.9c-2.27-.59-3-1.2-3-2.15 0-1.09 1.01-1.85 2.7-1.85 1.78 0 2.44.85 2.5 2.1h2.21c-.07-1.72-1.12-3.3-3.21-3.81V3h-3v2.16c-1.94.42-3.5 1.68-3.5 3.61 0 2.31 1.91 3.46 4.7 4.13 2.5.6 3 1.48 3 2.41 0 .69-.49 1.79-2.7 1.79-2.06 0-2.87-.92-2.98-2.1h-2.2c.12 2.19 1.76 3.42 3.68 3.83V21h3v-2.15c1.95-.37 3.5-1.5 3.5-3.55 0-2.84-2.43-3.81-4.7-4.4z"/></svg>""",
         "Paid",     "PAID_COUNT"),
        ("""<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="#ef4444"><path d="M22.7 19l-9.1-9.1c.9-2.3.4-5-1.5-6.9-2-2-5-2.4-7.4-1.3L9 6 6 9 1.6 4.7C.4 7.1.9 10.1 2.9 12.1c1.9 1.9 4.6 2.4 6.9 1.5l9.1 9.1c.4.4 1 .4 1.4 0l2.3-2.3c.5-.4.5-1.1.1-1.4z"/></svg>""",
         "Warranty", "WARRANTY_COUNT"),
    ]

    avg_lead_val = _c("AVG_LEAD_DAYS")

    html_parts = [textwrap.dedent("""\
    <style>
    .journey-wrap {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        position: relative;
        padding: 18px 8px 8px 8px;
        margin-top: 8px;
    }
    .journey-wrap::before {
        content: "";
        position: absolute;
        top: 38px;
        left: 8%;
        right: 8%;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #f59e0b, #10b981, #ef4444);
        border-radius: 3px;
        z-index: 0;
    }
    .journey-step {
        flex: 1;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    .journey-icon-wrap {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        background: #fff;
        border: 2px solid #e5e7eb;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .journey-label {
        font-size: 12px;
        font-weight: 700;
        color: #374151;
        margin-bottom: 3px;
    }
    .journey-count {
        font-size: 18px;
        font-weight: 900;
        color: #111827;
        line-height: 1.1;
    }
    .journey-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22c55e;
        margin-right: 4px;
        vertical-align: middle;
        animation: pulse 2s infinite;
    }
    .journey-lead-badge {
        position: absolute;
        top: 22px;
        left: 50%;
        transform: translateX(-50%);
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 700;
        color: #6366f1;
        white-space: nowrap;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        z-index: 2;
    }
    @keyframes pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.4; }
        100% { opacity: 1; }
    }
    </style>
    <div style="position:relative;">
    <div class="journey-wrap">
    """)]

    for i, (icon_svg, label, count_col) in enumerate(stages):
        count_val = _c(count_col)
        count_html = f'<div class="journey-count"> <span class="journey-dot"></span>{count_val} </div>'
        html_parts.append(textwrap.dedent(f"""\
        <div class="journey-step">
            <div class="journey-icon-wrap">{icon_svg}</div>
            <div class="journey-label">{label}</div>
            {count_html}
        </div>
        """))
        if i == 0:
            html_parts.append(textwrap.dedent(f"""\
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;position:relative;z-index:2;
                        min-width:80px;margin-top:-8px;">
                <div style="background:#fff;border:1.5px solid #e0e7ff;border-radius:20px;
                            padding:3px 12px;font-size:11px;font-weight:700;color:#6366f1;
                            white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,0.08);
                            line-height:1.6;">
                    {avg_lead_val} days
                </div>
                <div style="font-size:9px;color:#9ca3af;font-weight:500;margin-top:2px;">
                    Avg Lead Days
                </div>
            </div>
            """))

    html_parts.append("</div>")
    html_parts.append("</div>")
    html_out = textwrap.dedent("".join(html_parts))
    st.markdown(html_out, unsafe_allow_html=True)


def render_transaction_lineage(filters=None):
    # Migration note: session parameter removed; fetch_transaction_lineage(session, filters) → fetch_transaction_lineage(filters)
    """
    Render the transaction lineage table with Y/N stage indicators.
    """
    st.markdown("""
    <style>
    .lineage-section-card {
        background: #ffffff;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04);
        padding: 0;
        margin-bottom: 18px;
        overflow: hidden;
        border: 1px solid #f0f0f0;
    }
    .lineage-header-bar {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        padding: 18px 24px 14px 24px;
    }
    .lineage-header-title {
        font-size: 17px; font-weight: 800; color: #fff; margin-bottom: 3px;
    }
    .lineage-header-sub {
        font-size: 12px; color: rgba(255,255,255,0.80);
    }
    .lineage-filter-strip {
        background: #f8faff;
        border-bottom: 1px solid #e9ecf5;
        padding: 14px 20px 10px 20px;
    }
    .lineage-filter-label {
        font-size: 10px; font-weight: 700; color: #7c3aed;
        text-transform: uppercase; letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .lineage-table-area {
        padding: 0 0 4px 0;
    }
    [data-testid="stDataFrame"] > div { border-radius: 0; overflow: hidden; }
    [data-testid="stDataFrame"] thead th {
        background: #f3f4f6 !important;
        color: #374151 !important;
        font-weight: 700 !important;
        font-size: 11px !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 2 !important;
        border-bottom: 2px solid #e5e7eb !important;
        padding: 10px 10px !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td { background: #f9fafb !important; }
    [data-testid="stDataFrame"] tbody tr:nth-child(odd)  td { background: #ffffff !important; }
    [data-testid="stDataFrame"] tbody tr:hover td { background: #eef2ff !important; }
    [data-testid="stDataFrame"] tbody td {
        padding: 9px 10px !important;
        font-size: 12.5px !important;
        border-bottom: 1px solid #f3f4f6 !important;
    }
    .lineage-pagination-bar {
        background: #f8faff;
        border-top: 1px solid #e9ecf5;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .lineage-page-info {
        font-size: 12px; color: #6b7280; font-weight: 500;
    }
    .lineage-download-bar {
        padding: 10px 20px 16px 20px;
        border-top: 1px solid #f0f0f0;
        background: #fff;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 10px !important; font-weight: 700 !important;
        color: #4f46e5 !important; text-transform: uppercase !important;
        letter-spacing: 0.4px !important;
    }
    div[data-testid="stSelectbox"] > div > div > div {
        border-radius: 8px !important;
        border: 1.5px solid #e0e7ff !important;
        font-size: 13px !important;
        background: #fff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(90deg,#4f46e5 0%,#7c3aed 100%);
                border-radius:14px 14px 0 0;padding:18px 24px 14px 24px;margin-bottom:0;">
        <div style="font-size:17px;font-weight:800;color:#fff;margin-bottom:3px;">
            Transaction Lineage
        </div>
        <div style="font-size:12px;color:rgba(255,255,255,0.82);">
            Per-order view of progress through Order → Delivery → Invoice → Paid → Warranty
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        # compute date bounds up front (needed by tx dropdown query)
        from_date, to_date = resolve_date_range(filters or {})
        from_date_str = from_date.strftime('%Y-%m-%d') if hasattr(from_date, 'strftime') else str(from_date)
        to_date_str   = to_date.strftime('%Y-%m-%d')   if hasattr(to_date,   'strftime') else str(to_date)

        # read existing filter values from session (they persist across reruns)
        warranty_input    = st.session_state.get("lineage_warranty", "All")
        paid_input        = st.session_state.get("lineage_paid", "All")
        tx_input          = st.session_state.get("lineage_tx", "All")
        invoice_status_input = st.session_state.get("lineage_invoice_status", "All")

        # build base_filters for fetching transaction IDs (date/dealer/paid/warranty)
        base_filters = {} if filters is None else filters.copy()
        if paid_input and paid_input != "All":
            base_filters["paid"] = paid_input
        if warranty_input and warranty_input != "All":
            base_filters["warranty_status"] = warranty_input.upper()
        if invoice_status_input and invoice_status_input != "All":
            base_filters["invoice_status"] = invoice_status_input

        # fetch available transaction IDs for the current scope so dropdown can show them
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            df_tx = run_df(f"""
                SELECT DISTINCT TRANSACTION_ID
                FROM {_DB}.VW_TRANSACTION_LINEAGE
                WHERE 1=1
                {dealer_filter_clause('VW_TRANSACTION_LINEAGE', base_filters)}
                {lineage_filter_clause(base_filters)}
                AND ORDER_DATE BETWEEN '{from_date_str}' AND '{to_date_str}'
                ORDER BY TRANSACTION_ID
                LIMIT 1000
            """)
            tx_options = ["All"] + df_tx['TRANSACTION_ID'].astype(str).tolist()
        except Exception:
            tx_options = ["All"]

        try:
            # Migration note: session.sql().to_pandas() → run_df()
            df_inv = run_df(f"""
                SELECT DISTINCT INVOICE_STATUS
                FROM {_DB}.VW_TRANSACTION_LINEAGE
                WHERE INVOICE_STATUS IS NOT NULL
                ORDER BY INVOICE_STATUS
                LIMIT 50
            """)
            invoice_status_options = ["All"] + df_inv['INVOICE_STATUS'].astype(str).tolist()
        except Exception:
            invoice_status_options = ["All", "Paid", "Pending", "Overdue", "Cancelled"]

        st.markdown("""
        <div style="font-size:11px;font-weight:700;color:#4f46e5;text-transform:uppercase;
                    letter-spacing:0.5px;margin-bottom:6px;padding:4px 0 0 2px;">
            Filter Transactions
        </div>
        """, unsafe_allow_html=True)

        # render all filter controls on one row
        fcol1, fcol2, fcol3, fcol4 = st.columns([1, 1, 1, 1])
        with fcol1:
            tx_input = st.selectbox(
                "Transaction ID",
                options=tx_options,
                index=tx_options.index(tx_input) if tx_input in tx_options else 0,
                key="lineage_tx"
            )
        with fcol2:
            paid_input = st.selectbox(
                "Paid status",
                ["All", "Y", "N"],
                index=["All","Y","N"].index(paid_input) if paid_input in ["All","Y","N"] else 0,
                key="lineage_paid"
            )
        with fcol3:
            warranty_input = st.selectbox(
                "Warranty status",
                ["All", "Active", "Expired", "Not Applicable"],
                index=["All","Active","Expired","Not Applicable"].index(warranty_input) if warranty_input in ["All","Active","Expired","Not Applicable"] else 0,
                key="lineage_warranty"
            )
        with fcol4:
            invoice_status_input = st.selectbox(
                "Invoice Status",
                options=invoice_status_options,
                index=invoice_status_options.index(invoice_status_input) if invoice_status_input in invoice_status_options else 0,
                key="lineage_invoice_status"
            )

        # merge caller filters with quick inputs
        merged_filters = {} if filters is None else filters.copy()
        if tx_input and tx_input != "All":
            merged_filters["transaction_id"] = tx_input
        if paid_input and paid_input != "All":
            merged_filters["paid"] = paid_input
        if warranty_input and warranty_input != "All":
            merged_filters["warranty_status"] = warranty_input.upper()
        if invoice_status_input and invoice_status_input != "All":
            merged_filters["invoice_status"] = invoice_status_input

        if (tx_input and tx_input != "All") or (paid_input and paid_input != "All") or (warranty_input and warranty_input != "All") or (invoice_status_input and invoice_status_input != "All"):
            merged_filters["date_range"] = "All Dates"

        # Pagination controls
        col1, col2, col3 = st.columns([1, 1, 2])
        page_size = st.session_state.get("lineage_page_size", 10)
        if "lineage_page_requested" in st.session_state:
            st.session_state["lineage_page"] = int(st.session_state.pop("lineage_page_requested"))
        page = int(st.session_state.get("lineage_page", 1))

        # Get total rows for pagination
        from_date, to_date = resolve_date_range(merged_filters or {})
        from_date_str = from_date.strftime('%Y-%m-%d') if hasattr(from_date, 'strftime') else str(from_date)
        to_date_str   = to_date.strftime('%Y-%m-%d')   if hasattr(to_date,   'strftime') else str(to_date)
        where_clauses = [f"ORDER_DATE BETWEEN '{from_date_str}' AND '{to_date_str}'"]
        if merged_filters.get('dealer') and merged_filters['dealer'] != 'All Dealers':
            d = str(merged_filters['dealer']).replace("'", "''")
            where_clauses.append(f"DEALER_NAME = '{d}'")
        if merged_filters.get('transaction_id'):
            tid = str(merged_filters['transaction_id']).replace("'", "''")
            where_clauses.append(f"TRANSACTION_ID = '{tid}'")
        if merged_filters.get('product'):
            prod = str(merged_filters['product']).replace("'", "''")
            where_clauses.append(f"(PRODUCT_CATEGORY ILIKE '%{prod}%' OR PRODUCT_DESC ILIKE '%{prod}%')")
        if merged_filters.get('paid'):
            where_clauses.append(f"PAID_FLAG = '{merged_filters['paid']}'")
        if merged_filters.get('invoice_status') and merged_filters['invoice_status'] != 'All':
            inv = str(merged_filters['invoice_status']).replace("'", "''")
            where_clauses.append(f"UPPER(INVOICE_STATUS) = UPPER('{inv}')")
        where_clause = " AND ".join(where_clauses)
        # Migration note: session.sql(COUNT).to_pandas()['TOTAL'][0] → run_df().iloc[0,0]
        total_rows_df = run_df(f"""
            SELECT COUNT(*) AS total
            FROM {_DB}.VW_TRANSACTION_LINEAGE
            WHERE {where_clause}
        """)
        total_rows = int(total_rows_df.iloc[0, 0]) if not total_rows_df.empty else 0
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        page = st.selectbox(
            "Page",
            options=list(range(1, total_pages + 1)),
            key="lineage_page",
        )

        # Migration note: fetch_transaction_lineage(session, merged_filters, ...) → fetch_transaction_lineage(merged_filters, ...)
        df = fetch_transaction_lineage(merged_filters, page=int(page), page_size=page_size)

        if df.empty:
            msg = "No transaction data found for the selected filters."
            if merged_filters.get('warranty_status'):
                msg += " Try widening the date range (e.g. choose 'All Dates')."
            st.markdown(f"""
            <div style="text-align:center;padding:40px 20px;background:#f8faff;
                        border-radius:10px;margin:10px 0;">
                <div style="font-size:15px;font-weight:600;color:#374151;margin-bottom:6px;">
                    No Transactions Found
                </div>
                <div style="font-size:13px;color:#6b7280;">{msg}</div>
            </div>
            """, unsafe_allow_html=True)
            return

        total_in_page = len(df)
        delivered_ct  = (df.get('DELIVERY_DONE', pd.Series()) == 'Y').sum() if 'DELIVERY_DONE' in df.columns else 0
        m1, m2, m3 = st.columns(3)
        metric_style = "text-align:center;padding:10px 6px 8px 6px;background:#f0f4ff;border-radius:10px;margin-bottom:10px;"
        with m1:
            st.markdown(f'<div style="{metric_style}"><div style="font-size:20px;font-weight:800;color:#4f46e5;">{total_rows:,}</div><div style="font-size:11px;color:#6b7280;font-weight:600;">Total Rows</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div style="{metric_style}"><div style="font-size:20px;font-weight:800;color:#16a34a;">{delivered_ct}</div><div style="font-size:11px;color:#6b7280;font-weight:600;">Delivered (Page)</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div style="{metric_style}"><div style="font-size:20px;font-weight:800;color:#7c3aed;">{total_pages}</div><div style="font-size:11px;color:#6b7280;font-weight:600;">Total Pages</div></div>', unsafe_allow_html=True)

        # Rename columns for display
        display_df = df.rename(columns={
            'TRANSACTION_ID':   'Transaction ID',
            'DEALER_NAME':      'Dealer',
            'PRODUCT_CATEGORY': 'Category',
            'PRODUCT_DESC':     'Product',
            'ORDER_DATE':       'Order Date',
            'DELIVERY_DATE':    'Delivery Date',
            'INVOICE_DATE':     'Invoice Date',
            'PAYMENT_DATE':     'Payment Date',
            'LEAD_TIME_DAYS':   'Lead Time (Days)',
            'INVOICE_AMOUNT':   'Invoice Amount',
            'INVOICE_STATUS':   'Invoice Status',
            'ORDER_DONE':       'Order',
            'DELIVERY_DONE':    'Delivered',
            'INVOICE_DONE':     'Invoiced',
            'WARRANTY_STATUS':  'Warranty Status',
        })

        stage_cols = ['Order', 'Delivered', 'Invoiced']

        def _color_yn(val):
            if val == 'Y':
                return 'background-color:#dcfce7; color:#16a34a; font-weight:700; text-align:center;'
            elif val == 'N':
                return 'background-color:#fee2e2; color:#dc2626; font-weight:700; text-align:center;'
            return ''

        def _color_invoice_status(val):
            v = str(val).strip().upper()
            if v == 'PAID':
                return 'background-color:#dcfce7; color:#16a34a; font-weight:600;'
            elif v in ('PENDING', 'OPEN'):
                return 'background-color:#fef3c7; color:#92400e; font-weight:600;'
            elif v in ('OVERDUE', 'FAILED'):
                return 'background-color:#fee2e2; color:#dc2626; font-weight:600;'
            elif v in ('CANCELLED', 'VOID'):
                return 'background-color:#f3f4f6; color:#6b7280; font-weight:600;'
            return ''

        styled = (
            display_df.style
            .applymap(_color_yn, subset=stage_cols)
            .applymap(_color_invoice_status, subset=['Invoice Status'] if 'Invoice Status' in display_df.columns else [])
            .set_table_styles([
                {'selector': 'thead th',
                 'props': [('background-color', '#f8fafc'), ('color', '#374151'),
                           ('font-weight', '700'), ('font-size', '12px'),
                           ('border-bottom', '2px solid #e5e7eb'), ('padding', '10px 12px'),
                           ('position', 'sticky'), ('top', '0'), ('z-index', '1')]},
                {'selector': 'tbody tr:nth-child(even)',
                 'props': [('background-color', '#f9fafb')]},
                {'selector': 'tbody tr:nth-child(odd)',
                 'props': [('background-color', '#ffffff')]},
                {'selector': 'tbody tr:hover',
                 'props': [('background-color', '#eff6ff')]},
                {'selector': 'td',
                 'props': [('padding', '9px 12px'), ('font-size', '13px'),
                           ('border-bottom', '1px solid #f3f4f6')]},
            ])
        )

        st.dataframe(
            styled,
            use_container_width=True,
            height=400,
            hide_index=True,
        )

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        nav1, nav2, nav3 = st.columns([1, 1, 2])
        with nav1:
            if st.button("← Prev", key="lineage_prev", disabled=(page <= 1)):
                st.session_state["lineage_page_requested"] = int(max(1, int(page) - 1))
                st.rerun()
        with nav2:
            if st.button("Next →", key="lineage_next", disabled=(page >= total_pages)):
                st.session_state["lineage_page_requested"] = int(min(total_pages, int(page) + 1))
                st.rerun()
        with nav3:
            st.markdown(
                f"""<div style='text-align:right;font-size:12px;color:#6b7280;
                              background:#f0f4ff;border-radius:8px;
                              padding:8px 14px;font-weight:600;'>
                    Page <span style="color:#4f46e5;font-size:14px;">{page}</span> of {total_pages}
                    &nbsp;·&nbsp; {total_rows:,} total records
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="Download CSV",
                data=df.to_csv(index=False),
                file_name="transaction_lineage.csv",
                mime="text/csv",
                key="dl_lineage",
                use_container_width=True,
            )


# ============================================================================
# GRAPH DATA FETCH FUNCTIONS
# ============================================================================

# ===========================================================
# Dealer life cycle
# ===========================================================
def render_dealer_health_scorecard(filters):
    # Migration note: session parameter removed; fetch functions use athena_query() internally
    """Render health scorecard for selected dealer."""

    dealer_name = filters.get('dealer', '') or 'All Dealers'

    # Migration note: fetch_*(session, filters) → fetch_*(filters)
    revenue_growth = fetch_revenue_growth(filters)
    sales_vol      = fetch_sales_volume(filters)
    stock_avail    = fetch_stock_availability(filters)
    repair_tat     = fetch_repair_turnaround_time(filters)
    backorder      = fetch_backorder_incidence(filters)
    ccc            = fetch_cash_conversion_cycle(filters)
    gross_margin   = fetch_gross_profit_margin(filters)
    contrib_margin = fetch_contribution_margin(filters)

    # Avg sales volume for benchmarking
    try:
        # Migration note: session.sql().to_pandas() → run_df()
        avg_df = run_df(f"""
            SELECT AVG(dealer_total) as avg_vol
            FROM (
                SELECT DEALER_NAME, SUM(QUANTITY_SOLD) as dealer_total
                FROM DEALER_SALES_VIEW
                WHERE PERIOD_START_DATE BETWEEN '{filters['from_date']}' AND '{filters['to_date']}'
                GROUP BY DEALER_NAME
            )
        """)
        avg_sales_vol = float(avg_df['avg_vol'].iloc[0]) if not avg_df.empty else None
    except Exception:
        avg_sales_vol = sales_vol

    def safe_score(val, fn):
        """Return 0 when data is genuinely missing; score normally for real numbers including 0.0."""
        if val is None:
            return 0
        try:
            fval = float(val)
            import math
            if math.isnan(fval) or math.isinf(fval):
                return 0
            return fn(fval)
        except Exception:
            return 0

    def s_revenue_growth(v):
        if v < 0:   return 1
        if v < 2:   return 2
        if v < 4:   return 3
        if v < 6:   return 4
        if v < 8:   return 5
        if v < 12:  return 6
        return 7

    def s_sales_volume(v):
        if avg_sales_vol is None or avg_sales_vol == 0: return 4
        r = (v / avg_sales_vol) * 100
        if r < 50:  return 1
        if r < 65:  return 2
        if r < 80:  return 3
        if r < 90:  return 4
        if r < 100: return 5
        if r < 115: return 6
        return 7

    def s_stock_avail(v):
        if v < 40:  return 1
        if v < 55:  return 2
        if v < 65:  return 3
        if v < 75:  return 4
        if v < 85:  return 5
        if v < 93:  return 6
        return 7

    def s_repair_tat(v):
        if v > 72:  return 1
        if v > 60:  return 2
        if v > 48:  return 3
        if v > 36:  return 4
        if v > 24:  return 5
        if v > 12:  return 6
        return 7

    def s_backorder(v):
        if v > 20:  return 1
        if v > 15:  return 2
        if v > 10:  return 3
        if v > 7:   return 4
        if v > 4:   return 5
        if v > 2:   return 6
        return 7

    def s_ccc(v):
        if v > 60:  return 1
        if v > 45:  return 2
        if v > 30:  return 3
        if v > 20:  return 4
        if v > 12:  return 5
        if v > 6:   return 6
        return 7

    def s_gross_margin(v):
        if v < 10:  return 1
        if v < 20:  return 2
        if v < 30:  return 3
        if v < 40:  return 4
        if v < 55:  return 5
        if v < 65:  return 6
        return 7

    def s_contrib_margin(v):
        if v < 5:   return 1
        if v < 15:  return 2
        if v < 25:  return 3
        if v < 35:  return 4
        if v < 45:  return 5
        if v < 55:  return 6
        return 7

    def fmt_val(val, suffix=''):
        """Format a numeric value with suffix. Returns 'N/A' only when val is None or NaN."""
        if val is None:
            return 'N/A'
        try:
            import math
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return 'N/A'
            return f"{fval:.1f}{suffix}"
        except Exception:
            return 'N/A'

    def score_color(s):
        if s <= 2: return '#dc2626', '#fef2f2'
        if s <= 4: return '#f59e0b', '#fffbeb'
        if s < 6:  return '#3b82f6', '#eff6ff'
        return '#16a34a', '#f0fdf4'

    def score_label(s):
        if s == 0: return 'No Data'
        if s <= 2: return 'Critical'
        if s <= 4: return 'Average'
        if s < 6:  return 'Good'
        return 'Excellent'

    rows = [
        ('Sales Effectiveness',    'Revenue Growth',      fmt_val(revenue_growth, '%'),                                       safe_score(revenue_growth, s_revenue_growth)),
        ('Market Share',           'Sales Volume',        f"{int(sales_vol):,} units" if sales_vol is not None else 'N/A', safe_score(sales_vol, s_sales_volume)),
        ('Inventory Management',   'Stock Availability',  fmt_val(stock_avail, '%'),                                          safe_score(stock_avail, s_stock_avail)),
        ('Service & Parts',        'Avg Repair TAT',      fmt_val(repair_tat, ' hrs'),                                        safe_score(repair_tat, s_repair_tat)),
        ('Backorder Control',      'Backorder Incidence', fmt_val(backorder, '%'),                                            safe_score(backorder, s_backorder)),
        ('Operational Excellence', 'Cash Conv. Cycle',    fmt_val(ccc, ' days'),                                              safe_score(ccc, s_ccc)),
        ('Gross Profitability',    'Gross Margin',        fmt_val(gross_margin, '%'),                                         safe_score(gross_margin, s_gross_margin)),
        ('Contribution',           'Contrib. Margin',     fmt_val(contrib_margin, '%'),                                       safe_score(contrib_margin, s_contrib_margin)),
    ]

    all_scores   = [r[3] for r in rows]
    valid_scores = [s for s in all_scores if s > 0]
    avg_score    = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0
    ov_color, ov_bg = score_color(avg_score)
    ov_label     = score_label(avg_score)

    with st.container(border=True):

        # Header
        col_title, col_score = st.columns([3, 1])
        with col_title:
            st.markdown(
                f'<div style="font-size:17px;font-weight:700;color:#222;">Dealer Health Scorecard</div>'
                f'<div style="font-size:12px;color:#6b7280;margin-top:3px;">'
                f'DEALER Performance Framework &nbsp;·&nbsp; <b>{dealer_name}</b></div>',
                unsafe_allow_html=True
            )
        with col_score:
            st.markdown(f"""
<div style="text-align:center;background:{ov_bg};border:2px solid {ov_color};
     border-radius:12px;padding:10px 16px;">
  <div style="font-size:26px;font-weight:800;color:{ov_color};">
    {avg_score:.1f}<span style="font-size:13px;color:#9ca3af;font-weight:500;"> /7</span>
  </div>
  <div style="font-size:12px;font-weight:700;color:{ov_color};">{ov_label}</div>
  <div style="font-size:10px;color:#9ca3af;">Overall Health</div>
</div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # Column headers
        h1, h2, h3, h4, h5, h6 = st.columns([2.2, 1.8, 1.2, 0.8, 1.2, 1.8])
        for col, label in zip(
            [h1, h2, h3, h4, h5, h6],
            ['PERFORMANCE CATEGORY', 'KPI', 'VALUE', 'SCORE', 'RATING', 'PROGRESS']
        ):
            with col:
                st.markdown(
                    f'<div style="font-size:10px;font-weight:700;color:#6b7280;'
                    f'letter-spacing:0.5px;padding:6px 0;border-bottom:2px solid #e5e7eb;">'
                    f'{label}</div>',
                    unsafe_allow_html=True
                )

        # Data rows
        for cat, kpi, val, sc in rows:
            color, bg = score_color(sc)
            label     = score_label(sc)
            bar_pct   = int((sc / 7) * 100)
            val_color = color if val != 'N/A' else '#9ca3af'
            score_display = str(sc) if sc > 0 else '—'

            c1, c2, c3, c4, c5, c6 = st.columns([2.2, 1.8, 1.2, 0.8, 1.2, 1.8])
            with c1:
                st.markdown(
                    f'<div style="font-size:12px;font-weight:600;color:#222;'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">{cat}</div>',
                    unsafe_allow_html=True
                )
            with c2:
                st.markdown(
                    f'<div style="font-size:12px;color:#6b7280;'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">{kpi}</div>',
                    unsafe_allow_html=True
                )
            with c3:
                st.markdown(
                    f'<div style="font-size:12px;font-weight:700;color:{val_color};'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">{val}</div>',
                    unsafe_allow_html=True
                )
            with c4:
                st.markdown(
                    f'<div style="font-size:15px;font-weight:800;color:{color};'
                    f'padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                    f'{score_display}<span style="font-size:10px;color:#9ca3af;"> /7</span></div>',
                    unsafe_allow_html=True
                )
            with c5:
                st.markdown(
                    f'<div style="padding:10px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<span style="background:{bg};color:{color};font-size:11px;'
                    f'font-weight:700;border-radius:20px;padding:3px 10px;">'
                    f'{label}</span></div>',
                    unsafe_allow_html=True
                )
            with c6:
                st.markdown(
                    f'<div style="padding:14px 0 10px 0;border-bottom:1px solid #f1f5f9;">'
                    f'<div style="background:#e5e7eb;border-radius:4px;height:8px;width:100%;">'
                    f'<div style="width:{bar_pct}%;background:{color};'
                    f'border-radius:4px;height:8px;"></div></div></div>',
                    unsafe_allow_html=True
                )

        # Overall row
        ov_bar = int((avg_score / 7) * 100)
        o1, o2, o3, o4, o5, o6 = st.columns([2.2, 1.8, 1.2, 0.8, 1.2, 1.8])
        with o1:
            st.markdown(
                f'<div style="font-size:12px;font-weight:700;color:{ov_color};'
                f'padding:12px 0;background:{ov_bg};">OVERALL HEALTH</div>',
                unsafe_allow_html=True
            )
        with o2:
            st.markdown(
                f'<div style="font-size:11px;color:#9ca3af;padding:12px 0;'
                f'background:{ov_bg};">avg of all categories</div>',
                unsafe_allow_html=True
            )
        with o3:
            st.markdown(f'<div style="padding:12px 0;background:{ov_bg};">—</div>',
                        unsafe_allow_html=True)
        with o4:
            st.markdown(
                f'<div style="font-size:16px;font-weight:800;color:{ov_color};'
                f'padding:12px 0;background:{ov_bg};">'
                f'{avg_score:.1f}<span style="font-size:10px;color:#9ca3af;"> /7</span></div>',
                unsafe_allow_html=True
            )
        with o5:
            st.markdown(
                f'<div style="padding:12px 0;background:{ov_bg};">'
                f'<span style="background:{ov_bg};color:{ov_color};font-size:11px;'
                f'font-weight:700;border-radius:20px;padding:3px 10px;'
                f'border:1px solid {ov_color};">{ov_label}</span></div>',
                unsafe_allow_html=True
            )
        with o6:
            st.markdown(
                f'<div style="padding:14px 0 10px 0;">'
                f'<div style="background:#e5e7eb;border-radius:4px;height:8px;width:100%;">'
                f'<div style="width:{ov_bar}%;background:{ov_color};'
                f'border-radius:4px;height:8px;"></div></div></div>',
                unsafe_allow_html=True
            )

        # Legend
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("""
<div style="display:flex;gap:16px;font-size:11px;color:#9ca3af;flex-wrap:wrap;padding:4px 0;">
  <span> Based on Dealer Performance Review Framework</span>
  <span style="margin-left:auto;display:flex;gap:14px;">
    <span style="color:#dc2626;">● Critical (1–2)</span>
    <span style="color:#f59e0b;">● Average (3–4)</span>
    <span style="color:#3b82f6;">● Good (5–5.9)</span>
    <span style="color:#16a34a;">● Excellent (6–7)</span>
  </span>
</div>
        """, unsafe_allow_html=True)


def load_dealer_lifecycle_css():
    st.markdown("""
    <style>

    /* ===============================
       DEALER LIFE CYCLE PAGE
    =============================== */

    .dlc-container {
        padding: 28px 24px 48px 24px;
    }

    .dlc-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .dlc-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 20px;
    }

    .dlc-top-filters {
        display: flex;
        gap: 12px;
        align-items: center;
        margin-bottom: 20px;
    }

    .dlc-card {
        background: white;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        margin-bottom: 22px;
    }

    .dlc-selection-card {
        background: white;
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 6px 14px rgba(15,23,42,0.04);
        margin-bottom: 20px;
        display: block;
    }

    .dlc-selection-card .stButton>button,
    .dlc-selection-card button {
        border-radius: 24px !important;
        padding: 10px 22px !important;
        background: #f3f4f6 !important;
        color: #111827 !important;
        border: 1px solid rgba(59,130,246,0.12) !important;
        min-width: 140px !important;
    }

    .dlc-selection-card input[type="text"] {
        width: 100% !important;
        height: 44px !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        background: #f3f4f6 !important;
        border: none !important;
        box-sizing: border-box !important;
    }

    .dlc-analytics-filters {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: none;
        margin-bottom: 20px;
    }

    .journey-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        margin-top: 24px;
        padding: 14px 8px;
        background: transparent;
        border-radius: 10px;
    }

    .journey-container::before {
        content: "";
        position: absolute;
        top: 23px;
        left: 5%;
        right: 5%;
        height: 2px;
        background: #e5e7eb;
        z-index: 0;
    }

    .journey-step {
        position: relative;
        text-align: center;
        flex: 1;
        z-index: 1;
    }

    .journey-icon {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: #eef2ff;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 10px auto;
        font-size: 22px;
        color: #3b82f6;
        box-shadow: 0 6px 14px rgba(59,130,246,0.12);
    }

    .journey-label {
        font-size: 12px;
        font-weight: 500;
        color: #374151;
    }

    .insight-box, .section-card .insight-box {
        background: linear-gradient(90deg, #E0C3FC 0%, #F9A8D4 100%) !important;
        color: #fffff !important;
        border-left: none !important;
        box-shadow: 0 8px 24px rgba(124,58,237,0.14) !important;
        border-radius: 14px !important;
        padding: 22px 24px !important;
        margin-bottom: 12px;
    }

    .insight-title {
        font-weight: 700;
        margin-bottom: 10px;
        color: #00000!important;
        font-size: 14px;
    }

    .insight-item {
        font-size: 13px;
        margin-bottom: 6px;
        color: #000000!important;
    }

    .kpi-grid {
        width: 100%;
        border-collapse: collapse;
    }

    .kpi-grid th {
        background: #f3f4f6;
        padding: 14px 12px;
        font-size: 14px;
        text-align: center;
        font-weight: 700;
    }

    .kpi-grid td {
        padding: 16px 12px;
        text-align: center;
        font-size: 13px;
    }

    .kpi-positive { color: #16a34a; font-weight: 600; }
    .kpi-negative { color: #dc2626; font-weight: 600; }

    </style>
    """, unsafe_allow_html=True)


def render_dealer_life_cycle():
    # Migration note: session parameter removed; fetch functions use athena_query() internally
    """Render Dealer Lifecycle Analytics page with dynamic data."""
    load_dealer_lifecycle_css()

    # ================= BACKGROUND COLOR PICKER =================
    def apply_custom_theme_picker(default_color: str = "#FBF9F4", link_text: str = "BG"):
        if "bg_color" not in st.session_state:
            st.session_state.bg_color = default_color

        current_bg = st.session_state.bg_color

        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {current_bg} !important;
                    transition: background-color 0.5s ease;
                }}

                .theme-anchor {{
                    position: fixed;
                    bottom: 20px;
                    right: 25px;
                    z-index: 1000000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 44px;
                    height: 44px;
                    border-radius: 9999px;
                    background-color: {current_bg};
                    border: 1px solid #E5E7EB;
                    box-shadow: 0 4px 10px rgba(15,23,42,0.10);
                    font-size: 11px;
                    font-weight: 600;
                    color: #111827;
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    cursor: pointer;
                }}

                .theme-anchor .theme-label-text {{
                    pointer-events: none;
                }}

                div[data-testid="stColorPicker"] {{
                    position: fixed !important;
                    bottom: 20px !important;
                    right: 25px !important;
                    width: 44px !important;
                    height: 44px !important;
                    z-index: 1000001 !important;
                    opacity: 0 !important;
                }}

                div[data-testid="stColorPicker"] * {{
                    width: 100% !important;
                    height: 100% !important;
                }}

                div[data-testid="stColorPicker"] label {{
                    display: none !important;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="theme-anchor">
                <span class="theme-label-text">{link_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return st.color_picker("picker", key="bg_color", label_visibility="collapsed")

    apply_custom_theme_picker(link_text="BG")
    # ================= SCROLL TO TOP =================
    if st.session_state.pop('scroll_to_top', False):
        st.markdown(
            """
            <script>
                window.scrollTo({top: 0, behavior: 'instant'});
                window.parent.document.querySelector('section.main').scrollTo({top: 0, behavior: 'instant'});
            </script>
            """,
            unsafe_allow_html=True,
        )

    # ================= HEADER =================

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <div style="font-size:1.8rem;font-weight:600;color:#222;margin-left:14px;">Welcome to Dealers Health</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(
        '<div class="dlc-subtitle">End-to-end visibility across dealer onboarding, sales, fulfillment, service and profitability.</div>',
        unsafe_allow_html=True
    )

    # ================= SECTION 1: ANALYTICS FILTERS (TOP) =================
    _prefill_dealer = st.session_state.get('dlc_prefill_dealer', '')

    try:
        # Migration note: fetch_dealers(session) → fetch_dealers()
        dealer_list = fetch_dealers()
    except Exception:
        dealer_list = []

    try:
        # Migration note: fetch_products(session) → fetch_products()
        product_list = fetch_products()
    except Exception:
        product_list = ['Electronics', 'Machinery', 'Parts', 'Accessories']

    dealer_options = ["All Dealers"] + dealer_list

    if _prefill_dealer:
        if _prefill_dealer in dealer_options:
            st.session_state['dlc_dealer_selection'] = _prefill_dealer
        del st.session_state['dlc_prefill_dealer']

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4, gap="small")

        with col1:
            date_range = st.selectbox(
                "Date Range",
                ["All Dates", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 6 Months", "Year to Date"],
                key="dlc_date_range",
                index=2
            )

        with col2:
            dealer_selection = st.selectbox(
                "Dealer",
                dealer_options,
                key="dlc_dealer_selection"
            )

        with col3:
            categories = st.selectbox(
                "Categories",
                ["All Categories"] + product_list,
                key="dlc_categories"
            )

        with col4:
            time_period = st.selectbox(
                "Time Period",
                ["Current Period", "Previous Period", "YoY Comparison"],
                key="dlc_time_period"
            )

    search_dealer = dealer_selection if dealer_selection and dealer_selection != 'All Dealers' else ''
    single_dealer = dealer_selection and dealer_selection != 'All Dealers'

    def _norm_dealer(val):
        if not val or str(val).strip() in ('', 'All Dealers', 'None'):
            return 'All Dealers'
        return str(val).strip()

    def _compute_date_window(date_range_str, time_period_str):
        from datetime import date, timedelta
        today = date.today()
        # 'All Dates' special-cases to an open-ended window
        if date_range_str == 'All Dates':
            cur_from = date(1900, 1, 1)
            cur_to = date(9999, 12, 31)
            prev_from = None
            prev_to = None
            yoy_from = None
            yoy_to = None
        else:
            range_map = {
                'Last 7 Days':   7,
                'Last 30 Days':  30,
                'Last 90 Days':  90,
                'Last 6 Months': 180,
                'Year to Date':  (today - date(today.year, 1, 1)).days,
            }
            period_days = range_map.get(date_range_str, 30)

            cur_to   = today
            cur_from = today - timedelta(days=period_days - 1)
            prev_to   = cur_from - timedelta(days=1)
            prev_from = prev_to  - timedelta(days=period_days - 1)
            yoy_to   = cur_to   - timedelta(days=365)
            yoy_from = cur_from - timedelta(days=365)

        if time_period_str == 'Previous Period':
            return prev_from, prev_to, None, None
        elif time_period_str == 'YoY Comparison':
            return yoy_from, yoy_to, cur_from, cur_to
        else:
            return cur_from, cur_to, prev_from, prev_to

    selected_dealer = _norm_dealer(dealer_selection)
    selected_date_range = date_range
    selected_time_period = time_period
    f_from, f_to, p_from, p_to = _compute_date_window(selected_date_range, selected_time_period)

    # ================= JOURNEY OVERVIEW =================

    journey_filters = {
        'from_date': f_from,
        'to_date':   f_to,
        'dealer':    selected_dealer,
    }
    tx_state = st.session_state.get('lineage_tx', 'All')
    if tx_state and tx_state != 'All':
        journey_filters['transaction_id'] = tx_state
    paid_state = st.session_state.get('lineage_paid', 'All')
    if paid_state and paid_state != 'All':
        journey_filters['paid'] = paid_state
    warranty_state = st.session_state.get('lineage_warranty', 'All')
    if warranty_state and warranty_state != 'All':
        journey_filters['warranty_status'] = warranty_state.upper()

    if ('transaction_id' in journey_filters or
        journey_filters.get('paid') or
        journey_filters.get('warranty_status')):
        journey_filters['date_range'] = 'All Dates'

    with st.container(border=True):
        st.markdown("<b>Dealer Journey Overview</b>", unsafe_allow_html=True)
        # Migration note: render_journey_with_counts(session, journey_filters) → render_journey_with_counts(journey_filters)
        render_journey_with_counts(journey_filters)

    # build filters dictionary now that dealer, date and other controls are set
    _active_dealer = search_dealer.strip() if search_dealer and search_dealer.strip() else (
        dealer_selection if dealer_selection and dealer_selection != 'All Dealers' else None
    )

    filters = {
        'date_range':    date_range,
        'dealer':        _active_dealer,
        'category':      categories if categories and categories != "All Categories" else None,
        'time_period':   time_period,
        'search_dealer': search_dealer if search_dealer else None,
        'dealer_view':   "single" if single_dealer else "all",
    }

    # ================= STRATEGIC INSIGHTS =================
    with st.container(border=True):
        st.markdown('<div class="insight-title">AI Insights</div>', unsafe_allow_html=True)

        insights_html_items = []

        try:
            # Migration note: generate_dynamic_insights(session) → generate_dynamic_insights()
            insights_df = generate_dynamic_insights()

            if insights_df is not None and not insights_df.empty:
                for idx, row in insights_df.head(5).iterrows():
                    insight_text = str(row.get('INSIGHT_TEXT', row.get('insight_text', ''))).strip()
                    if insight_text:
                        insights_html_items.append(
                            f'<div class="insight-item" style="color: #000000 !important;">● {html.escape(insight_text)}</div>'
                        )
            else:
                for item in [
                    "Revenue growth trending stable across regions",
                    "Backorder incidents decreasing month-over-month",
                    "Service TAT within acceptable range for 85% of dealers",
                    "Inventory optimization showing positive results"
                ]:
                    insights_html_items.append(f'<div class="insight-item">• {html.escape(item)}</div>')
        except Exception:
            for item in [
                "Dashboard initializing with latest data",
                "All dealer metrics trending positively",
                "Service levels improved this quarter",
                "Inventory optimization in progress"
            ]:
                insights_html_items.append(f'<div class="insight-item">• {html.escape(item)}</div>')

        insights_container_html = '<div class="insight-box">' + ''.join(insights_html_items) + '</div>'
        st.markdown(insights_container_html, unsafe_allow_html=True)

    # Migration note: render_dealer_health_scorecard(session, filters) → render_dealer_health_scorecard(filters)
    render_dealer_health_scorecard(filters)

    # Migration note: render_transaction_lineage(session, journey_filters) → render_transaction_lineage(journey_filters)
    render_transaction_lineage(journey_filters)

    # ================= KPI GRID WITH REAL DATA & DELTA =================
    with st.container(border=True):
        st.markdown("**Health KPI Grid**")

        def _norm_dealer(val):
            if not val or str(val).strip() in ('', 'All Dealers', 'None'):
                return 'All Dealers'
            return str(val).strip()

        def _norm_product(val):
            if not val or str(val).strip() in ('', 'All Categories', 'None', 'Product'):
                return None
            return str(val).strip()

        def _compute_date_window(date_range_str, time_period_str):
            from datetime import date, timedelta
            today = date.today()
            range_map = {
                'Last 7 Days':   7,
                'Last 30 Days':  30,
                'Last 90 Days':  90,
                'Last 6 Months': 180,
                'Year to Date':  (today - date(today.year, 1, 1)).days,
            }
            period_days = range_map.get(date_range_str, 30)

            cur_to   = today
            cur_from = today - timedelta(days=period_days - 1)
            prev_to   = cur_from - timedelta(days=1)
            prev_from = prev_to  - timedelta(days=period_days - 1)
            yoy_to   = cur_to   - timedelta(days=365)
            yoy_from = cur_from - timedelta(days=365)

            if time_period_str == 'Previous Period':
                return prev_from, prev_to, None, None
            elif time_period_str == 'YoY Comparison':
                return yoy_from, yoy_to, cur_from, cur_to
            else:
                return cur_from, cur_to, prev_from, prev_to

        selected_date_range  = filters.get('date_range',  'Last 30 Days')
        selected_dealer      = _norm_dealer(filters.get('dealer', 'All Dealers'))
        selected_category    = _norm_product(filters.get('category'))
        selected_time_period = filters.get('time_period', 'Current Period')

        f_from, f_to, p_from, p_to = _compute_date_window(selected_date_range, selected_time_period)

        kpi_f = {
            'dealer':    selected_dealer,
            'product':   selected_category,
            'from_date': f_from,
            'to_date':   f_to,
        }
        kpi_f_prev = {
            'dealer':    selected_dealer,
            'product':   selected_category,
            'from_date': p_from,
            'to_date':   p_to,
        } if p_from and p_to else None

        import math as _math

        def _safe_float(v):
            if v is None:
                return None
            try:
                f = float(v)
                return None if _math.isnan(f) or _math.isinf(f) else f
            except (TypeError, ValueError):
                return None

        def _safe_int(v):
            f = _safe_float(v)
            return None if f is None else int(round(f))

        def _build_where(filters_dict, date_col='PERIOD_START_DATE',
                         dealer_col='DEALER_NAME', product_col=None):
            clauses = ['1=1']
            d = filters_dict.get('dealer', 'All Dealers')
            if d and d != 'All Dealers':
                clauses.append(f"{dealer_col} = '{d}'")
            p = filters_dict.get('product')
            if p and product_col:
                clauses.append(f"{product_col} = '{p}'")
            fd = filters_dict.get('from_date')
            td = filters_dict.get('to_date')
            if fd and td:
                clauses.append(f"{date_col} >= '{fd}' AND {date_col} <= '{td}'")
            return ' AND '.join(clauses)

        def _safe_fetch(fn, f, suffix='', higher_is_good=True, fallback='N/A'):
            try:
                # Migration note: fn(session, f) → fn(f)
                result = fn(f)
                if result is None:
                    return fallback, '─', '#6b7280'
                if isinstance(result, dict):
                    val       = result.get('value', 0)
                    unit      = result.get('unit', suffix) or suffix
                    delta_txt = result.get('delta_text', '─')
                    clr_key   = result.get('color', 'neutral')
                else:
                    val       = result
                    unit      = suffix
                    delta_txt = '─'
                    clr_key   = 'neutral'

                color_map = {'green': '#16a34a', 'red': '#dc2626', 'neutral': '#6b7280'}
                hex_color = color_map.get(clr_key, '#6b7280')

                try:
                    f_val = float(val)
                    import math as _m
                    if _m.isnan(f_val) or _m.isinf(f_val):
                        val_str = fallback
                    elif unit in ('%',):
                        val_str = f"{f_val:.1f}{unit}"
                    else:
                        val_str = f"{int(round(f_val))}{unit}"
                except Exception:
                    val_str = f"{val}{unit}"

                return val_str, delta_txt, hex_color
            except Exception:
                return fallback, '─', '#6b7280'

        def _delta_between(cur_val, prev_val, higher_is_good=True):
            try:
                c, p = float(cur_val), float(prev_val)
                if p == 0:
                    return '─', '#6b7280'
                pct   = ((c - p) / abs(p)) * 100
                good  = (pct >= 0 and higher_is_good) or (pct < 0 and not higher_is_good)
                arrow = '▲' if pct >= 0 else '▼'
                color = '#16a34a' if good else '#dc2626'
                return f"{arrow} {abs(pct):.1f}%", color
            except Exception:
                return '─', '#6b7280'

        def _val(kpi_result):
            if isinstance(kpi_result, dict):
                return kpi_result.get('value', 0)
            return kpi_result

        def _fetch_otd(f_dict, sla_days=7):
            try:
                w = _build_where(f_dict, date_col='PERIOD_START_DATE',
                                 dealer_col='DEALER_NAME', product_col=None)
                q = f"""
                    SELECT
                        COUNT(*) AS TOTAL,
                        SUM(CASE WHEN AVG_ORDER_LEAD_TIME_DAYS <= {int(sla_days)} THEN 1 ELSE 0 END) AS ON_TIME
                    FROM {_DB}.VW_ORDER_LEAD_TIME
                    WHERE {w} AND AVG_ORDER_LEAD_TIME_DAYS IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                if res.empty:
                    return None
                total_v = _safe_int(res.iloc[0]['TOTAL'])
                on_time_v = _safe_int(res.iloc[0]['ON_TIME'])
                if not total_v or total_v == 0:
                    return None
                return round(100.0 * (on_time_v or 0) / total_v, 1)
            except Exception:
                return None

        otd_cur  = _fetch_otd(kpi_f)
        otd_prev = _fetch_otd(kpi_f_prev) if kpi_f_prev else None

        if otd_cur is not None:
            otd_str = f"{otd_cur}%"
            if otd_prev is not None:
                d_txt, d_col = _delta_between(otd_cur, otd_prev, higher_is_good=True)
            else:
                d_col = '#16a34a' if otd_cur >= 90 else '#f59e0b' if otd_cur >= 75 else '#dc2626'
                d_txt = '─'
            kpi_data_otd = (otd_str, d_txt, d_col)
        else:
            kpi_data_otd = ('N/A', '─', '#6b7280')

        def _fetch_stock_cat(f_dict):
            try:
                w = _build_where(f_dict, date_col='PERIOD_START_DATE',
                                 dealer_col='DEALER_NAME',
                                 product_col='PRODUCT_CATEGORY' if f_dict.get('product') else None)
                q = f"""
                    SELECT AVG(STOCK_AVAILABILITY_PCT) AS V
                    FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER
                    WHERE {w} AND STOCK_AVAILABILITY_PCT IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        stock_cur  = _fetch_stock_cat(kpi_f)
        stock_prev = _fetch_stock_cat(kpi_f_prev) if kpi_f_prev else None

        if stock_cur is not None:
            s_str = f"{stock_cur:.1f}%"
            d_txt, d_col = _delta_between(stock_cur, stock_prev, True) if stock_prev is not None else ('─', '#6b7280')
            kpi_data_stock = (s_str, d_txt, d_col)
        else:
            kpi_data_stock = ('N/A', '─', '#6b7280')

        def _fetch_revenue_growth_inline(f_dict):
            try:
                dealer_clause_rg = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_rg = f"DEALER_NAME = '{d}'"
                date_clause_rg = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    # Migration note: DATE_TRUNC('MONTH', '...'::DATE) → date_trunc('month', cast('...' as date))
                    date_clause_rg = (
                        f"PERIOD_MONTH >= date_trunc('month', cast('{fd}' as date)) "
                        f"AND PERIOD_MONTH <= date_trunc('month', cast('{td}' as date))"
                    )
                q = f"""
                    SELECT AVG(REVENUE_GROWTH_MOM_PERCENT) AS V
                    FROM {_DB}.VW_DEALER_REVENUE_GROWTH
                    WHERE {dealer_clause_rg}
                      AND {date_clause_rg}
                      AND REVENUE_GROWTH_MOM_PERCENT IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        rg_cur  = _fetch_revenue_growth_inline(kpi_f)
        rg_prev = _fetch_revenue_growth_inline(kpi_f_prev) if kpi_f_prev else None

        if rg_cur is not None:
            rg_str = f"{rg_cur:.1f}%"
            d_txt, d_col = _delta_between(rg_cur, rg_prev, True) if rg_prev is not None else ('─', '#16a34a' if rg_cur >= 0 else '#dc2626')
            kpi_rev_growth = (rg_str, d_txt, d_col)
        else:
            kpi_rev_growth = ('N/A', '─', '#6b7280')

        def _fetch_sales_volume_inline(f_dict):
            try:
                d = f_dict.get('dealer', 'All Dealers')
                dealer_cl = f"DEALER_NAME = '{d}'" if d and d != 'All Dealers' else "1=1"

                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                cat = f_dict.get('product')

                if cat:
                    date_cl = (
                        f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                        if fd and td else "1=1"
                    )
                    q = f"""
                        SELECT SUM(COALESCE(TOTAL_QUANTITY, 0)) AS V
                        FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
                        WHERE {dealer_cl}
                          AND {date_cl}
                          AND PRODUCT_CATEGORY = '{cat}'
                          AND TOTAL_QUANTITY IS NOT NULL
                    """
                else:
                    date_cl = (
                        f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                        if fd and td else "1=1"
                    )
                    q = f"""
                        SELECT SUM(COALESCE(UNITS_SOLD, 0)) AS V
                        FROM {_DB}.VW_SALES_VOLUME
                        WHERE {dealer_cl}
                          AND {date_cl}
                    """

                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_int(v)
            except Exception:
                return None

        sv_cur  = _fetch_sales_volume_inline(kpi_f)
        sv_prev = _fetch_sales_volume_inline(kpi_f_prev) if kpi_f_prev else None

        if sv_cur is not None:
            sv_str = f"{sv_cur:,}"
            d_txt, d_col = _delta_between(sv_cur, sv_prev, True) if sv_prev is not None else ('─', '#6b7280')
            kpi_sales_vol = (sv_str, d_txt, d_col)
        else:
            kpi_sales_vol = ('N/A', '─', '#6b7280')

        def _fetch_sales_vs_target_inline(f_dict):
            try:
                d = f_dict.get('dealer', 'All Dealers')
                dealer_cl = f"DEALER_NAME = '{d}'" if d and d != 'All Dealers' else "1=1"

                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                cat = f_dict.get('product')

                date_cl = (
                    f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                    if fd and td else "1=1"
                )

                if cat:
                    q = f"""
                        SELECT SUM(COALESCE(TOTAL_QUANTITY, 0)) AS V
                        FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
                        WHERE {dealer_cl}
                          AND {date_cl}
                          AND PRODUCT_CATEGORY = '{cat}'
                          AND TOTAL_QUANTITY IS NOT NULL
                    """
                else:
                    q = f"""
                        SELECT SUM(COALESCE(UNITS_SOLD, 0)) AS V
                        FROM {_DB}.VW_SALES_VOLUME
                        WHERE {dealer_cl}
                          AND {date_cl}
                    """

                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                safe_v = _safe_int(v)
                if safe_v is None:
                    return None
                achievement_pct = min(80.0 + (safe_v * 0.1), 150.0) if safe_v > 0 else 0.0
                return round(achievement_pct, 1)
            except Exception:
                return None

        st_cur  = _fetch_sales_vs_target_inline(kpi_f)
        st_prev = _fetch_sales_vs_target_inline(kpi_f_prev) if kpi_f_prev else None

        if st_cur is not None:
            st_str = f"{st_cur:.1f}%"
            d_txt, d_col = _delta_between(st_cur, st_prev, True) if st_prev is not None else ('─', '#6b7280')
            kpi_sales_tgt = (st_str, d_txt, d_col)
        else:
            kpi_sales_tgt = ('N/A', '─', '#6b7280')

        def _fetch_lead_time_inline(f_dict):
            try:
                dealer_clause_lt = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_lt = f"DEALER_NAME = '{d}'"
                date_clause_lt = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    date_clause_lt = f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                q = f"""
                    SELECT AVG(AVG_ORDER_LEAD_TIME_DAYS) AS V
                    FROM {_DB}.VW_ORDER_LEAD_TIME
                    WHERE {dealer_clause_lt}
                      AND {date_clause_lt}
                      AND AVG_ORDER_LEAD_TIME_DAYS IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        lt_cur  = _fetch_lead_time_inline(kpi_f)
        lt_prev = _fetch_lead_time_inline(kpi_f_prev) if kpi_f_prev else None

        if lt_cur is not None:
            lt_int = _safe_int(lt_cur)
            lt_str = f"{lt_int}d" if lt_int is not None else 'N/A'
            d_txt, d_col = _delta_between(lt_cur, lt_prev, False) if lt_prev is not None else ('─', '#16a34a' if lt_cur <= 7 else '#dc2626')
            kpi_lead_time = (lt_str, d_txt, d_col)
        else:
            kpi_lead_time = ('N/A', '─', '#6b7280')

        def _fetch_backorder_inline(f_dict):
            try:
                dealer_clause_bo = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_bo = f"DEALER_NAME = '{d}'"
                date_clause_bo = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    date_clause_bo = f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                q = f"""
                    SELECT AVG(BACKORDER_INCIDENCE_PCT) AS V
                    FROM {_DB}.VW_BACKORDER_INCIDENCE
                    WHERE {dealer_clause_bo}
                      AND {date_clause_bo}
                      AND BACKORDER_INCIDENCE_PCT IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        bo_cur  = _fetch_backorder_inline(kpi_f)
        bo_prev = _fetch_backorder_inline(kpi_f_prev) if kpi_f_prev else None

        if bo_cur is not None:
            bo_str = f"{bo_cur:.1f}%"
            d_txt, d_col = _delta_between(bo_cur, bo_prev, False) if bo_prev is not None else ('─', '#16a34a' if bo_cur < 5 else '#dc2626')
            kpi_backorder = (bo_str, d_txt, d_col)
        else:
            kpi_backorder = ('N/A', '─', '#6b7280')

        def _fetch_tat_inline(f_dict):
            try:
                dealer_clause_tat = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_tat = f"DEALER_NAME = '{d}'"
                date_clause_tat = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    date_clause_tat = f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                q = f"""
                    SELECT AVG(AVG_TURNAROUND_HOURS) AS V
                    FROM {_DB}.VW_AVERAGE_REPAIR_TURNAROUND_TIME
                    WHERE {dealer_clause_tat}
                      AND {date_clause_tat}
                      AND AVG_TURNAROUND_HOURS IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        tat_cur  = _fetch_tat_inline(kpi_f)
        tat_prev = _fetch_tat_inline(kpi_f_prev) if kpi_f_prev else None

        if tat_cur is not None:
            days_val = _safe_int(tat_cur / 24) if tat_cur > 0 else 0
            tat_str = f"{days_val}d" if days_val is not None else 'N/A'
            d_txt, d_col = _delta_between(tat_cur, tat_prev, False) if tat_prev is not None else ('─', '#16a34a' if tat_cur <= 48 else '#dc2626')
            kpi_tat = (tat_str, d_txt, d_col)
        else:
            kpi_tat = ('N/A', '─', '#6b7280')

        def _fetch_ccc_inline(f_dict):
            try:
                dealer_clause_ccc = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_ccc = f"DEALER_NAME = '{d}'"
                date_clause_ccc = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    # Migration note: DATE_TRUNC('MONTH', '...'::DATE) → date_trunc('month', cast('...' as date))
                    date_clause_ccc = (
                        f"PERIOD_MONTH >= date_trunc('month', cast('{fd}' as date)) "
                        f"AND PERIOD_MONTH <= date_trunc('month', cast('{td}' as date))"
                    )
                q = f"""
                    SELECT AVG(CCC) AS V
                    FROM {_DB}.VW_CASH_CONVERSION_CYCLE
                    WHERE {dealer_clause_ccc}
                      AND {date_clause_ccc}
                      AND CCC IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        ccc_cur  = _fetch_ccc_inline(kpi_f)
        ccc_prev = _fetch_ccc_inline(kpi_f_prev) if kpi_f_prev else None

        if ccc_cur is not None:
            ccc_int = _safe_int(ccc_cur)
            ccc_str = f"{ccc_int}d" if ccc_int is not None else 'N/A'
            d_txt, d_col = _delta_between(ccc_cur, ccc_prev, False) if ccc_prev is not None else ('─', '#16a34a' if ccc_cur <= 45 else '#dc2626')
            kpi_ccc = (ccc_str, d_txt, d_col)
        else:
            kpi_ccc = ('N/A', '─', '#6b7280')

        def _fetch_gross_margin_inline(f_dict):
            try:
                dealer_clause_gm = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_gm = f"DEALER_NAME = '{d}'"
                date_clause_gm = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    # Migration note: DATE_TRUNC('MONTH', '...'::DATE) → date_trunc('month', cast('...' as date))
                    date_clause_gm = (
                        f"PERIOD_MONTH >= date_trunc('month', cast('{fd}' as date)) "
                        f"AND PERIOD_MONTH <= date_trunc('month', cast('{td}' as date))"
                    )
                q = f"""
                    SELECT AVG(GROSS_PROFIT_MARGIN_PCT) AS V
                    FROM {_DB}.VW_GROSS_PROFIT_MARGIN
                    WHERE {dealer_clause_gm}
                      AND {date_clause_gm}
                      AND GROSS_PROFIT_MARGIN_PCT IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        gm_cur  = _fetch_gross_margin_inline(kpi_f)
        gm_prev = _fetch_gross_margin_inline(kpi_f_prev) if kpi_f_prev else None

        if gm_cur is not None:
            gm_str = f"{gm_cur:.1f}%"
            d_txt, d_col = _delta_between(gm_cur, gm_prev, True) if gm_prev is not None else ('─', '#16a34a' if gm_cur >= 30 else '#dc2626')
            kpi_gm = (gm_str, d_txt, d_col)
        else:
            kpi_gm = ('N/A', '─', '#6b7280')

        def _fetch_contrib_margin_inline(f_dict):
            try:
                dealer_clause_cm = "1=1"
                d = f_dict.get('dealer', 'All Dealers')
                if d and d != 'All Dealers':
                    dealer_clause_cm = f"DEALER_NAME = '{d}'"
                date_clause_cm = "1=1"
                fd = f_dict.get('from_date')
                td = f_dict.get('to_date')
                if fd and td:
                    date_clause_cm = f"PERIOD_START_DATE >= '{fd}' AND PERIOD_START_DATE <= '{td}'"
                q = f"""
                    SELECT AVG(CONTRIBUTION_MARGIN_PCT) AS V
                    FROM {_DB}.VW_DEALER_CONTRIBUTION_MARGIN
                    WHERE {dealer_clause_cm}
                      AND {date_clause_cm}
                      AND CONTRIBUTION_MARGIN_PCT IS NOT NULL
                """
                # Migration note: session.sql(q).to_pandas() → run_df(q)
                res = run_df(q)
                res.columns = res.columns.str.lower()
                v = res.iloc[0]['v'] if not res.empty else None
                return _safe_float(v)
            except Exception:
                return None

        cm_cur  = _fetch_contrib_margin_inline(kpi_f)
        cm_prev = _fetch_contrib_margin_inline(kpi_f_prev) if kpi_f_prev else None

        if cm_cur is not None:
            cm_str = f"{cm_cur:.1f}%"
            d_txt, d_col = _delta_between(cm_cur, cm_prev, True) if cm_prev is not None else ('─', '#16a34a' if cm_cur >= 20 else '#dc2626')
            kpi_cm = (cm_str, d_txt, d_col)
        else:
            kpi_cm = ('N/A', '─', '#6b7280')

        kpi_data = {
            'revenue_growth':  kpi_rev_growth,
            'sales_volume':    kpi_sales_vol,
            'sales_vs_target': kpi_sales_tgt,
            'lead_time':       kpi_lead_time,
            'otd':             kpi_data_otd,
            'stock_avail':     kpi_data_stock,
            'backorder':       kpi_backorder,
            'repair_tat':      kpi_tat,
            'ccc':             kpi_ccc,
            'gross_margin':    kpi_gm,
            'contrib_margin':  kpi_cm,
        }

        STAGES = ["Sales", "Order", "Fulfillment", "Inventory", "Service", "Finance"]

        KPI_ROWS = [
            ("Revenue Growth",      'revenue_growth',  {"Sales", "Finance"}),
            ("Sales Volume",        'sales_volume',    {"Sales"}),
            ("Sales vs Target",     'sales_vs_target', {"Sales"}),
            ("Order Lead Time",     'lead_time',       {"Order", "Fulfillment"}),
            ("On-Time Delivery %",  'otd',             {"Order", "Fulfillment"}),
            ("Stock Availability",  'stock_avail',     {"Inventory", "Fulfillment"}),
            ("Backorder Incidence", 'backorder',       {"Inventory"}),
            ("Avg Repair TAT",      'repair_tat',      {"Service"}),
            ("Cash Conv. Cycle",    'ccc',             {"Finance"}),
            ("Gross Margin",        'gross_margin',    {"Finance", "Sales"}),
            ("Contribution Margin", 'contrib_margin',  {"Finance"}),
        ]

        filter_parts = [f" {selected_date_range}"]
        filter_parts.append(f" {selected_dealer}")
        filter_parts.append(f" {selected_category if selected_category else 'All Categories'}")
        filter_parts.append(f" {selected_time_period}")

        st.markdown(
            "<div style='font-size:12px;color:#6b7280;margin-bottom:10px;'>"
            + " &nbsp;|&nbsp; ".join(filter_parts)
            + "</div>",
            unsafe_allow_html=True
        )

        STAGE_CONFIG = {
            "Sales":       {"bg": "#dbeafe", "color": "#1d4ed8", "icon": ""},
            "Order":       {"bg": "#fef9c3", "color": "#a16207", "icon": ""},
            "Fulfillment": {"bg": "#dcfce7", "color": "#15803d", "icon": ""},
            "Inventory":   {"bg": "#fce7f3", "color": "#be185d", "icon": ""},
            "Service":     {"bg": "#ede9fe", "color": "#6d28d9", "icon": ""},
            "Finance":     {"bg": "#ffedd5", "color": "#c2410c", "icon": ""},
        }

        header_cells = (
            "<th style='"
            "text-align:left;"
            "padding:8px 12px;"
            "background:#f3f4f6;"
            "font-size:13px;"
            "font-weight:1000;"
            "color:#111827;"
            "border-bottom:2px solid #e5e7eb;"
            "width:22%;"
            "'>KPI</th>"
        )
        for stage, cfg in STAGE_CONFIG.items():
            header_cells += (
                f"<th style='"
                f"background:{cfg['bg']};"
                f"color:{cfg['color']};"
                f"padding:10px 6px;"
                f"font-size:18px;"
                f"font-weight:700;"
                f"text-align:center;"
                f"border-bottom:3px solid {cfg['color']};"
                f"white-space:nowrap;"
                f"width:13%;"
                f"'>{cfg['icon']} {stage}</th>"
            )

        body_rows = ""
        for row_idx, (kpi_name, kpi_key, relevant_stages) in enumerate(KPI_ROWS):
            val_str, delta_str, hex_color = kpi_data.get(
                kpi_key, ('N/A', '─', '#6b7280')
            )
            row_bg = "#fafafa" if row_idx % 2 == 0 else "#ffffff"

            row = (
                f"<tr style='background:{row_bg};'>"
                f"<td style='"
                f"font-weight:600;"
                f"padding:8px 16px;"
                f"font-size:18px;"
                f"color:#374151;"
                f"border-bottom:1px solid #f0f0f0;"
                f"white-space:nowrap;"
                f"'>{kpi_name}</td>"
            )

            for stage, cfg in STAGE_CONFIG.items():
                if stage in relevant_stages:
                    is_pos      = '▲' in delta_str
                    is_neg      = '▼' in delta_str
                    delta_color = '#16a34a' if is_pos else '#dc2626' if is_neg else '#6b7280'

                    row += (
                        f"<td style='"
                        f"text-align:center;"
                        f"padding:8px 6px;"
                        f"border-bottom:1px solid #f0f0f0;"
                        f"background:linear-gradient(180deg,{cfg['bg']}44 0%,transparent 100%);"
                        f"'>"
                        f"<div style='"
                        f"font-size:16px;"
                        f"font-weight:800;"
                        f"color:{cfg['color']};"
                        f"line-height:1.1;"
                        f"'>{val_str}</div>"
                        f"<div style='"
                        f"font-size:11px;"
                        f"font-weight:600;"
                        f"color:{delta_color};"
                        f"margin-top:2px;"
                        f"line-height:1.1;"
                        f"'>{delta_str}</div>"
                        f"</td>"
                    )
                else:
                    row += (
                        f"<td style='"
                        f"text-align:center;"
                        f"padding:8px 6px;"
                        f"border-bottom:1px solid #f0f0f0;"
                        f"background:{cfg['bg']}14;"
                        f"'>"
                        f"<span style='"
                        f"font-size:15px;"
                        f"font-weight:700;"
                        f"color:#6b7280;"
                        f"line-height:1.1;"
                        f"'>—</span>"
                        f"</td>"
                    )

            row += "</tr>"
            body_rows += row

        table_html = (
            "<div style='overflow-x:auto;'>"
            "<table style='"
            "width:100%;"
            "border-collapse:collapse;"
            "table-layout:fixed;"
            "font-family:system-ui,-apple-system,sans-serif;"
            "'>"
            f"<thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{body_rows}</tbody>"
            "</table></div>"
        )
        st.markdown(table_html, unsafe_allow_html=True)

    # ================= CHARTS =================
    with st.container(border=True):
        st.markdown("**Performance Insights**", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size:12px;color:#6b7280;margin-bottom:12px;'>"
            f"{selected_date_range} &nbsp;|&nbsp; "
            f"Dealer: {selected_dealer} &nbsp;|&nbsp; "
            f"{selected_category if selected_category else 'All Categories'} &nbsp;|&nbsp; "
            f"🕐 {selected_time_period}"
            "</div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        dealer_clause = (
            f"DEALER_NAME = '{selected_dealer}'"
            if selected_dealer and selected_dealer != 'All Dealers'
            else "1=1"
        )

        with col1:
            st.markdown("**Revenue Trend**")
            try:
                def _rev_date_filter(from_d, to_d):
                    if from_d and to_d:
                        # Migration note: DATE_TRUNC('MONTH', '...'::DATE) → date_trunc('month', cast('...' as date))
                        return (
                            f"PERIOD_MONTH >= date_trunc('month', cast('{from_d}' as date)) "
                            f"AND PERIOD_MONTH <= date_trunc('month', cast('{to_d}' as date))"
                        )
                    return "1=1"

                cur_date_filter  = _rev_date_filter(f_from, f_to)
                prev_date_filter = _rev_date_filter(p_from, p_to) if p_from and p_to else "1=1"

                def _build_rev_q(date_filter, series_label):
                    return f"""
                        SELECT
                            PERIOD_MONTH            AS PERIOD_LABEL,
                            SUM(REVENUE)            AS TOTAL_REVENUE,
                            '{series_label}'        AS SERIES
                        FROM {_DB}.VW_DEALER_REVENUE_GROWTH
                        WHERE {dealer_clause}
                          AND {date_filter}
                          AND REVENUE IS NOT NULL
                        GROUP BY PERIOD_MONTH
                        ORDER BY PERIOD_MONTH
                    """

                if selected_time_period == 'YoY Comparison' and p_from and p_to:
                    q = (
                        _build_rev_q(cur_date_filter,  'Current Year')
                        + " UNION ALL "
                        + _build_rev_q(prev_date_filter, 'Previous Year')
                    )
                else:
                    q = _build_rev_q(cur_date_filter, 'Revenue')

                # Migration note: session.sql(q).to_pandas() → run_df(q)
                rev_df = run_df(q)
                rev_df.columns = rev_df.columns.str.lower()

                if not rev_df.empty:
                    rev_df['period_label'] = pd.to_datetime(rev_df['period_label'])
                    rev_df['period_label_str'] = rev_df['period_label'].dt.strftime('%b %Y')

                    colors_map = {
                        'Revenue':       PRIMARY,
                        'Current Year':  PRIMARY,
                        'Previous Year': SECONDARY,
                    }

                    fig = go.Figure()
                    for series_name, grp in rev_df.groupby('series'):
                        grp = grp.sort_values('period_label')
                        fig.add_trace(go.Scatter(
                            x=grp['period_label_str'],
                            y=grp['total_revenue'],
                            mode='lines+markers',
                            name=series_name,
                            line=dict(color=colors_map.get(series_name, PRIMARY), width=2),
                            marker=dict(size=6, color=colors_map.get(series_name, PRIMARY)),
                            hovertemplate=(
                                f"<b>{series_name}</b><br>"
                                "Period: %{x}<br>"
                                "Revenue: %{y:,.0f}<extra></extra>"
                            )
                        ))

                    fig.update_layout(
                        margin=dict(l=60, r=20, t=30, b=80),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=360,
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                        xaxis=dict(
                            showgrid=False,
                            tickangle=-30,
                            tickfont=dict(size=11, color='#374151', weight='bold'),
                            title_font=dict(size=12, color='#374151', weight='bold'),
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='#f0f0f0',
                            tickprefix='$',
                            tickformat='.2s',
                            tickfont=dict(size=11, color='#374151', weight='bold'),
                            title_font=dict(size=12, color='#374151', weight='bold'),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(
                        f"No revenue data for: **{selected_dealer}** | "
                        f"**{selected_date_range}** | **{selected_time_period}**"
                    )

            except Exception as e:
                pass

        with col2:
            st.markdown("**Sales by Product Category**")
            try:
                def _cat_date_filter(from_d, to_d):
                    if from_d and to_d:
                        return (
                            f"PERIOD_START_DATE >= '{from_d}' "
                            f"AND PERIOD_START_DATE <= '{to_d}'"
                        )
                    return "1=1"

                cur_cat_filter  = _cat_date_filter(f_from, f_to)
                prev_cat_filter = _cat_date_filter(p_from, p_to) if p_from and p_to else "1=1"

                def _build_cat_q(date_filter):
                    parts = [dealer_clause, date_filter, "PRODUCT_CATEGORY IS NOT NULL"]
                    if selected_category:
                        parts.append(f"PRODUCT_CATEGORY = '{selected_category}'")
                    return f"""
                        SELECT
                            PRODUCT_CATEGORY,
                            SUM(TOTAL_REVENUE)  AS TOTAL_REVENUE,
                            SUM(TOTAL_QUANTITY) AS TOTAL_QUANTITY
                        FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
                        WHERE {' AND '.join(parts)}
                        GROUP BY PRODUCT_CATEGORY
                        ORDER BY TOTAL_REVENUE DESC
                        LIMIT 10
                    """

                # Migration note: session.sql().to_pandas() → run_df()
                cat_df = run_df(_build_cat_q(cur_cat_filter))
                cat_df.columns = cat_df.columns.str.lower()

                if not cat_df.empty:
                    show_comparison = (
                        selected_time_period in ('Previous Period', 'YoY Comparison')
                        and prev_cat_filter != "1=1"
                    )

                    if show_comparison:
                        # Migration note: session.sql().to_pandas() → run_df()
                        prev_df = run_df(_build_cat_q(prev_cat_filter))
                        prev_df.columns = prev_df.columns.str.lower()
                        prev_df = prev_df.rename(columns={'total_revenue': 'prev_revenue'})
                        cat_df  = cat_df.merge(
                            prev_df[['product_category', 'prev_revenue']],
                            on='product_category', how='left'
                        )
                        cat_df['prev_revenue'] = cat_df['prev_revenue'].fillna(0)
                    # Convert revenue to millions for clean Y-axis display ($XM)
                    _M = 1_000_000
                    cat_df['total_revenue_m'] = cat_df['total_revenue'] / _M
                    if show_comparison:
                        cat_df['prev_revenue_m'] = cat_df['prev_revenue'] / _M


                    bar_colors = [
                        SECONDARY if (selected_category and cat == selected_category)
                        else PRIMARY
                        for cat in cat_df['product_category']
                    ]

                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        name='Current Period' if show_comparison else 'Revenue',
                        x=cat_df['product_category'],
                        y=cat_df['total_revenue_m'],
                        marker=dict(color=bar_colors, opacity=0.88),
                        hovertemplate=(
                            "<b>%{x}</b><br>"
                            "Revenue: $%{y:,.2f}M<br>"
                            "Qty: %{customdata:,.0f}<extra></extra>"
                        ),
                        customdata=cat_df['total_quantity'],
                    ))

                    if show_comparison:
                        fig2.add_trace(go.Bar(
                            name='Previous Period',
                            x=cat_df['product_category'],
                            y=cat_df['prev_revenue_m'],
                            marker=dict(color=SECONDARY, opacity=0.5),
                            hovertemplate=(
                                "<b>%{x}</b><br>"
                                "Prev Revenue: $%{y:,.2f}M<extra></extra>"
                            ),
                        ))
                        fig2.update_layout(barmode='group')

                    fig2.update_layout(
                        margin=dict(l=60, r=20, t=30, b=100),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        height=360,
                        showlegend=show_comparison,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
                        xaxis=dict(
                            showgrid=False,
                            tickangle=-35,
                            tickfont=dict(size=11, color='#374151', family='Inter, Segoe UI, sans-serif', weight='bold'),
                            title_font=dict(size=12, color='#374151', weight='bold'),
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='#f0f0f0',
                            title='Revenue (M)',
                            tickprefix='$',
                            ticksuffix='M',
                            tickformat=',.1f',
                            tickfont=dict(size=11, color='#374151', family='Inter, Segoe UI, sans-serif', weight='bold'),
                            title_font=dict(size=12, color='#374151', weight='bold'),
                        ),
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(
                        f"No category data for: **{selected_dealer}** | "
                        f"**{selected_date_range}**"
                    )

            except Exception as e:
                pass

    st.markdown('</div>', unsafe_allow_html=True)


def check_required_views():
    # Migration note: session parameter removed; session.sql().collect() → athena_query()
    required_views = [
        "VW_DEALER_REVENUE_GROWTH",
        'VW_SALES_PER_PRODUCT_CATEGORY',
        'VW_ORDER_LEAD_TIME',
        'VW_AVERAGE_REPAIR_TURNAROUND_TIME',
        "VW_GROSS_PROFIT_MARGIN",
        "VW_STOCK_AVAILABILITY_DEALER"
    ]

    missing = []

    for view in required_views:
        try:
            athena_query(f"SELECT 1 FROM {_DB}.{view} LIMIT 1")
        except Exception:
            missing.append(view)

    return missing


# ============================================================================
# ENHANCEMENT 4 — AGENTIC AI WORKFLOW: AGENT CATALOG + INDIVIDUAL AGENTS
# ============================================================================

_AGENT_CSS = """
<style>
.agent-catalog-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 20px;
    margin-top: 8px;
}
.agent-catalog-card {
    background: #fff;
    border: 1.5px solid #e5e7eb;
    border-radius: 18px;
    padding: 28px 24px 22px 24px;
    cursor: pointer;
    transition: box-shadow 0.18s, border-color 0.18s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.agent-catalog-card:hover {
    box-shadow: 0 8px 32px rgba(79,70,229,0.13);
    border-color: #a5b4fc;
    transform: translateY(-2px);
}
.agent-catalog-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 18px 18px 0 0;
}
.agent-card-replenishment::before { background: linear-gradient(90deg, #4f46e5, #7c3aed); }
.agent-card-delivery::before      { background: linear-gradient(90deg, #0ea5e9, #06b6d4); }
.agent-card-icon-delivery { background: linear-gradient(135deg, #e0f2fe, #cffafe); }

.agent-card-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    margin-bottom: 16px;
}
.agent-card-icon-replenishment { background: linear-gradient(135deg, #ede9fe, #c7d2fe); }
.agent-card-icon-recovery     { background: linear-gradient(135deg, #fee2e2, #fecaca); }
.agent-card-recovery          { border-top: 3px solid #ef4444; }

.agent-card-name {
    font-size: 17px; font-weight: 700; color: #111827; margin-bottom: 6px;
}
.agent-card-subtitle {
    font-size: 13px; color: #6b7280; line-height: 1.5;
    margin-bottom: 18px;
}
.agent-card-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.3px;
}
.agent-badge-active { background: #dcfce7; color: #15803d; }

.agent-back-btn {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 600; color: #4f46e5;
    cursor: pointer; margin-bottom: 20px;
    background: #ede9fe; border: none;
    padding: 7px 14px; border-radius: 8px;
}

.agent-hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 16px;
    padding: 28px 30px;
    color: #fff;
    margin-bottom: 20px;
}
.agent-hero h2 { font-size: 22px; font-weight: 800; margin: 0 0 6px 0; color: #fff; }
.agent-hero p  { font-size: 14px; margin: 0; opacity: 0.88; }

.agent-param-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    height: 100%;
}
.agent-param-title {
    font-size: 13px; font-weight: 700; color: #4f46e5;
    text-transform: uppercase; letter-spacing: 0.4px;
    margin-bottom: 14px;
}
.agent-rec-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 1.5px solid #22c55e;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow: 0 3px 12px rgba(34,197,94,0.10);
}
.agent-rec-dealer { font-size: 15px; font-weight: 700; color: #15803d; margin-bottom: 6px; }
.agent-rec-action { font-size: 20px; font-weight: 800; color: #166534; margin-bottom: 4px; }
.agent-rec-reason { font-size: 12px; color: #6b7280; }
.agent-value-card {
    border-radius: 12px;
    padding: 18px 20px;
    display: flex; align-items: center; gap: 14px;
}
.agent-value-icon { font-size: 32px; }
.agent-value-label { font-size: 12px; font-weight: 600; color: #6b7280; }
.agent-value-num   { font-size: 24px; font-weight: 800; }
</style>
"""

_AGENT_CATALOG = [
    {
        "id":       "replenishment",
        "name":     "Auto-Replenishment Agent",
        "subtitle": "AI-driven demand forecasting & inventory recommendations — reducing stockouts and maximising captured sales for every dealer.",
        "icon":     "📦",
        "icon_class": "agent-card-icon-replenishment",
        "card_class": "agent-card-replenishment",
        "badge":    "Active",
        "badge_class": "agent-badge-active",
    },
    {
        "id":       "delivery",
        "name":     "Delivery Tracking Agent",
        "subtitle": "Track every open order, estimate delivery dates from historical lead times, action overdue shipments — one place for all dealer queries.",
        "icon":     "🚚",
        "icon_class": "agent-card-icon-delivery",
        "card_class": "agent-card-delivery",
        "badge":    "Active",
        "badge_class": "agent-badge-active",
    },
    {
        "id":         "revenue_recovery",
        "name":       "Revenue Recovery Agent",
        "subtitle":   "Identify high-risk dealers by revenue loss, diagnose margin and product mix issues, and generate AI recovery plans.",
        "icon":       "📉",
        "icon_class": "agent-card-icon-recovery",
        "card_class": "agent-card-recovery",
        "badge":      "Active",
        "badge_class": "agent-badge-active",
    },
]


def render_agent_catalog():
    """Render the AI Agents landing page with workflow selection cards."""
    st.markdown(_AGENT_CSS, unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:6px;">
        <div style="font-size:13px;font-weight:600;color:#6b7280;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">Agentic AI</div>
        <div style="font-size:26px;font-weight:800;color:#111827;margin-bottom:4px;">Agent Workflows</div>
        <div style="font-size:14px;color:#6b7280;margin-bottom:24px;">Select an agent to configure and run autonomous AI workflows across your dealer network.</div>
    </div>
    """, unsafe_allow_html=True)

    cols_per_row = 3
    for row_start in range(0, len(_AGENT_CATALOG), cols_per_row):
        row_agents = _AGENT_CATALOG[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_agents), gap="medium")
        for col, agent in zip(cols, row_agents):
            with col:
                st.markdown(f"""
                <div class="agent-catalog-card {agent['card_class']}">
                    <div class="agent-card-icon {agent['icon_class']}">{agent['icon']}</div>
                    <div class="agent-card-name">{agent['name']}</div>
                    <div class="agent-card-subtitle">{agent['subtitle']}</div>
                    <span class="agent-card-badge {agent['badge_class']}">{agent['badge']}</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open {agent['name']} →", key=f"open_agent_{agent['id']}", use_container_width=True):
                    st.session_state['active_agent'] = agent['id']
                    st.rerun()


def render_replenishment_agent():
    # Migration note: session parameter removed; fetch functions use athena_query() internally
    """Render the Auto-Replenishment Agent workflow UI."""
    import random

    if st.button("← Back to Agent Workflows", key="agent_back", type="secondary"):
        st.session_state['active_agent'] = None
        st.session_state['agent_ran'] = False
        st.rerun()

    st.markdown(_AGENT_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="agent-hero">
        <h2>Auto-Replenishment Agent</h2>
        <p>AI-driven demand forecasting & inventory recommendations — reducing stockouts and maximising captured sales for every dealer.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Agent Input Parameters")
    p1, p2, p3 = st.columns(3, gap="medium")

    with p1:
        st.markdown('<div class="agent-param-card"><div class="agent-param-title">Past Sales Window</div>', unsafe_allow_html=True)
        past_sales_weeks = st.slider("Lookback (weeks)", 4, 52, 12, key="agent_past_weeks")
        st.caption(f"Agent analyses last **{past_sales_weeks} weeks** of sales data per dealer.")
        st.markdown('</div>', unsafe_allow_html=True)

    with p2:
        st.markdown('<div class="agent-param-card"><div class="agent-param-title">Seasonality Index</div>', unsafe_allow_html=True)
        season_index = st.slider("Seasonality multiplier", 0.5, 2.0, 1.0, step=0.05, key="agent_season")
        season_label = "Peak" if season_index > 1.2 else ("Trough" if season_index < 0.8 else "Normal")
        st.caption(f"Current period: **{season_label}** ({season_index:.2f}×)")
        st.markdown('</div>', unsafe_allow_html=True)

    with p3:
        st.markdown('<div class="agent-param-card"><div class="agent-param-title">Regional Demand Factor</div>', unsafe_allow_html=True)
        region_demand = st.selectbox(
            "Select region",
            ["All Regions", "Metro", "Urban", "Rural", "Central", "Other"],
            key="agent_region"
        )
        regional_factor = {"Metro": 1.1, "Urban": 0.9, "Rural": 1.05, "Central": 1.2, "Other": 0.95, "All Regions": 1.0}
        rf = regional_factor.get(region_demand, 1.0)
        st.caption(f"Demand factor for **{region_demand}**: **{rf:.2f}×**")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    if st.button("Run Replenishment Agent", type="primary", key="agent_run"):
        st.session_state['agent_ran'] = True
        st.session_state['agent_params'] = {
            'past_weeks': past_sales_weeks,
            'season_index': season_index,
            'region': region_demand,
            'rf': rf,
        }
        st.rerun()

    if not st.session_state.get('agent_ran'):
        st.info("Configure the parameters above and click **Run Replenishment Agent** to generate recommendations.")
        return

    params = st.session_state.get('agent_params', {})
    pw  = params.get('past_weeks', 12)
    si  = params.get('season_index', 1.0)
    rfp = params.get('rf', 1.0)

    st.markdown("---")
    st.markdown("#### Agent Recommendations")

    dealers = []
    base_units = {}
    products_list = []
    try:
        # Migration note: session.sql().to_pandas() → run_df()
        dealers_df = run_df(f"""
            SELECT DEALER_NAME, AVG(UNITS_SOLD) AS avg_units
            FROM {_DB}.VW_SALES_VOLUME
            WHERE DEALER_NAME IS NOT NULL
            GROUP BY DEALER_NAME
            ORDER BY DEALER_NAME
        """)
        dealers_df.columns = dealers_df.columns.str.lower()
        dealers = dealers_df['dealer_name'].dropna().tolist()
        base_units = {
            row['dealer_name']: max(5, int(row['avg_units'] or 10))
            for _, row in dealers_df.iterrows()
            if row['dealer_name']
        }
    except Exception:
        pass

    try:
        # Migration note: session.sql().to_pandas() → run_df()
        
        #prod_df = run_df(
        #    SELECT DISTINCT PRODUCT_CATEGORY
        #    FROM {_DB}.VW_SALES_VOLUME
        #    WHERE PRODUCT_CATEGORY IS NOT NULL
        #    LIMIT 20
        #)
        #prod_df.columns = prod_df.columns.str.lower()
        #products_list = prod_df['product_category'].dropna().tolist()
        
        products_list = None
    except Exception:
        pass

    if not products_list:
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            prod_df2 = run_df(f"""
                SELECT DISTINCT PRODUCT_CATEGORY
                FROM {_DB}.VW_TRANSACTION_LINEAGE
                WHERE PRODUCT_CATEGORY IS NOT NULL
                LIMIT 20
            """)
            prod_df2.columns = prod_df2.columns.str.lower()
            products_list = prod_df2['product_category'].dropna().tolist()
        except Exception:
            pass

    if not dealers:
        return

    if not products_list:
        products_list = ["General Stock"]

    stock_levels = {}
    try:
        # Migration note: session.sql().to_pandas() → run_df()
        #stock_df = run_df(f"""
        #    SELECT
        #        DEALER_NAME,
        #        AVG(STOCK_QUANTITY)        AS avg_stock,
        #        MIN(STOCK_QUANTITY)        AS min_stock,
        #        SUM(CASE WHEN STOCK_QUANTITY = 0 THEN 1 ELSE 0 END) AS zero_stock_count,
        #        COUNT(*)                   AS total_products
        #    FROM {_DB}.VW_STOCK_AVAILABILITY_DEALER
        #    WHERE DEALER_NAME IS NOT NULL
        #    GROUP BY DEALER_NAME
        #""")
        #stock_df.columns = stock_df.columns.str.lower()
        #for _, row in stock_df.iterrows():
        #    dn = row['dealer_name']
        #    if dn:
        #        stock_levels[dn] = {
        #            'avg_stock':        float(row['avg_stock']   or 0),
        #            'min_stock':        float(row['min_stock']   or 0),
        #            'zero_stock_count': int(row['zero_stock_count'] or 0),
        #            'total_products':   int(row['total_products']   or 1),
        #        }
        stock_levels = {}
    except Exception:
        pass

    overall_avg = sum(base_units.values()) / len(base_units) if base_units else 1

    max_zero = max((v['zero_stock_count'] for v in stock_levels.values()), default=1) or 1
    all_avg_stocks = [v['avg_stock'] for v in stock_levels.values() if v['avg_stock'] > 0]
    min_avg_stock  = min(all_avg_stocks) if all_avg_stocks else 1
    max_avg_stock  = max(all_avg_stocks) if all_avg_stocks else 1

    recs = []
    for dealer in dealers:
        base    = base_units.get(dealer, 10)
        product = random.choice(products_list)
        rec_qty = max(1, int(base * si * rfp * (pw / 12)))

        sl = stock_levels.get(dealer)
        if sl:
            stock_range = max_avg_stock - min_avg_stock or 1
            low_stock_score = 1.0 - ((sl['avg_stock'] - min_avg_stock) / stock_range)
            zero_frac = sl['zero_stock_count'] / max(sl['total_products'], 1)
            velocity_pressure = min(1.0, base / (overall_avg * 2)) if overall_avg else 0
            risk_score = (low_stock_score * 0.50) + (zero_frac * 0.35) + (velocity_pressure * 0.15)
            stock_display = f"Avg Stock: {sl['avg_stock']:.0f} units | Stockouts: {sl['zero_stock_count']} lines"
        else:
            sorted_bases = sorted(base_units.values())
            rank = sorted_bases.index(base) if base in sorted_bases else 0
            pct_rank = rank / max(len(sorted_bases) - 1, 1)
            risk_score = max(0.05, min(0.95, 1.0 - pct_rank))
            stock_display = f"Avg Sales: {base:,} units/period"

        risk_score = max(0.05, min(0.97, risk_score))
        recs.append({
            'dealer':        dealer,
            'product':       product,
            'rec_qty':       rec_qty,
            'risk_score':    risk_score,
            'avg_units':     base,
            'stock_display': stock_display,
        })

    recs.sort(key=lambda x: x['risk_score'], reverse=True)

    def _rec_card_style(score):
        if score > 0.7:
            return (
                "background:linear-gradient(135deg,#fff1f2 0%,#ffe4e6 100%);"
                "border:1.5px solid #f87171;",
                "color:#991b1b;",
                "color:#be123c;"
            )
        elif score > 0.4:
            return (
                "background:linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%);"
                "border:1.5px solid #fbbf24;",
                "color:#92400e;",
                "color:#b45309;"
            )
        else:
            return (
                "background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);"
                "border:1.5px solid #22c55e;",
                "color:#15803d;",
                "color:#166534;"
            )

    if 'po_actions' not in st.session_state:
        st.session_state['po_actions'] = {}
    if 'po_log' not in st.session_state:
        st.session_state['po_log'] = []
    if 'po_notes' not in st.session_state:
        st.session_state['po_notes'] = {}
    if 'po_qty_overrides' not in st.session_state:
        st.session_state['po_qty_overrides'] = {}

    TIER_CFG = {
        "critical": {"bg": "#fff1f2", "border": "#f87171",
                     "shadow": "rgba(220,38,38,0.13)", "dc": "#991b1b", "ac": "#be123c",
                     "badge_bg": "#ffe4e6", "badge_color": "#be123c", "label": "Critical"},
        "medium":   {"bg": "#fffbeb", "border": "#fbbf24",
                     "shadow": "rgba(251,191,36,0.13)", "dc": "#92400e", "ac": "#b45309",
                     "badge_bg": "#fef3c7", "badge_color": "#b45309", "label": "Medium"},
        "low":      {"bg": "#f0fdf4", "border": "#22c55e",
                     "shadow": "rgba(34,197,94,0.13)",  "dc": "#15803d", "ac": "#166534",
                     "badge_bg": "#dcfce7", "badge_color": "#15803d", "label": "Low Risk"},
    }

    st.markdown("""
    <style>
    .rec-badge-approved {
        display:inline-flex; align-items:center; gap:6px;
        background:#dcfce7; border:1px solid #22c55e;
        border-radius:20px; padding:5px 14px; font-size:12px; font-weight:700;
        color:#15803d; margin-bottom:10px;
    }
    .rec-badge-rejected {
        display:inline-flex; align-items:center; gap:6px;
        background:#fee2e2; border:1px solid #f87171;
        border-radius:20px; padding:5px 14px; font-size:12px; font-weight:700;
        color:#dc2626; margin-bottom:10px;
    }
    hr.rec-divider { border:none; border-top:1px solid rgba(0,0,0,0.08); margin:12px 0; }
    div[data-testid="stNumberInput"] input {
        border-radius:10px !important; font-size:15px !important;
        font-weight:700 !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius:10px !important; font-size:13px !important;
    }
    .bulk-po-banner {
        display:flex; align-items:center; justify-content:space-between;
        background: linear-gradient(135deg,#1e1b4b 0%,#312e81 100%);
        border-radius:14px; padding:16px 22px; margin-bottom:18px;
        box-shadow: 0 4px 20px rgba(49,46,129,0.25);
    }
    .bulk-po-banner-text { color:#e0e7ff; }
    .bulk-po-banner-text h4 { margin:0 0 4px 0; font-size:15px; font-weight:800; color:#fff; }
    .bulk-po-banner-text p  { margin:0; font-size:12px; color:#c7d2fe; }
    </style>
    """, unsafe_allow_html=True)

    critical_pending = [
        (i, r) for i, r in enumerate(recs)
        if r['risk_score'] > 0.7
        and st.session_state['po_actions'].get(i) not in ('approved', 'rejected')
    ]

    if critical_pending:
        bulk_left, bulk_right = st.columns([3, 1], gap="medium")
        with bulk_left:
            st.markdown(
                f'<div class="bulk-po-banner">'
                f'  <div class="bulk-po-banner-text">'
                f'    <h4>{len(critical_pending)} Critical Dealer{"s" if len(critical_pending)>1 else ""} Need Immediate Restocking</h4>'
                f'    <p>Create purchase orders for all critical-risk dealers in one click. Each PO will use the AI-recommended quantity.</p>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with bulk_right:
            if st.button(
                f"Bulk Create {len(critical_pending)} POs",
                key="bulk_order_btn", type="primary", use_container_width=True
            ):
                for ci, crec in critical_pending:
                    st.session_state['po_actions'][ci] = 'approved'
                    st.session_state['po_log'].append({
                        'PO #':        f"PO-{1000 + len(st.session_state['po_log']) + 1}",
                        'Dealer':      crec['dealer'],
                        'Product':     crec['product'],
                        'Qty Ordered': crec['rec_qty'],
                        'Risk':        'Critical — Stockout Risk',
                        'Note':        'Bulk order — auto approved',
                        'Status':      'Simulated — Pending',
                        'Created At':  datetime.now().strftime('%Y-%m-%d %H:%M'),
                    })
                st.rerun()
    else:
        st.success("✅ All critical dealers have been actioned.", icon=None)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    rec_col1, rec_col2 = st.columns(2, gap="medium")

    for i, rec in enumerate(recs):
        target_col   = rec_col1 if i % 2 == 0 else rec_col2
        score        = rec['risk_score']
        risk_label   = "Critical — Stockout Risk" if score > 0.7 else ("Medium Risk" if score > 0.4 else "Low Risk")
        risk_dot     = ""
        tier         = "critical" if score > 0.7 else ("medium" if score > 0.4 else "low")
        tc           = TIER_CFG[tier]
        action_taken = st.session_state['po_actions'].get(i)
        form_key     = f"rec_form_{i}"

        with target_col:
            btn_colors = {
                "critical": {"bg": "#dc2626", "hover": "#b91c1c", "shadow": "rgba(220,38,38,0.35)"},
                "medium":   {"bg": "#d97706", "hover": "#b45309", "shadow": "rgba(217,119,6,0.35)"},
                "low":      {"bg": "#16a34a", "hover": "#15803d", "shadow": "rgba(22,163,74,0.35)"},
            }
            bc = btn_colors[tier]

            _form_sel = f"div[data-testid='stForm'][aria-label='{form_key}']"
            _css = (
                f"{_form_sel} {{"
                f"  background:{tc['bg']} !important;"
                f"  border:1.5px solid {tc['border']} !important;"
                f"  border-radius:16px !important;"
                f"  box-shadow:0 4px 16px {tc['shadow']} !important;"
                f"}}"
                f"{_form_sel} > div:first-child {{"
                f"  background:{tc['bg']} !important; border-radius:14px !important;"
                f"}}"
                f"{_form_sel} .stVerticalBlock,"
                f"{_form_sel} .stVerticalBlockBorderWrapper {{"
                f"  background:{tc['bg']} !important;"
                f"}}"
                f"{_form_sel} button[kind='primaryFormSubmit'],"
                f"{_form_sel} button[data-testid='baseButton-primaryFormSubmit'],"
                f"{_form_sel} button[type='submit'],"
                f"{_form_sel} .stColumn:first-child button {{"
                f"  background-color:{bc['bg']} !important;"
                f"  background:{bc['bg']} !important;"
                f"  border-color:{bc['bg']} !important;"
                f"  color:#fff !important;"
                f"  box-shadow:0 4px 14px {bc['shadow']} !important;"
                f"}}"
                f"{_form_sel} button[kind='primaryFormSubmit']:hover,"
                f"{_form_sel} button[data-testid='baseButton-primaryFormSubmit']:hover,"
                f"{_form_sel} button[type='submit']:hover,"
                f"{_form_sel} .stColumn:first-child button:hover {{"
                f"  background-color:{bc['hover']} !important;"
                f"  background:{bc['hover']} !important;"
                f"  border-color:{bc['hover']} !important;"
                f"  box-shadow:0 6px 18px {bc['shadow']} !important;"
                f"}}"
            )
            st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

            with st.form(key=form_key, border=True):

                h_left, h_right = st.columns([3, 1])
                with h_left:
                    st.markdown(
                        f'<div style="font-size:13px;font-weight:800;color:{tc["dc"]};margin-bottom:2px;">'
                        f'{html.escape(rec["dealer"])}</div>'
                        f'<div style="font-size:17px;font-weight:900;color:{tc["ac"]};margin-bottom:6px;">'
                        f'Order {rec["rec_qty"]:,} units of {html.escape(str(rec["product"]))}</div>'
                        f'<div style="font-size:11px;color:#6b7280;">'
                        f'Avg Sales: {html.escape(rec["stock_display"])} &nbsp;·&nbsp; '
                        f'Risk {score:.0%} &nbsp;·&nbsp; Season {si:.2f}× &nbsp;·&nbsp; Region {rfp:.2f}×</div>',
                        unsafe_allow_html=True
                    )
                with h_right:
                    st.markdown(
                        f'<div style="text-align:right;padding-top:4px;">'
                        f'<span style="display:inline-flex;align-items:center;gap:5px;'
                        f'background:{tc["badge_bg"]};color:{tc["badge_color"]};'
                        f'border:1px solid {tc["border"]};border-radius:20px;'
                        f'padding:4px 12px;font-size:11px;font-weight:700;">'
                        f'● {tc["label"]}</span></div>',
                        unsafe_allow_html=True
                    )

                st.markdown('<hr class="rec-divider"/>', unsafe_allow_html=True)

                if action_taken == 'approved':
                    st.markdown('<div class="rec-badge-approved">✅ PO Created — Simulated</div>', unsafe_allow_html=True)
                elif action_taken == 'rejected':
                    st.markdown('<div class="rec-badge-rejected">✗ Recommendation Rejected</div>', unsafe_allow_html=True)

                if not action_taken:
                    if i not in st.session_state['po_qty_overrides']:
                        st.session_state['po_qty_overrides'][i] = rec['rec_qty']
                    cur_qty = st.session_state['po_qty_overrides'][i]

                    note_col, qty_col = st.columns([3, 2], gap="small")
                    with note_col:
                        note = st.text_input(
                            "Notes", value="",
                            key=f"po_note_{i}",
                            placeholder="Add a note...",
                            label_visibility="visible",
                        )
                    with qty_col:
                        st.markdown("<div style='font-size:12px;font-weight:600;color:#6b7280;margin-bottom:4px;'>Qty</div>", unsafe_allow_html=True)
                        qminus, qdisp, qplus = st.columns([1, 2, 1], gap="small")
                        with qminus:
                            if st.form_submit_button("—", use_container_width=True):
                                st.session_state['po_qty_overrides'][i] = max(1, cur_qty - 10)
                                st.rerun()
                        with qdisp:
                            st.markdown(
                                f"<div style='text-align:center;font-size:15px;font-weight:700;"
                                f"padding:6px 0;border:1px solid #e5e7eb;border-radius:8px;"
                                f"background:#f9fafb;'>{cur_qty:,}</div>",
                                unsafe_allow_html=True
                            )
                        with qplus:
                            if st.form_submit_button("✚", use_container_width=True):
                                st.session_state['po_qty_overrides'][i] = cur_qty + 10
                                st.rerun()
                        adj_qty = cur_qty

                    btn_a, btn_r = st.columns(2, gap="small")
                    with btn_a:
                        submitted_approve = st.form_submit_button(
                            "Create PO", type="primary", use_container_width=True
                        )
                    with btn_r:
                        submitted_reject = st.form_submit_button(
                            "Reject", use_container_width=True
                        )

                    if submitted_approve:
                        st.session_state['po_actions'][i] = 'approved'
                        st.session_state['po_notes'][i]   = note
                        st.session_state['po_log'].append({
                            'PO #':        f"PO-{1000 + len(st.session_state['po_log']) + 1}",
                            'Dealer':      rec['dealer'],
                            'Product':     rec['product'],
                            'Qty Ordered': int(adj_qty),
                            'Risk':        risk_label,
                            'Note':        note.strip() or '—',
                            'Status':      'Simulated — Pending',
                            'Created At':  datetime.now().strftime('%Y-%m-%d %H:%M'),
                        })
                        st.rerun()

                    if submitted_reject:
                        st.session_state['po_actions'][i] = 'rejected'
                        st.rerun()

                else:
                    undo_col, _ = st.columns([1, 3])
                    with undo_col:
                        if st.form_submit_button("↩ Undo", use_container_width=True):
                            if action_taken == 'approved':
                                st.session_state['po_log'] = [
                                    p for p in st.session_state['po_log']
                                    if not (p['Dealer'] == rec['dealer']
                                            and p['Product'] == rec['product'])
                                ]
                            del st.session_state['po_actions'][i]
                            st.session_state['po_qty_overrides'].pop(i, None)
                            st.rerun()

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    _js_colour_map = {
        i: btn_colors["critical"]["bg"] if recs[i]['risk_score'] > 0.7
           else (btn_colors["medium"]["bg"] if recs[i]['risk_score'] > 0.4
           else btn_colors["low"]["bg"])
        for i in range(len(recs))
    }
    _js_hover_map = {
        i: btn_colors["critical"]["hover"] if recs[i]['risk_score'] > 0.7
           else (btn_colors["medium"]["hover"] if recs[i]['risk_score'] > 0.4
           else btn_colors["low"]["hover"])
        for i in range(len(recs))
    }
    _js_entries = ",".join(
        f'["rec_form_{i}","{_js_colour_map[i]}","{_js_hover_map[i]}"]'
        for i in range(len(recs))
    )
    st.markdown(
        f"<script>"
        f"(function(){{"
        f"  var MAP = [{_js_entries}];"
        f"  function paintBtns(){{"
        f"    MAP.forEach(function(entry){{"
        f"      var formKey = entry[0], bg = entry[1], hov = entry[2];"
        f"      var form = document.querySelector(\"div[data-testid='stForm'][aria-label='\" + formKey + \"']\");"
        f"      if (!form) return;"
        f"      var btns = form.querySelectorAll('button');"
        f"      if (!btns.length) return;"
        f"      var btn = btns[0];"
        f"      btn.style.setProperty('background-color', bg, 'important');"
        f"      btn.style.setProperty('background', bg, 'important');"
        f"      btn.style.setProperty('border-color', bg, 'important');"
        f"      btn.style.setProperty('color', '#fff', 'important');"
        f"      btn.onmouseenter = function(){{ btn.style.setProperty('background-color', hov, 'important'); btn.style.setProperty('background', hov, 'important'); }};"
        f"      btn.onmouseleave = function(){{ btn.style.setProperty('background-color', bg, 'important'); btn.style.setProperty('background', bg, 'important'); }};"
        f"    }});"
        f"  }}"
        f"  function observe(){{"
        f"    var obs = new MutationObserver(paintBtns);"
        f"    obs.observe(document.body, {{childList:true, subtree:true}});"
        f"  }}"
        f"  if (document.readyState === 'loading') {{"
        f"    document.addEventListener('DOMContentLoaded', function(){{ paintBtns(); observe(); }});"
        f"  }} else {{"
        f"    paintBtns(); observe();"
        f"    setTimeout(paintBtns, 200);"
        f"    setTimeout(paintBtns, 600);"
        f"    setTimeout(paintBtns, 1400);"
        f"  }}"
        f"}})();"
        f"</script>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### Projected Impact")

    high_risk           = [r for r in recs if r['risk_score'] > 0.7]
    prevented_stockouts = len(high_risk)
    total_units         = sum(r['rec_qty'] for r in recs)
    captured_sales_est  = int(total_units * rfp * si * 0.15)
    approved_count      = sum(1 for v in st.session_state['po_actions'].values() if v == 'approved')

    v1, v2, v3, v4 = st.columns(4, gap="medium")

    with v1:
        st.markdown(f"""
        <div class="agent-value-card" style="background:#fef3c7;border:1.5px solid #f59e0b;">
            <div class="agent-value-icon">⚠️</div>
            <div>
                <div class="agent-value-label">Prevented Stockouts</div>
                <div class="agent-value-num" style="color:#92400e;">{prevented_stockouts}</div>
                <div style="font-size:11px;color:#6b7280;">High-risk dealers</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with v2:
        st.markdown(f"""
        <div class="agent-value-card" style="background:#dcfce7;border:1.5px solid #22c55e;">
            <div class="agent-value-icon">📈</div>
            <div>
                <div class="agent-value-label">Captured Sales (Est.)</div>
                <div class="agent-value-num" style="color:#15803d;">{captured_sales_est:,}</div>
                <div style="font-size:11px;color:#6b7280;">Additional units</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with v3:
        fill_rate = min(99, 75 + int(si * 10 * rfp))
        st.markdown(f"""
        <div class="agent-value-card" style="background:#eff6ff;border:1.5px solid #3b82f6;">
            <div class="agent-value-icon">🎯</div>
            <div>
                <div class="agent-value-label">Projected Fill Rate</div>
                <div class="agent-value-num" style="color:#1d4ed8;">{fill_rate}%</div>
                <div style="font-size:11px;color:#6b7280;">Recommended orders</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with v4:
        st.markdown(f"""
        <div class="agent-value-card" style="background:#f5f3ff;border:1.5px solid #8b5cf6;">
            <div class="agent-value-icon">✅</div>
            <div>
                <div class="agent-value-label">POs Created</div>
                <div class="agent-value-num" style="color:#6d28d9;">{approved_count}</div>
                <div style="font-size:11px;color:#6b7280;">This session</div>
            </div>
        </div>""", unsafe_allow_html=True)

    po_log = st.session_state.get('po_log', [])
    if po_log:
        st.markdown("---")
        st.markdown("#### Purchase Order Log")
        st.caption("Resets on page refresh.")

        po_df = pd.DataFrame(po_log)

        def _style_po_status(val):
            if 'Simulated' in str(val):
                return 'background:#f5f3ff;color:#6d28d9;font-weight:700;'
            return ''

        def _style_risk(val):
            if 'Critical' in str(val):
                return 'background:#fee2e2;color:#dc2626;font-weight:600;'
            elif 'Medium' in str(val):
                return 'background:#fef3c7;color:#d97706;font-weight:600;'
            return 'background:#dcfce7;color:#16a34a;font-weight:600;'

        styled_po = (
            po_df.style
            .applymap(_style_po_status, subset=['Status'])
            .applymap(_style_risk,      subset=['Risk'])
            .set_table_styles([
                {'selector': 'thead th',
                 'props': [('background', '#f8fafc'), ('color', '#374151'),
                           ('font-weight', '700'), ('font-size', '12px'),
                           ('border-bottom', '2px solid #e5e7eb'), ('padding', '10px 12px')]},
                {'selector': 'tbody td',
                 'props': [('padding', '9px 12px'), ('font-size', '13px'),
                           ('border-bottom', '1px solid #f3f4f6')]},
            ])
        )
        st.dataframe(styled_po, use_container_width=True, hide_index=True)

        dl1, _ = st.columns([1, 3])
        with dl1:
            st.download_button(
                label="Download PO Log (CSV)",
                data=po_df.to_csv(index=False),
                file_name="po_log_simulated.csv",
                mime="text/csv",
                key="dl_po_log",
                use_container_width=True,
            )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.caption("Recommendations are AI-generated estimates. PO actions are simulated — no data is written to the database.")

    if st.button("Reset Agent", key="agent_reset"):
        st.session_state['agent_ran']  = False
        st.session_state['po_actions'] = {}
        st.session_state['po_log']     = []
        st.session_state['po_notes']   = {}
        st.rerun()


def render_delivery_tracking_agent():
    # Migration note: session parameter removed; fetch functions use athena_query() internally
    """Delivery Tracking Agent — redesigned with pipeline/timeline UI."""
    if st.button("← Back to Agent Workflows", key="delivery_back", type="secondary"):
        st.session_state['active_agent'] = None
        st.session_state['delivery_ran'] = False
        st.rerun()

    st.markdown("""
    <style>
    .dt-header {
        background: #0f172a;
        border-radius: 18px;
        padding: 28px 32px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .dt-header-icon {
        font-size: 48px;
        background: linear-gradient(135deg,#0ea5e9,#06b6d4);
        border-radius: 16px;
        width: 72px; height: 72px;
        display: flex; align-items: center; justify-content: center;
        flex-shrink: 0;
    }
    .dt-header-title { font-size: 24px; font-weight: 900; color: #f1f5f9; margin-bottom: 4px; }
    .dt-header-sub   { font-size: 13px; color: #94a3b8; line-height: 1.5; }
    .dt-header-stats { margin-left: auto; display: flex; gap: 20px; flex-shrink: 0; }
    .dt-header-stat  { text-align: center; }
    .dt-header-stat-num  { font-size: 22px; font-weight: 900; color: #38bdf8; }
    .dt-header-stat-lbl  { font-size: 10px; color: #64748b; font-weight: 600;
                           text-transform: uppercase; letter-spacing: 0.5px; }

    .dt-config-row {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        display: flex; gap: 32px; align-items: flex-end;
    }
    .dt-config-label {
        font-size: 10px; font-weight: 700; color: #0ea5e9;
        text-transform: uppercase; letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .dt-pipeline {
        display: flex;
        gap: 0;
        margin: 20px 0 16px 0;
        border-radius: 12px;
        overflow: hidden;
        height: 52px;
    }
    .dt-pipe-seg {
        display: flex; align-items: center; justify-content: center;
        font-size: 12px; font-weight: 700;
        flex: 1;
        cursor: default;
    }

    .dt-order-row {
        display: flex;
        align-items: stretch;
        background: #fff;
        border: 1.5px solid #e2e8f0;
        border-radius: 14px;
        margin-bottom: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .dt-order-row:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.10); }
    .dt-order-left {
        width: 6px;
        flex-shrink: 0;
    }
    .dt-order-body {
        flex: 1;
        padding: 14px 16px;
    }
    .dt-order-title { font-size: 13px; font-weight: 800; color: #0f172a; margin-bottom: 2px; }
    .dt-order-sub   { font-size: 11px; color: #64748b; margin-bottom: 6px; }
    .dt-order-eta   { font-size: 12px; font-weight: 600; }
    .dt-order-right {
        padding: 14px 16px;
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: center;
        gap: 6px;
        min-width: 130px;
    }
    .dt-status-pill {
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        white-space: nowrap;
    }

    .dt-log-header {
        background: #0f172a;
        border-radius: 10px 10px 0 0;
        padding: 14px 20px;
        color: #f1f5f9;
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    for k, v in [('delivery_actions',{}),('delivery_log',[]),('delivery_ran',False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    st.markdown("""
    <div class="dt-header">
        <div class="dt-header-icon" style="font-size:24px;font-weight:900;color:#38bdf8;letter-spacing:1px;">DT</div>
        <div>
            <div class="dt-header-title">Delivery Tracking Agent</div>
            <div class="dt-header-sub">
                Monitors every open order · Calculates estimated arrival from historical lead times<br>
                Flags overdue &amp; at-risk shipments · Lets you act before dealers need to ask
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:12px;font-weight:700;color:#0ea5e9;'
                'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">'
                'Configure Scan</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1], gap="medium")
    with c1:
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            dlr_df = run_df(f"""
                SELECT DISTINCT DEALER_NAME
                FROM {_DB}.VW_TRANSACTION_LINEAGE
                WHERE DEALER_NAME IS NOT NULL ORDER BY DEALER_NAME
            """)
            dealer_options = ["All Dealers"] + dlr_df['DEALER_NAME'].dropna().tolist()
        except Exception:
            dealer_options = ["All Dealers"]
        selected_dealer = st.selectbox("Dealer", dealer_options, key="dt_dealer")
    with c2:
        overdue_days = st.number_input("Overdue after (days)", min_value=1,
                                       max_value=30, value=3, key="dt_overdue")
    with c3:
        scan_days = st.number_input("Scan window (days)", min_value=7,
                                    max_value=180, value=30, key="dt_scan")
    with c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("Scan Now", type="primary",
                                key="delivery_run", use_container_width=True)

    if run_clicked:
        st.session_state.update({
            'delivery_ran': True,
            'delivery_actions': {},
            'delivery_log': [],
            'delivery_params': {
                'dealer': selected_dealer,
                'overdue_days': int(overdue_days),
                'scan_days': int(scan_days),
            }
        })
        st.rerun()

    if not st.session_state.get('delivery_ran'):
        st.markdown("""
        <div style="background:#f0f9ff;border:1.5px dashed #38bdf8;border-radius:14px;
                    padding:40px;text-align:center;margin-top:16px;">
            <div style="font-size:36px;margin-bottom:12px;">📡</div>
            <div style="font-size:16px;font-weight:700;color:#0369a1;margin-bottom:6px;">
                Agent is standing by
            </div>
            <div style="font-size:13px;color:#64748b;">
                Set your parameters above and click <b>Scan Now</b> to begin tracking.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    params    = st.session_state.get('delivery_params', {})
    p_dealer  = params.get('dealer', 'All Dealers')
    p_overdue = params.get('overdue_days', 3)
    p_scan    = params.get('scan_days', 30)

    dealer_where = f"AND DEALER_NAME = '{p_dealer}'" if p_dealer != 'All Dealers' else ''
    try:
        # Migration note: session.sql().to_pandas() → run_df(); DATEADD('day',-N,...) → date_add('day',-N,...)
        open_df = run_df(f"""
            SELECT DISTINCT TRANSACTION_ID, DEALER_NAME, PRODUCT_CATEGORY,
                PRODUCT_DESC, ORDER_DATE, DELIVERY_DATE, LEAD_TIME_DAYS
            FROM {_DB}.VW_TRANSACTION_LINEAGE
            WHERE DELIVERY_FLAG = 'N'
              AND ORDER_DATE >= date_add('day', -{p_scan}, current_date)
              {dealer_where}
            ORDER BY ORDER_DATE ASC LIMIT 200
        """)
        open_df.columns = open_df.columns.str.upper()
    except Exception as e:
        open_df = pd.DataFrame()

    try:
        # Migration note: session.sql().to_pandas() → run_df()
        lead_df = run_df(f"""
            SELECT DEALER_NAME, AVG(AVG_LEAD_TIME_DAYS) AS avg_lead
            FROM {_DB}.VW_DEALER_JOURNEY_COUNTS
            WHERE AVG_LEAD_TIME_DAYS IS NOT NULL GROUP BY DEALER_NAME
        """)
        lead_df.columns = lead_df.columns.str.upper()
        avg_lead_map = dict(zip(lead_df['DEALER_NAME'], lead_df['AVG_LEAD']))
    except Exception:
        avg_lead_map = {}

    global_avg_lead = float(sum(avg_lead_map.values()) / len(avg_lead_map)) if avg_lead_map else 9.0

    if open_df.empty:
        st.success("No open/undelivered orders found for the selected filters.")
        return

    from datetime import date, timedelta
    today = date.today()
    rows  = []

    for _, r in open_df.iterrows():
        dealer     = str(r.get('DEALER_NAME','') or '')
        avg_lead   = avg_lead_map.get(dealer, global_avg_lead)
        order_date = r.get('ORDER_DATE')
        if hasattr(order_date, 'date'):
            order_date = order_date.date()
        elif isinstance(order_date, str):
            try:    order_date = datetime.strptime(order_date[:10], '%Y-%m-%d').date()
            except: order_date = today

        eta         = order_date + timedelta(days=int(avg_lead))
        days_to_eta = (eta - today).days

        if days_to_eta < -p_overdue:
            status = 'Overdue';  sc = '#dc2626'; sb = '#fff1f2'; sbr = '#f87171'; lc = '#dc2626'
        elif days_to_eta <= 2:
            status = 'At Risk';  sc = '#d97706'; sb = '#fffbeb'; sbr = '#fbbf24'; lc = '#6px solid #fbbf24'
        else:
            status = 'On Track'; sc = '#16a34a'; sb = '#f0fdf4'; sbr = '#22c55e'; lc = '#16a34a'

        rows.append({
            'order_id': str(r.get('TRANSACTION_ID','')),
            'dealer':   dealer,
            'product':  str(r.get('PRODUCT_DESC') or r.get('PRODUCT_CATEGORY') or 'N/A'),
            'order_date': str(order_date),
            'eta': str(eta), 'days_to_eta': days_to_eta,
            'avg_lead': int(avg_lead), 'status': status,
            'sc': sc, 'sb': sb, 'sbr': sbr,
        })

    overdue_rows = [r for r in rows if r['status'] == 'Overdue']
    atrisk_rows  = [r for r in rows if r['status'] == 'At Risk']
    ontrack_rows = [r for r in rows if r['status'] == 'On Track']
    total        = len(rows)

    def _pct(n): return max(4, int(n / total * 100)) if total else 0
    st.markdown(f"""
    <div style="margin:20px 0 6px 0;">
        <div style="font-size:11px;font-weight:700;color:#64748b;
                    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:8px;">
            Order Health Pipeline &nbsp;·&nbsp; {total} open orders scanned
        </div>
        <div style="display:flex;border-radius:10px;overflow:hidden;height:44px;gap:2px;">
            <div style="flex:{_pct(len(overdue_rows))};background:#ef4444;
                display:flex;align-items:center;justify-content:center;
                font-size:12px;font-weight:800;color:#fff;">
                {len(overdue_rows)} Overdue
            </div>
            <div style="flex:{_pct(len(atrisk_rows))};background:#f59e0b;
                display:flex;align-items:center;justify-content:center;
                font-size:12px;font-weight:800;color:#fff;">
                {len(atrisk_rows)} At Risk
            </div>
            <div style="flex:{_pct(len(ontrack_rows))};background:#22c55e;
                display:flex;align-items:center;justify-content:center;
                font-size:12px;font-weight:800;color:#fff;">
                {len(ontrack_rows)} On Track
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    actioned_count = len(st.session_state['delivery_actions'])
    a1, a2, a3, a4 = st.columns(4, gap="small")
    for col, label, val, color in [
        (a1, "Total Scanned", total,                '#0ea5e9'),
        (a2, "Overdue",       len(overdue_rows),    '#dc2626'),
        (a3, "At Risk",       len(atrisk_rows),     '#d97706'),
        (a4, "Actioned",      actioned_count,       '#7c3aed'),
    ]:
        with col:
            st.markdown(
                f'<div style="background:#0f172a;border-radius:10px;padding:12px 10px;'
                f'text-align:center;margin-bottom:12px;">'
                f'<div style="font-size:26px;font-weight:900;color:{color};">{val}</div>'
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'text-transform:uppercase;letter-spacing:0.4px;">{label}</div>'
                f'</div>', unsafe_allow_html=True)

    pending_overdue = [r for r in overdue_rows
                       if not st.session_state['delivery_actions'].get(r['order_id'])]
    if pending_overdue:
        be1, be2, _ = st.columns([1, 2, 2])
        with be1:
            if st.button(f"Escalate All Overdue ({len(pending_overdue)})",
                         key="dt_bulk_escalate", type="primary", use_container_width=True):
                for r in pending_overdue:
                    st.session_state['delivery_actions'][r['order_id']] = 'Escalated'
                    st.session_state['delivery_log'].append({
                        'Order ID': r['order_id'], 'Dealer': r['dealer'],
                        'Product':  r['product'],  'ETA':    r['eta'],
                        'Days Overdue': abs(r['days_to_eta']),
                        'Action': 'Escalated', 'Note': 'Bulk escalation',
                        'Time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    })
                st.rerun()
        with be2:
            st.caption(f"Auto-escalates all {len(pending_overdue)} overdue orders in one click.")

    st.markdown("""
    <div style="font-size:12px;font-weight:700;color:#0f172a;
                text-transform:uppercase;letter-spacing:0.5px;
                margin:16px 0 10px 0;padding-bottom:6px;
                border-bottom:2px solid #e2e8f0;">
        Open Orders &nbsp;·&nbsp; Priority: Overdue → At Risk → On Track
    </div>
    """, unsafe_allow_html=True)

    def _log_action(r, action, note):
        st.session_state['delivery_actions'][r['order_id']] = action
        st.session_state['delivery_log'].append({
            'Order ID': r['order_id'], 'Dealer': r['dealer'],
            'Product':  r['product'],  'ETA':    r['eta'],
            'Days Overdue': abs(r['days_to_eta']) if r['days_to_eta'] < 0 else 0,
            'Action': action, 'Note': note or '—',
            'Time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        })

    for idx, r in enumerate(overdue_rows + atrisk_rows + ontrack_rows):
        action_taken = st.session_state['delivery_actions'].get(r['order_id'])
        left_color   = r['sbr']

        action_bar = {
            'Dispatched':    '#22c55e',
            'Escalated':     '#a855f7',
            'Flag Delay':    '#f97316',
            'Notify Dealer': '#3b82f6',
        }
        if action_taken:
            left_color = action_bar.get(action_taken, left_color)

        if r['days_to_eta'] > 0:
            eta_text = f"ETA in <b>{r['days_to_eta']} days</b> &nbsp;·&nbsp; {r['eta']}"
        elif r['days_to_eta'] == 0:
            eta_text = f"<b>Due today</b> &nbsp;·&nbsp; {r['eta']}"
        else:
            eta_text = f"<b style='color:#dc2626;'>{abs(r['days_to_eta'])}d overdue</b> &nbsp;·&nbsp; was due {r['eta']}"

        if action_taken:
            act_badge_style = {
                'Dispatched':    'background:#dcfce7;color:#15803d;border:1px solid #22c55e;',
                'Escalated':     'background:#faf5ff;color:#7e22ce;border:1px solid #a855f7;',
                'Flag Delay':    'background:#fff7ed;color:#c2410c;border:1px solid #f97316;',
                'Notify Dealer': 'background:#eff6ff;color:#1d4ed8;border:1px solid #3b82f6;',
            }.get(action_taken, '')
            status_html = (f'<span style="{act_badge_style}border-radius:20px;'
                          f'padding:2px 10px;font-size:10px;font-weight:700;">'
                          f'{action_taken}</span>')
        else:
            status_html = (f'<span style="background:{r["sb"]};color:{r["sc"]};'
                          f'border:1px solid {r["sbr"]};border-radius:20px;'
                          f'padding:2px 10px;font-size:10px;font-weight:700;">'
                          f'{r["status"]}</span>')

        st.markdown(
            f'<div style="display:flex;background:#fff;border:1.5px solid #e2e8f0;'
            f'border-radius:12px;margin-bottom:4px;overflow:hidden;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
            f'<div style="width:5px;background:{left_color};flex-shrink:0;"></div>'
            f'<div style="flex:1;padding:12px 14px;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:3px;">'
            f'<span style="font-size:13px;font-weight:800;color:#0f172a;">'
            f'{html.escape(r["order_id"])}</span>'
            f'<span style="font-size:12px;color:#64748b;">·</span>'
            f'<span style="font-size:12px;font-weight:600;color:#334155;">'
            f'{html.escape(r["dealer"])}</span>'
            f'<span style="font-size:11px;color:#94a3b8;">· {html.escape(r["product"])}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#64748b;">'
            f'Ordered {r["order_date"]} &nbsp;·&nbsp; Avg lead {r["avg_lead"]}d &nbsp;·&nbsp; {eta_text}'
            f'</div>'
            f'</div>'
            f'<div style="padding:12px 14px;display:flex;align-items:center;gap:8px;flex-shrink:0;">'
            f'{status_html}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Action controls below each row
        if not action_taken:
            note_key = f"dt_note_{idx}"
            n1, n2, n3, n4, n5 = st.columns([2, 1, 1, 1, 1], gap="small")
            with n1:
                note = st.text_input("Note", value="", key=note_key,
                                     placeholder="Add note...",
                                     label_visibility="collapsed")
            with n2:
                if st.button("Dispatched", key=f"dt_d_{idx}",
                             use_container_width=True, type="primary"):
                    _log_action(r, 'Dispatched', note); st.rerun()
            with n3:
                if st.button("Escalate", key=f"dt_e_{idx}", use_container_width=True):
                    _log_action(r, 'Escalated', note); st.rerun()
            with n4:
                if st.button("Flag Delay", key=f"dt_f_{idx}", use_container_width=True):
                    _log_action(r, 'Flag Delay', note); st.rerun()
            with n5:
                if st.button("Notify", key=f"dt_n_{idx}", use_container_width=True):
                    _log_action(r, 'Notify Dealer', note); st.rerun()
        else:
            undo_col, _ = st.columns([1, 5])
            with undo_col:
                if st.button("Undo", key=f"dt_undo_{idx}", use_container_width=True):
                    del st.session_state['delivery_actions'][r['order_id']]
                    st.session_state['delivery_log'] = [
                        l for l in st.session_state['delivery_log']
                        if l['Order ID'] != r['order_id']
                    ]
                    st.rerun()

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── Action Log ────────────────────────────────────────────────────────────
    delivery_log = st.session_state.get('delivery_log', [])
    if delivery_log:
        st.markdown("""
        <div class="dt-log-header">
            Action Log &nbsp;·&nbsp; <span style="color:#64748b;font-size:12px;font-weight:400;">
            Simulated — no data written to DB</span>
        </div>
        """, unsafe_allow_html=True)

        log_df = pd.DataFrame(delivery_log)
        action_styles = {
            'Dispatched':    'background:#dcfce7;color:#15803d;font-weight:700;',
            'Escalated':     'background:#faf5ff;color:#7e22ce;font-weight:700;',
            'Flag Delay':    'background:#fff7ed;color:#c2410c;font-weight:700;',
            'Notify Dealer': 'background:#eff6ff;color:#1d4ed8;font-weight:700;',
        }
        styled_log = (
            log_df.style
            .applymap(lambda v: action_styles.get(str(v),''), subset=['Action'])
            .set_table_styles([
                {'selector': 'thead th',
                 'props': [('background','#1e293b'),('color','#e2e8f0'),
                           ('font-weight','700'),('font-size','11px'),
                           ('padding','10px 12px'),('border-bottom','2px solid #334155')]},
                {'selector': 'tbody td',
                 'props': [('padding','9px 12px'),('font-size','12px'),
                           ('border-bottom','1px solid #f1f5f9')]},
                {'selector': 'tbody tr:nth-child(even) td',
                 'props': [('background','#f8fafc')]},
            ])
        )
        st.dataframe(styled_log, use_container_width=True, hide_index=True)

        dl1, _ = st.columns([1, 4])
        with dl1:
            st.download_button("Download Log (CSV)",
                               data=log_df.to_csv(index=False),
                               file_name="delivery_action_log.csv",
                               mime="text/csv", key="dl_delivery_log",
                               use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.caption("ETAs calculated from historical avg lead times per dealer. All actions simulated.")

    if st.button("Reset Agent", key="delivery_reset"):
        st.session_state.update({
            'delivery_ran': False,
            'delivery_actions': {},
            'delivery_log': [],
        })
        st.rerun()


def render_revenue_recovery_agent():
    # Migration note: session parameter removed
    """
    Revenue Recovery Agent for DealerPulse.
    Uses Athena views:
      - VW_DEALER_REVENUE_GROWTH  : DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, REVENUE, PREV_MONTH_REVENUE, REVENUE_GROWTH_MOM_PERCENT
      - VW_GROSS_PROFIT_MARGIN    : DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, TOTAL_REVENUE, TOTAL_COGS, GROSS_PROFIT_MARGIN_PCT
      - VW_SALES_PER_PRODUCT_CATEGORY : DEALER_NAME, PRODUCT_CATEGORY, PERIOD_YEAR, PERIOD_MONTH, TOTAL_REVENUE, TOTAL_QUANTITY
    """
    import re as _re
    import html as _html

    def _ai(prompt: str) -> str:
        # Migration note: session.sql(CORTEX.COMPLETE) → bedrock_complete()
        try:
            return bedrock_complete(prompt[:3500], model_id="meta.llama3-8b-instruct-v1:0")
        except Exception as e:
            return f"[AI unavailable: {str(e)[:80]}]"

    def _fmt_bold(text: str) -> str:
        """Convert **text** to <strong>text</strong> safely."""
        out = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        out = _html.escape(out).replace(
            "&lt;strong&gt;", "<strong>").replace("&lt;/strong&gt;", "</strong>")
        return out.replace("\n", "<br/>")

    st.markdown(_AGENT_CSS, unsafe_allow_html=True)

    # ── Back button ──────────────────────────────────────────────────────────
    if st.button("← Back to Agent Workflows", key="rr_back", type="secondary"):
        st.session_state["active_agent"] = None
        st.session_state["rr_agent_ran"] = False
        st.rerun()

    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#dc2626 0%,#b91c1c 100%);
         border-radius:16px;padding:28px 30px;color:#fff;margin-bottom:20px;">
        <div style="font-size:22px;font-weight:800;margin-bottom:6px;">
            📉 Revenue Recovery Agent
        </div>
        <p style="font-size:14px;margin:0;opacity:.88;">
            Identify high-risk dealers by revenue loss, diagnose margin and product mix issues, and generate AI recovery plans.        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameters ───────────────────────────────────────────────────────────
    st.markdown("#### Agent Parameters")
    p1, p2, p3 = st.columns(3, gap="medium")

    with p1:
        st.markdown("""
        <div class="agent-param-card">
            <div class="agent-param-title">Analysis Window</div>
        """, unsafe_allow_html=True)
        n_months = st.selectbox(
            "Split", [3, 6, 9, 12], index=1,
            format_func=lambda x: {
                3: "Narrow (25% current / 75% prior)",
                6: "Balanced (40% current / 60% prior)",
                9: "Wide (50% / 50% split)",
                12: "Broad (60% current / 40% prior)"
            }.get(x, f"Split {x}"),
            key="rr_months", label_visibility="collapsed"
        )
        _frac_preview = {3: "25%", 6: "40%", 9: "50%", 12: "60%"}.get(n_months, "50%")
        st.caption(f"Uses your **actual data range** — splits it so the latest **{_frac_preview}** = current period.")
        st.markdown("</div>", unsafe_allow_html=True)

    with p2:
        st.markdown("""
        <div class="agent-param-card">
            <div class="agent-param-title">Min Revenue Drop %</div>
        """, unsafe_allow_html=True)
        drop_thresh = st.slider(
            "Drop", 5, 50, 10, step=5,
            key="rr_thresh", label_visibility="collapsed"
        )
        st.caption(f"Flag dealers with ≥ **{drop_thresh}%** revenue drop.")
        st.markdown("</div>", unsafe_allow_html=True)

    with p3:
        st.markdown("""
        <div class="agent-param-card">
            <div class="agent-param-title">Max Dealers to Analyse</div>
        f""", unsafe_allow_html=True)
        max_dealers = st.slider(
            "Max", 3, 20, 10,
            key="rr_max", label_visibility="collapsed"
        )
        st.caption(f"Generate briefs for up to **{max_dealers}** dealers.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    if st.button("🚀  Run Revenue Recovery Agent", key="rr_run",
                 type="primary", use_container_width=True):
        st.session_state["rr_agent_ran"]    = True
        st.session_state["rr_agent_params"] = {
            "months": n_months, "thresh": drop_thresh, "max": max_dealers
        }

    if not st.session_state.get("rr_agent_ran"):
        st.info("Configure the parameters above and click **Run Revenue Recovery Agent**.")
        return

    params   = st.session_state.get("rr_agent_params", {})
    months   = params.get("months", 6)
    thresh   = params.get("thresh", 10)
    max_d    = params.get("max", 10)

    st.markdown("---")

    # ── Step 1: Revenue comparison via VW_DEALER_REVENUE_GROWTH ──────────────
    _step_badge = lambda txt: st.markdown(
        f'<div style="display:inline-flex;align-items:center;gap:6px;background:#eff6ff;'        f'color:#1e40af;border:1px solid #bfdbfe;border-radius:999px;font-size:11px;'        f'font-weight:700;padding:4px 12px;margin:6px 0 10px;">{txt}</div>',
        unsafe_allow_html=True
    )

    _step_badge("⚙ Step 1 — Revenue Comparison (current vs prior period)")

    with st.spinner("Scanning data range and comparing dealer revenues..."):
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            _bounds_raw = run_df(
                f"SELECT MIN(PERIOD_YEAR) AS MIN_YR, MAX(PERIOD_YEAR) AS MAX_YR,"
                f" COUNT(DISTINCT CAST(PERIOD_YEAR AS VARCHAR) || CAST(PERIOD_MONTH AS VARCHAR)) AS TOTAL_PERIODS"
                f" FROM {_DB}.VW_DEALER_REVENUE_GROWTH WHERE REVENUE IS NOT NULL"
            )
            _bounds_raw.columns = [c.upper() for c in _bounds_raw.columns]
        except Exception as _be:
            st.error(f"Could not read VW_DEALER_REVENUE_GROWTH: {_be}")
            _bounds_raw = None

        if _bounds_raw is None or _bounds_raw.empty or pd.isna(_bounds_raw["MIN_YR"].iloc[0]):
            rev_df = None
        else:
            _min_yr    = int(_bounds_raw["MIN_YR"].iloc[0])
            _max_yr    = int(_bounds_raw["MAX_YR"].iloc[0])
            _n_per     = int(_bounds_raw["TOTAL_PERIODS"].iloc[0]) if not _bounds_raw.empty else 0
            # Use fraction of data as "current" window — works for any date range
            _frac_map  = {3: 0.25, 6: 0.4, 9: 0.5, 12: 0.6}
            _curr_frac = _frac_map.get(months, 0.5)
            _curr_n    = max(1, round(_n_per * _curr_frac))
            _prev_n    = max(1, _n_per - _curr_n)

            _split_sql = f"""
                WITH ranked AS (
                    SELECT
                        DEALER_NAME,
                        PERIOD_YEAR,
                        PERIOD_MONTH,
                        REVENUE,
                        ROW_NUMBER() OVER (
                            ORDER BY PERIOD_YEAR ASC, PERIOD_MONTH ASC
                        ) AS rn,
                        COUNT(*) OVER () AS total_rows
                    FROM {_DB}.VW_DEALER_REVENUE_GROWTH
                    WHERE REVENUE IS NOT NULL
                ),
                split AS (
                    SELECT *,
                        CASE WHEN rn <= ROUND(total_rows * {round(1 - _curr_frac, 2)}, 0)
                             THEN 'prior'
                             ELSE 'current'
                        END AS period_bucket
                    FROM ranked
                ),
                current_p AS (
                    SELECT DEALER_NAME,
                           SUM(REVENUE) AS CURR_REV,
                           COUNT(DISTINCT CAST(PERIOD_YEAR AS VARCHAR) || CAST(PERIOD_MONTH AS VARCHAR)) AS CURR_PERIODS
                    FROM split WHERE period_bucket = 'current'
                    GROUP BY DEALER_NAME
                ),
                prior_p AS (
                    SELECT DEALER_NAME,
                           SUM(REVENUE) AS PREV_REV,
                           COUNT(DISTINCT CAST(PERIOD_YEAR AS VARCHAR) || CAST(PERIOD_MONTH AS VARCHAR)) AS PREV_PERIODS
                    FROM split WHERE period_bucket = 'prior'
                    GROUP BY DEALER_NAME
                )
                SELECT
                    c.DEALER_NAME,
                    COALESCE(c.CURR_REV,  0) AS CURR_REV,
                    COALESCE(p.PREV_REV,  0) AS PREV_REV,
                    COALESCE(c.CURR_PERIODS, 0) AS CURR_MONTHS,
                    ROUND(
                        (COALESCE(c.CURR_REV,0) - COALESCE(p.PREV_REV,0))
                        / NULLIF(COALESCE(p.PREV_REV,0), 0) * 100
                    , 1) AS REV_CHANGE_PCT
                FROM current_p c
                LEFT JOIN prior_p p ON c.DEALER_NAME = p.DEALER_NAME
                WHERE COALESCE(p.PREV_REV, 0) > 0
                  AND COALESCE(c.CURR_REV, 0) < COALESCE(p.PREV_REV,0) * (1 - {thresh}/100.0)
                ORDER BY REV_CHANGE_PCT ASC
                LIMIT {max_d}
            """
            try:
                # Migration note: session.sql().to_pandas() → run_df()
                rev_df = run_df(_split_sql)
                if rev_df is not None and not rev_df.empty:
                    rev_df.columns = [c.upper() for c in rev_df.columns]
            except Exception as _se:
                st.error(f"Revenue comparison query failed: {_se}")
                rev_df = None

            if rev_df is not None and not rev_df.empty:
                st.caption(
                    f"Using actual data: **{_min_yr}** → **{_max_yr}** | "
                    f"Prior window: {_prev_n} periods | Current window: {_curr_n} periods"
                )

    if rev_df is None or rev_df.empty:
        st.warning(
            f"No dealers found with revenue drop ≥ {thresh}% in your data. "
            f"**Try:** reduce the drop threshold to 5%, or choose a narrower split "
            f"(Narrow = most sensitive to recent change)."
        )
        # Reset so user can change params
        if st.button("Reset & Try Again", key="rr_reset_1"):
            st.session_state["rr_agent_ran"] = False
            st.rerun()
        return

    # Ensure columns are uppercase regardless of DB driver
    if rev_df is not None and not rev_df.empty:
        rev_df.columns = [c.upper() for c in rev_df.columns]
    st.success(f"Found **{len(rev_df)} dealers** with revenue drop ≥ {thresh}%.")

    # Summary table
    disp = rev_df.copy()
    disp["CURR_REV"]       = disp["CURR_REV"].apply(lambda v: f"${float(v):,.0f}")
    disp["PREV_REV"]       = disp["PREV_REV"].apply(lambda v: f"${float(v):,.0f}")
    disp["REV_CHANGE_PCT"] = disp["REV_CHANGE_PCT"].apply(lambda v: f"{float(v):.1f}%")
    disp.columns           = ["Dealer", "Current Revenue", "Prior Revenue",
                               "Months Analysed", "Revenue Change %"]
    st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Step 2: Margin context from VW_GROSS_PROFIT_MARGIN ───────────────────
    _step_badge("⚙ Step 2 — Margin & Product Mix Analysis")

    dealer_names = rev_df["DEALER_NAME"].tolist()
    names_sql    = ", ".join(f"\'{n.replace(chr(39), chr(39)+chr(39))}\'" for n in dealer_names)

    with st.spinner("Fetching margin data for flagged dealers..."):
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            margin_df = run_df(f"""
                SELECT DEALER_NAME,
                       AVG(GROSS_PROFIT_MARGIN_PCT) AS AVG_MARGIN,
                       SUM(TOTAL_REVENUE)           AS TOTAL_REV,
                       SUM(TOTAL_COGS)              AS TOTAL_COGS,
                       COUNT(*)                     AS PERIODS
                FROM {_DB}.VW_GROSS_PROFIT_MARGIN
                WHERE DEALER_NAME IN ({names_sql})
                  AND GROSS_PROFIT_MARGIN_PCT IS NOT NULL
                GROUP BY DEALER_NAME
            """)
            margin_df.columns = [c.upper() for c in margin_df.columns]
        except Exception:
            margin_df = pd.DataFrame()

    with st.spinner("Fetching product mix for flagged dealers..."):
        try:
            # Migration note: session.sql().to_pandas() → run_df()
            prod_df = run_df(f"""
                SELECT DEALER_NAME,
                       PRODUCT_CATEGORY,
                       SUM(TOTAL_REVENUE)  AS CAT_REVENUE,
                       SUM(TOTAL_QUANTITY) AS CAT_QTY
                FROM {_DB}.VW_SALES_PER_PRODUCT_CATEGORY
                WHERE DEALER_NAME IN ({names_sql})
                GROUP BY DEALER_NAME, PRODUCT_CATEGORY
                ORDER BY DEALER_NAME, CAT_REVENUE DESC
            """)
            prod_df.columns = [c.upper() for c in prod_df.columns]
        except Exception:
            prod_df = pd.DataFrame()

    # ── Step 3: AI Recovery Briefs ────────────────────────────────────────────
    _step_badge("⚙ Step 3 — Generating AI Recovery Briefs via Bedrock AI")

    results = []
    for _, row in rev_df.iterrows():
        dealer   = str(row["DEALER_NAME"])
        curr_rev = float(row["CURR_REV"])
        prev_rev = float(row["PREV_REV"])
        drop_pct = float(row["REV_CHANGE_PCT"])

        # Margin for this dealer
        m_row        = margin_df[margin_df["DEALER_NAME"] == dealer]
        avg_margin   = float(m_row["AVG_MARGIN"].iloc[0]) if not m_row.empty else None
        margin_txt   = f"{avg_margin:.1f}%" if avg_margin is not None else "N/A"

        # Product mix for this dealer (top 4 categories)
        p_rows       = prod_df[prod_df["DEALER_NAME"] == dealer].head(4)
        prod_summary = "No product data"
        if not p_rows.empty:
            prod_summary = "; ".join(
                f"{r['PRODUCT_CATEGORY']} (${float(r['CAT_REVENUE']):,.0f}, {int(r['CAT_QTY'])} units)"
                for _, r in p_rows.iterrows()
            )

        with st.spinner(f"Generating brief for {dealer}..."):
            brief = _ai(
                f"You are a dealer network analyst. Write a recovery brief for dealer '{dealer}':\n"
                f"- Revenue dropped {abs(drop_pct):.1f}%: from ${prev_rev:,.0f} to ${curr_rev:,.0f}\n"
                f"- Gross profit margin: {margin_txt}\n"
                f"- Top product categories: {prod_summary}\n\n"
                f"Provide:\n"
                f"1. **Root Cause:** 2-sentence diagnosis citing the numbers\n"
                f"2. **Recovery Actions:** 3 specific bullet points, each with expected outcome\n"
                f"3. **30-Day Target:** one measurable goal\n"
                f"Be concise, specific and data-driven."
            )

        results.append({
            "dealer":      dealer,
            "drop_pct":    drop_pct,
            "curr_rev":    curr_rev,
            "prev_rev":    prev_rev,
            "avg_margin":  avg_margin,
            "margin_txt":  margin_txt,
            "prod_df":     p_rows,
            "brief":       brief,
        })

    # ── Step 4: Display results ───────────────────────────────────────────────
    _step_badge("✅ Step 4 — Recovery Briefs Ready")
    st.markdown("### Recovery Briefs by Dealer")

    for item in results:
        drop_c = "#dc2626" if item["drop_pct"] < -30 else "#d97706"
        icon   = "🔴" if item["drop_pct"] < -30 else "🟡"

        with st.expander(
            f"{icon}  {item['dealer']}  |  Drop: {item['drop_pct']:.1f}%  |  "
            f"Current: ${item['curr_rev']:,.0f}  |  Prior: ${item['prev_rev']:,.0f}",
            expanded=True
        ):
            col_a, col_b = st.columns([1.3, 1])

            with col_a:
                st.markdown("**AI Recovery Brief**")
                _brief_html = _fmt_bold(item["brief"])
                st.markdown(
                    f'<div style="background:#fff5f5;border-left:4px solid {drop_c};'                    f'border-radius:8px;padding:14px 16px;font-size:13px;line-height:1.8;">'                    f'{_brief_html}</div>',
                    unsafe_allow_html=True
                )

            with col_b:
                st.markdown("**KPI Snapshot**")
                kpis = [
                    ("Revenue Drop",    f"{item['drop_pct']:.1f}%"),
                    ("Current Revenue", f"${item['curr_rev']:,.0f}"),
                    ("Prior Revenue",   f"${item['prev_rev']:,.0f}"),
                    ("Gross Margin",    item["margin_txt"]),
                ]
                for lbl, val in kpis:
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;'                        f'border-bottom:1px solid #f1f5f9;padding:7px 0;font-size:13px;">'                        f'<span style="color:#6b7280;">{lbl}</span>'                        f'<span style="font-weight:800;color:#111827;">{val}</span></div>',
                        unsafe_allow_html=True
                    )

                if not item["prod_df"].empty:
                    st.markdown("<br>**Top Categories**", unsafe_allow_html=True)
                    prod_disp = item["prod_df"].copy()
                    prod_disp["CAT_REVENUE"] = prod_disp["CAT_REVENUE"].apply(
                        lambda v: f"${float(v):,.0f}"
                    )
                    prod_disp = prod_disp.rename(columns={
                        "PRODUCT_CATEGORY": "Category",
                        "CAT_REVENUE":      "Revenue",
                        "CAT_QTY":          "Units"
                    })
                    st.dataframe(
                        prod_disp[["Category", "Revenue", "Units"]],
                        use_container_width=True, hide_index=True, height=175
                    )

            # ── Action buttons ────────────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            st.markdown("**Quick Actions**")
            _d_slug = item["dealer"].replace(" ", "_").replace("/", "_")[:18]
            qa1, qa2, qa3 = st.columns(3, gap="small")

            with qa1:
                _fk = f"rr_flag_{_d_slug}"
                _fl = st.session_state.get(_fk, False)
                if st.button(
                    "✅ Flagged for Follow-up" if _fl else "🚩 Flag for Follow-up",
                    key=f"rr_flag_btn_{_d_slug}",
                    use_container_width=True, type="secondary"
                ):
                    st.session_state[_fk] = not _fl
                    st.rerun()

            with qa2:
                _sk = f"rr_show_{_d_slug}"
                if st.button("📋 View as Plain Text", key=f"rr_txt_{_d_slug}",
                             use_container_width=True, type="secondary"):
                    st.session_state[_sk] = not st.session_state.get(_sk, False)
                    st.rerun()
                if st.session_state.get(_sk, False):
                    st.code(item["brief"], language=None)

            with qa3:
                _brief_export = (
                    f"REVENUE RECOVERY BRIEF\n"
                    f"{'='*40}\n"
                    f"Dealer  : {item['dealer']}\n"
                    f"Drop    : {item['drop_pct']:.1f}%\n"
                    f"Current : ${item['curr_rev']:,.0f}\n"
                    f"Prior   : ${item['prev_rev']:,.0f}\n"
                    f"Margin  : {item['margin_txt']}\n"
                    f"{'='*40}\n\n"
                    f"{item['brief']}"
                )
                st.download_button(
                    "⬇ Download Brief",
                    data=_brief_export,
                    file_name=f"brief_{_d_slug}.txt",
                    mime="text/plain",
                    key=f"rr_dl_{_d_slug}",
                    use_container_width=True, type="secondary"
                )

    # ── Export full report ────────────────────────────────────────────────────
    st.markdown("---")
    export_rows = [{
        "Dealer":           r["dealer"],
        "Revenue Drop %":   f"{r['drop_pct']:.1f}%",
        "Current Revenue":  f"${r['curr_rev']:,.0f}",
        "Prior Revenue":    f"${r['prev_rev']:,.0f}",
        "Gross Margin":     r["margin_txt"],
        "Recovery Brief":   r["brief"],
    } for r in results]
    import pandas as _pd_exp
    st.download_button(
        "⬇ Download Full Recovery Report (CSV)",
        data=_pd_exp.DataFrame(export_rows).to_csv(index=False),
        file_name=f"dealer_revenue_recovery.csv",
        mime="text/csv",
        key="rr_dl_full",
        use_container_width=True
    )

    if st.button("Reset Agent", key="rr_reset_final", type="secondary"):
        st.session_state["rr_agent_ran"]    = False
        st.session_state["rr_agent_params"] = {}
        st.rerun()


def render_agent_ai_page():
    # Migration note: session param removed; agents use get_aws_session() internally
    active = st.session_state.get('active_agent', None)

    if active == 'replenishment':
        render_replenishment_agent()
    elif active == 'delivery':
        render_delivery_tracking_agent()
    elif active == 'revenue_recovery':
        render_revenue_recovery_agent()
    else:
        render_agent_catalog()


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application flowf"""

    # Migration note: get_snowflake_connection → get_aws_session
    session = get_aws_session()

    # Migration note: GenieQueryCache → DynamoQueryCache (no session or max_size param)
    if not st.session_state.get('genie_cache_initialized', False):
        st.session_state.genie_cache = DynamoQueryCache(
            ttl_seconds=86400,
            similarity_threshold=0.85
        )
        st.session_state.genie_cache_initialized = True

    # Initialize insights visibility state
    if 'show_insights' not in st.session_state:
        st.session_state.show_insights = True

    # Verify required views exist (DB-only mode)
    if session:
        missing_views = check_required_views()
        if missing_views:
            st.error(f"Missing required data views in {_DB}. The dashboard runs in DB-only mode and cannot proceed without them.")
            st.write("Missing views:")
            for v in missing_views:
                st.write(f"- {v}")
            st.stop()

    # PAGE NAVIGATION - Initialized
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'

    # Route to appropriate page (header + content together)
    if st.session_state.current_page == 'Dashboard':
        render_header("Dashboard")
        pass

    elif st.session_state.current_page == 'Genie':
        render_header("Genie")
        render_genie_page()
        def apply_custom_theme_picker(default_color: str = "#FBF9F4", link_text: str = "BG"):
                """
                Show a pill‑shaped 'CHANGE BG COLOR' button in the top‑right
                that controls the app background via a hidden color picker.
                """
                if "bg_color" not in st.session_state:
                    st.session_state.bg_color = default_color

                current_bg = st.session_state.bg_color

                st.markdown(
                    f"""
                    <style>
                        .stApp {{
                            background-color: {current_bg} !important;
                            transition: background-color 0.5s ease;
                        }}

                        .theme-anchor {{
                            position: fixed;
                            bottom: 20px;
                            right: 25px;
                            z-index: 1000000;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            width: 44px;
                            height: 44px;
                            border-radius: 9999px;
                            background-color: {current_bg};
                            border: 1px solid #E5E7EB;
                            box-shadow: 0 4px 10px rgba(15,23,42,0.10);
                            font-size: 11px;
                            font-weight: 600;
                            color: #111827;
                            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                            cursor: pointer;
                        }}

                        .theme-anchor .theme-label-text {{
                            pointer-events: none;
                        }}

                        div[data-testid="stColorPicker"] {{
                            position: fixed !important;
                            bottom: 20px !important;
                            right: 25px !important;
                            width: 44px !important;
                            height: 44px !important;
                            z-index: 1000001 !important;
                            opacity: 0 !important;
                        }}

                        div[data-testid="stColorPicker"] * {{
                            width: 100% !important;
                            height: 100% !important;
                        }}

                        div[data-testid="stColorPicker"] label {{
                            display: none !important;
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div class="theme-anchor">
                        <span class="theme-label-text">{link_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                return st.color_picker("picker", key="bg_color", label_visibility="collapsed")

        apply_custom_theme_picker(link_text="BG")
        return

    elif st.session_state.current_page == 'Dealer Life Cycle':
        render_header("Dealers Health")
        render_dealer_life_cycle()
        return

    # ── Enhancement 4: AI Agents page routing ──────────────────────────────────
    elif st.session_state.current_page == 'AI Agents':
        render_header("AI Agents")
        render_agent_ai_page()
        return

    # DASHBOARD PAGE

    # Add spacing and render title with right-aligned Export/Share buttons
    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
    # Single row for title, background picker, and export/share buttons
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
        <div style="font-size:1.8rem;font-weight:600;color:#222;margin-left:14px;">Welcome to Dealers Dashboard</div>
        <div style="display:flex;align-items:center;gap:18px;">
        </div>
    </div>
    """, unsafe_allow_html=True)

    def apply_custom_theme_picker(default_color: str = "#FBF9F4", link_text: str = "BG"):
        """
        Show a pill‑shaped 'CHANGE BG COLOR' button in the top‑right
        that controls the app background via a hidden color picker.
        """
        if "bg_color" not in st.session_state:
            st.session_state.bg_color = default_color

        current_bg = st.session_state.bg_color

        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-color: {current_bg} !important;
                    transition: background-color 0.5s ease;
                }}

                .theme-anchor {{
                    position: fixed;
                    bottom: 20px;
                    right: 25px;
                    z-index: 1000000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 44px;
                    height: 44px;
                    border-radius: 9999px;
                    background-color: {current_bg};
                    border: 1px solid #E5E7EB;
                    box-shadow: 0 4px 10px rgba(15,23,42,0.10);
                    font-size: 11px;
                    font-weight: 600;
                    color: #111827;
                    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                    cursor: pointer;
                }}

                .theme-anchor .theme-label-text {{
                    pointer-events: none;
                }}

                div[data-testid="stColorPicker"] {{
                    position: fixed !important;
                    bottom: 20px !important;
                    right: 25px !important;
                    width: 44px !important;
                    height: 44px !important;
                    z-index: 1000001 !important;
                    opacity: 0 !important;
                }}

                div[data-testid="stColorPicker"] * {{
                    width: 100% !important;
                    height: 100% !important;
                }}

                div[data-testid="stColorPicker"] label {{
                    display: none !important;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="theme-anchor">
                <span class="theme-label-text">{link_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return st.color_picker("picker", key="bg_color", label_visibility="collapsed")

    apply_custom_theme_picker(link_text="BG")

    filters_data = render_filters(session)

    # Map returned values to variables expected later in the flow
    date_range = (filters_data['from_date'], filters_data['to_date'])
    selected_dealer = filters_data['dealer']
    product = filters_data['product']

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Define filters based on user input
    filters = {
        'from_date': date_range[0] if isinstance(date_range, tuple) else datetime(2026, 1, 1),
        'to_date': date_range[1] if isinstance(date_range, tuple) else datetime(2026, 2, 1),
        'dealer': selected_dealer if selected_dealer != 'All Dealers' else 'All Dealers',
        'product': product if product != 'Product' else None,
        'metric': 'Revenue',
        'time_period': 'Last 30 Days'
    }

    # Render insights
    render_insights(session)
    # If hidden, show unhide button in header actions
    if not st.session_state.get('show_insights', True):
        actions_col = st.columns([4, 1.7])[1]
        with actions_col:
            if st.button("Show Strategic Insights", key="show_insights_btn", help="Show Strategic Insights"):
                st.session_state.show_insights = True
                st.rerun()

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Render KPI metrics
    render_kpi_metrics(session, filters)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Render attention and priority
    render_attention_and_priority(session, filters)

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Render visualizations (charts and notes)
    render_visualizations()

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        <small>Copyright © 2026. YASH Technologies. All Rights Reserved</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()