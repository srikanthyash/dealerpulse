"""
ai_service.py — AI routing, prompt logic, SQL generation, and Genie pipeline.

Migration note: replaces every SNOWFLAKE.CORTEX.COMPLETE call site in the
routing / analyst / forecast sections (lines 2258, 5373, 7127, 7289, 8291,
8393, 8446, 8515, 8583), generate_sql_with_cortex (line 7834),
call_cortex_analyst (line 8154), generate_forecast_prediction_text (line 8845),
and GenieLongTermMemory._build_memory_from_history (line 2176).
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import yaml

from athena_client import athena_query
from bedrock_client import (
    bedrock_complete,
    bedrock_complete_mistral7b,
    load_yaml_from_s3,
)
from config_loader import get_aws_session, get_config
from utils import get_current_user

logger = logging.getLogger(__name__)
_DB: str = get_config()["athena"]["database"]


# ============================================================================
# DECISION SUPPORT INSTRUCTION — prompt injected into every Genie analyst call
# Source lines 3479-3575 (moved as-is; schema prefix unchanged)
# ============================================================================

DECISION_SUPPORT_INSTRUCTION = f"""
You are a Dealer Performance Analyst AI with programmatic access to the {_DB} database.

Purpose: Produce accurate, auditable, and runnable analyses that map directly to views present in the semantic model.

RESPONSE FORMAT (MANDATORY):
- Every response MUST contain three named sections in this exact order and format. Use XML-style tags when returning runnable SQL.
  1) <DESCRIPTIVE> ... </DESCRIPTIVE>
  2) <PRESCRIPTIVE> ... </PRESCRIPTIVE>
  3) <PREDICTIVE> ... </PREDICTIVE>

DESCRIPTIVE: Only facts, metrics, and short observations derived from the query results. Cite exact numbers, dealer identifiers (dealer_name or DEALER_NAME), and time periods. If a metric is not available, explicitly state which table or column is missing and why the analysis cannot be completed.

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
    - Ensure alias uniqueness: do not reuse the same alias for multiple expressions.
    - Avoid ambiguous aliases in ORDER BY / GROUP BY / HAVING; reference the full alias name rather than positional indexes.
6. JOIN KEYS & PREFERRED JOINS:
    - Inventory & backorder views: prefer joining on (VW_STOCK_AVAILABILITY_DEALER.Dealer_name and VW_BACKORDER_INCIDENCE.DEALER_NAME).
    - Sales and operational metrics: join on `dealer_name` when available.
    - Explicitly state which join key is used in the SQL comment and only use a second key when a deterministic mapping is provided in the semantic model.
7. WINDOW & CTE PATTERN:
    - When producing leaderboards, top-N, or YoY comparisons, compute base aggregates in CTEs and then use window functions for ratios and rankings in a final SELECT.
8. DEFAULT LIMITS & SAFETY:
    - For exploratory responses that return raw rows, apply `LIMIT 1000` unless the user explicitly requests a larger export.
    - For time-bounded analyses, always include FROM/TO filters when dates exist.
9. RESULTS & EXPLANATIONS:
    - When returning SQL, wrap the complete runnable statement with <SQL>...</SQL> tags.
    - Add a one-line plain-text rationale above the SQL explaining why this query answers the question.
10. ERROR HANDLING:
    - If a GROUP BY error would occur because a column is selected but not grouped, rewrite the query to pre-aggregate that column in a CTE.

SEMANTIC MODEL USAGE GUIDELINES:
- Use the semantic model's `dimensions`, `time_dimensions`, and `facts` metadata to map natural-language column references to actual column expressions.
- Prefer the canonical metric names declared under `facts`.
- Use synonyms mapping to interpret user queries but always output the actual column names used in the SQL.

If these constraints cannot be satisfied, respond with a clear explanation of the missing semantic element and suggest an alternative using available tables/columns.
"""


# ============================================================================
# GLOBAL ROUTING STATE — populated from YAML at startup
# Source lines 155-159
# ============================================================================

_INTENTS: Dict[str, Dict] = {}
_ENTITY_PATTERNS: Dict[str, Any] = {}
_KEYWORD_TABLES: Dict[str, List] = {}
_ROUTING_INITIALIZED = False
_YAML_MODEL: Dict[str, Any] = {}


# ============================================================================
# YAML ROUTING BUILDERS — extract routing config from YAML model
# Source lines 196-297
# ============================================================================

def _build_intents_from_yaml(model: dict) -> Dict:
    """Extract INTENTS catalog from YAML — reads bedrock_training_patterns first,
    falls back to cortex_optimization.cortex_training_patterns."""
    intents: Dict[str, Dict] = {}

    # Primary: bedrock_training_patterns (AWS-native flat list)
    bedrock_patterns = model.get("bedrock_training_patterns", [])
    if isinstance(bedrock_patterns, list):
        for pattern in bedrock_patterns:
            if not isinstance(pattern, dict):
                continue
            intent_name = pattern.get("intent", "")
            if not intent_name:
                continue
            keywords = pattern.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [keywords]
            verified_query = pattern.get("verified_query", "")
            intents[intent_name] = {
                "keywords":         keywords,
                "also_needs":       [],
                "anti_words":       [],
                "weight":           10,
                "verified_queries": [verified_query] if verified_query else [],
                "description":      pattern.get("description", ""),
            }

    if intents:
        return intents

    # Fallback: cortex_optimization.cortex_training_patterns (legacy Snowflake section)
    patterns = model.get("cortex_optimization", {}).get("cortex_training_patterns", {})
    priority_levels = sorted(k for k in patterns if k.startswith("priority_"))
    for priority_key in priority_levels:
        for pattern in patterns.get(priority_key, []):
            if not isinstance(pattern, dict):
                continue
            intent_id = pattern.get("id", "")
            if not intent_id:
                continue
            intent_name = "_".join(intent_id.split("_")[1:])
            triggers = pattern.get("trigger", [])
            if isinstance(triggers, str):
                triggers = [triggers]
            pre_built_query = pattern.get("pre_built_query")
            intents[intent_name] = {
                "keywords":         triggers,
                "also_needs":       [],
                "anti_words":       [],
                "weight":           10,
                "verified_queries": [pre_built_query] if pre_built_query else [],
                "description":      pattern.get("description", ""),
            }
    return intents


def _build_entity_patterns_from_yaml(model: dict) -> Dict:
    """Auto-discover entity patterns from YAML and build regex patterns."""
    common_entities = {
        "tier":    [r"\b(platinum|gold|silver|bronze)\b"],
        "type":    [r"\b(franchise|independent|authorized|dealer)\b"],
        "state":   [r"\b(maharashtra|delhi|tamil nadu|karnataka|telangana|andhra pradesh|uttar pradesh|punjab|rajasthan|west bengal|kerala|madhya pradesh|bihar|jharkhand|odisha|chhattisgarh|goa|himachal pradesh|uttarakhand|jammu|kashmir|haryana|assam|meghalaya|mizoram|nagaland|manipur|tripura|sikkim|arunachal pradesh)\b"],
        "city":    [r"\b(mumbai|delhi|bangalore|hyderabad|chennai|kolkata|pune|ahmedabad|jaipur|lucknow|chandigarh|patna|indore|bhopal|nagpur|aurangabad|surat|vadodara|agra|varanasi|bodhgaya|amritsar|shimla|darjeeling)\b"],
        "region":  [r"\b(north|south|east|west|central|midwest|northeast|southeast|northwest|southwest|emea|apac|latam|amer)\b"],
        "year":    [r"\b(20\d{2})\b"],
        "month":   [r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b"],
        "top_n":   [r"\btop\s*(\d+)\b"],
    }
    entity_patterns: Dict[str, Any] = {}
    for entity_type, patterns_list in common_entities.items():
        combined = "|".join(patterns_list)
        entity_patterns[entity_type] = re.compile(combined, re.IGNORECASE)
    return entity_patterns


def _build_keyword_tables_from_yaml(model: dict) -> Dict:
    """Build keyword → table mapping from YAML, with defaults."""
    return {
        "unit|volume|sold|sales":                      ["VW_SALES_VOLUME", "VW_SALES_PER_PRODUCT_CATEGORY"],
        "margin|profit|profitability|cogs|gross|gpm":  ["VW_GROSS_PROFIT_MARGIN", "VW_DEALER_CONTRIBUTION_MARGIN"],
        "growth|revenue|trending|declining":           ["VW_DEALER_REVENUE_GROWTH", "VW_GROSS_PROFIT_MARGIN"],
        "cash|ccc|working capital|dso|dio|dpo":        ["VW_CASH_CONVERSION_CYCLE"],
        "service|repair|turnaround|efficiency":        ["VW_AVERAGE_REPAIR_TURNAROUND_TIME"],
        "inventory|stock|backorder|availability|shortage": ["VW_STOCK_AVAILABILITY_DEALER", "VW_BACKORDER_INCIDENCE"],
        "lead time|delivery|fulfillment|order":        ["VW_ORDER_LEAD_TIME"],
        "cost|expense|spending":                       ["VW_GROSS_PROFIT_MARGIN"],
        "where|located|location|city|state|country":   ["VW_DEALER_LOCATION"],
    }


def _build_routing_from_yaml(yaml_content: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Build complete routing engine from YAML string. Source line 166."""
    try:
        model = yaml.safe_load(yaml_content)
    except Exception as exc:
        logger.error("[ROUTING BUILD] YAML parse failed: %s", exc)
        return {}, {}, {}, {}
    intents       = _build_intents_from_yaml(model)
    entity_pats   = _build_entity_patterns_from_yaml(model)
    keyword_tables = _build_keyword_tables_from_yaml(model)
    return intents, entity_pats, keyword_tables, model


def _initialize_routing_from_yaml(yaml_content: str) -> None:
    """Populate global routing state from YAML. Source line 440."""
    global _INTENTS, _ENTITY_PATTERNS, _KEYWORD_TABLES, _ROUTING_INITIALIZED, _YAML_MODEL
    _INTENTS, _ENTITY_PATTERNS, _KEYWORD_TABLES, _YAML_MODEL = _build_routing_from_yaml(yaml_content)
    _ROUTING_INITIALIZED = True
    logger.info("[ROUTING INIT] ✅ %d intents, %d entity patterns", len(_INTENTS), len(_ENTITY_PATTERNS))


def _get_dynamic_intents() -> Dict:
    if not _ROUTING_INITIALIZED:
        logger.warning("[ROUTING] Intents not initialized — using fallback")
        return INTENTS
    return _INTENTS


def _get_dynamic_entity_patterns() -> Dict:
    if not _ROUTING_INITIALIZED:
        logger.warning("[ROUTING] Entity patterns not initialized — using fallback")
        return ENTITY_PATTERNS
    return _ENTITY_PATTERNS


def _get_dynamic_keyword_tables() -> Dict:
    if not _ROUTING_INITIALIZED:
        logger.warning("[ROUTING] Keyword tables not initialized — using fallback")
        return {}
    return _KEYWORD_TABLES


# ============================================================================
# YAML MODEL LOADER
# Migration note: replaces load_yaml_model() (source lines 925-983) which
# read from @DEALER.BUSINESS_VAULT.DEALER_STAGE → now reads from S3 via
# bedrock_client.load_yaml_from_s3(), local file fallback unchanged.
# ============================================================================

@st.cache_data(ttl=7200)
def load_yaml_model(yaml_filename: str | None = None) -> dict:
    """
    Load dealer semantic model.  S3 first, then local file, then fallback.

    Migration note: stage read (session.file.get_stream) replaced by
    load_yaml_from_s3() from bedrock_client (S3 config_prefix + yaml_filename).
    """
    cfg = get_config()
    filename = yaml_filename or cfg["s3"].get("yaml_filename", "dealer_model.yml")

    # Priority 1: S3 (replaces @DEALER.BUSINESS_VAULT.DEALER_STAGE)
    try:
        model = load_yaml_from_s3(filename)
        if model:
            logger.info("[YAML] ✅ Loaded from S3: %s", filename)
            return model
    except Exception as exc:
        logger.debug("[YAML] S3 load failed: %s", exc)

    # Priority 2: local file paths
    local_paths = [
        filename,
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.getcwd(), filename),
        "dealer_model.yml",
        "dealer_model.yml",
        "semantic_model.yml",
    ]
    for path in local_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    model = yaml.safe_load(f)
                    if model:
                        logger.info("[YAML] ✅ Loaded from local file: %s", path)
                        return model
            except Exception as exc:
                logger.debug("[YAML] Local load failed (%s): %s", path, exc)

    # Fallback minimal model
    logger.warning("[YAML] ⚠️ No YAML found — using minimal fallback")
    return {
        "name": "DEALER_KPI_MODEL",
        "description": "Fallback minimal model",
        "tables": [],
        "verified_queries": [],
    }


def yaml_model_as_context(model: dict | None = None) -> str:
    """Serialise the YAML model as a compact string for prompt injection."""
    m = model or load_yaml_model()
    if not m:
        return ""
    return yaml.dump(m, default_flow_style=False, allow_unicode=True)


# ============================================================================
# RouteResult + static fallback intents
# Source lines 483-601
# ============================================================================

@dataclass
class RouteResult:
    """Result of routing a question to a verified query or SQL generation."""
    source:         str
    verified_query: Optional[str]
    sql:            str
    intent:         str
    entities:       dict = field(default_factory=dict)
    explanation:    str  = ""


# Static fallback intents used before YAML is loaded
INTENTS: Dict[str, Dict] = {
    "region_revenue": {
        "keywords": ["region", "regional", "area", "zone", "geography"],
        "also_needs": ["revenue", "sales", "earn", "income", "highest", "best", "top"],
        "anti_words": [],
        "weight": 10,
        "verified_queries": ["dealer_region_revenue"],
    },
    "profitability": {
        "keywords": ["profit", "margin", "profitable", "profitability", "losing money", "loss", "gross margin", "contribution"],
        "also_needs": [],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["dealer_profitability_analysis", "dealer_gross_margin"],
    },
    "top_profitability": {
        "keywords": ["top", "best", "highest", "most profitable", "leading"],
        "also_needs": ["profit", "margin", "profitab"],
        "anti_words": [],
        "weight": 10,
        "verified_queries": ["top_5_dealers_by_profitability"],
    },
    "revenue_growth": {
        "keywords": ["growth", "growing", "grew", "increase", "mom", "month over month", "trending up", "revenue growth"],
        "also_needs": [],
        "anti_words": ["region", "regional", "area"],
        "weight": 8,
        "verified_queries": ["dealer_revenue_growth", "high_growth_dealers"],
    },
    "top_revenue": {
        "keywords": ["top", "best", "highest", "leading", "most revenue"],
        "also_needs": ["revenue", "sales", "earn"],
        "anti_words": ["region", "regional", "area", "profit", "margin"],
        "weight": 10,
        "verified_queries": ["top_5_dealers_by_revenue"],
    },
    "declining_revenue": {
        "keywords": ["declin", "drop", "fall", "worst", "lowest", "struggling", "slow dealer"],
        "also_needs": ["revenue", "sales", "perform"],
        "anti_words": [],
        "weight": 9,
        "verified_queries": ["slow_dealers", "declining_product_categories"],
    },
    "inventory": {
        "keywords": ["inventory", "stock", "backorder", "availability", "on hand", "reserved", "demand"],
        "also_needs": [],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["dealer_inventory_health", "dealer_inventory_issues"],
    },
    "service_efficiency": {
        "keywords": ["repair", "turnaround", "service", "workshop", "fix", "maintenance"],
        "also_needs": [],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["compare_service_efficiency_across_dealers", "dealer_service_efficiency"],
    },
    "lead_time": {
        "keywords": ["lead time", "delivery", "order time", "waiting", "fulfilment"],
        "also_needs": [],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["lead_time_analysis_2026"],
    },
    "cash_cycle": {
        "keywords": ["cash", "ccc", "cash conversion", "dso", "dio", "dpo", "working capital", "liquidity"],
        "also_needs": [],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["check_dealer_ccc"],
    },
    "health_scorecard": {
        "keywords": ["scorecard", "health", "overall", "summary", "overview", "performance summary", "kpi", "critical"],
        "also_needs": [],
        "anti_words": [],
        "weight": 7,
        "verified_queries": ["dealer_health_scorecard"],
    },
    "sales_category": {
        "keywords": ["category", "product", "segment", "parts", "accessories", "vehicle type"],
        "also_needs": ["sales", "revenue", "sold", "sell"],
        "anti_words": [],
        "weight": 8,
        "verified_queries": ["dealer_sales_by_category"],
    },
}

# Static fallback entity patterns
ENTITY_PATTERNS = {
    "region": re.compile(r"\b(north|south|east|west|central|midwest|northeast|southeast|northwest|southwest|emea|apac|latam|amer)\b", re.I),
    "state":  re.compile(r"\b(maharashtra|delhi|tamil nadu|karnataka|telangana|andhra pradesh|uttar pradesh|punjab|rajasthan|west bengal|kerala|madhya pradesh|bihar|jharkhand|odisha|chhattisgarh|goa|himachal pradesh|uttarakhand|jammu|kashmir|haryana|assam|meghalaya|mizoram|nagaland|manipur|tripura|sikkim|arunachal pradesh)\b", re.I),
    "city":   re.compile(r"\b(mumbai|delhi|bangalore|hyderabad|chennai|kolkata|pune|ahmedabad|jaipur|lucknow|chandigarh|patna|indore|bhopal|nagpur|aurangabad|surat|vadodara|agra|varanasi|bodhgaya|amritsar|shimla|darjeeling)\b", re.I),
    "tier":   re.compile(r"\b(platinum|gold|silver|bronze)\b", re.I),
    "type":   re.compile(r"\b(franchise|independent|authorized|dealer)\b", re.I),
    "year":   re.compile(r"\b(20\d{2})\b"),
    "month":  re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b", re.I),
    "top_n":  re.compile(r"\btop\s*(\d+)\b", re.I),
    "dealer": re.compile(r"\bdealer\s*(\d+)\b", re.I),
}


# ============================================================================
# ROUTING HELPERS — no Snowflake dependency, moved as-is
# Source lines 617-842
# ============================================================================

def _norm(text: str) -> str:
    return text.lower().strip()


def _extract_entities(question: str) -> dict:
    found: dict = {}
    q = _norm(question)
    patterns = _get_dynamic_entity_patterns()
    for name, pattern in patterns.items():
        matches = pattern.findall(q)
        if not matches:
            continue
        normalized = []
        for match in matches:
            m_str = " ".join(str(x).strip() for x in match) if isinstance(match, tuple) else str(match).strip()
            if m_str and m_str.lower() not in [n.lower() for n in normalized]:
                normalized.append(m_str)
        if normalized:
            found[name] = normalized if len(normalized) > 1 else normalized[0]
    return found


def _score_intent(question: str) -> list:
    """Return intents sorted by descending match score. Source line 655."""
    q = _norm(question)
    scores: Dict[str, int] = {}
    intents = _get_dynamic_intents()
    for intent_name, cfg in intents.items():
        score = 0
        keyword_hit = False
        for kw in cfg.get("keywords", []):
            if kw in q:
                score += cfg.get("weight", 8)
                keyword_hit = True
        for aw in cfg.get("anti_words", []):
            if aw in q:
                score = -1
                break
        if score < 0 or not keyword_hit:
            continue
        also_needs = cfg.get("also_needs", [])
        if also_needs:
            if any(an in q for an in also_needs):
                score *= 2
            else:
                score = max(0, score - cfg.get("weight", 8))
        if score > 0:
            scores[intent_name] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _find_best_verified_query(intent: str, entities: dict, verified_queries: dict) -> Optional[Tuple[str, str]]:
    """Given an intent, return (vq_name, sql). Source line 691."""
    intents = _get_dynamic_intents()
    if intent not in intents:
        return None
    for candidate in intents[intent].get("verified_queries", []):
        if candidate in verified_queries:
            vq_sql = verified_queries[candidate]
            sql = vq_sql.get("sql", "") if isinstance(vq_sql, dict) else vq_sql
            return (candidate, sql) if sql else None
    return None


def _apply_entity_filters(sql: str, entities: dict, intent: str) -> str:
    """Apply region/tier/type/state/city filters to verified query SQL. Source line 710."""
    if not entities:
        return sql
    filters = []
    if "region" in entities:
        filters.append(f"LOCATION_REGION = '{str(entities['region']).upper()}'")
    if "state" in entities:
        state = entities["state"]
        state = state[0] if isinstance(state, (tuple, list)) else state
        state = " ".join(w.capitalize() for w in str(state).split())
        if state:
            filters.append(f"LOCATION_STATE = '{state}'")
    if "city" in entities:
        city = entities["city"]
        city = city[0] if isinstance(city, (tuple, list)) else city
        city = " ".join(w.capitalize() for w in str(city).split())
        if city:
            filters.append(f"LOCATION_CITY = '{city}'")
    if "tier" in entities:
        tier = entities["tier"]
        tier = tier[0] if isinstance(tier, (tuple, list)) else tier
        if tier:
            filters.append(f"DEALER_TIER = '{str(tier).upper()}'")
    if "type" in entities:
        dtype = entities["type"]
        dtype = dtype[0] if isinstance(dtype, (tuple, list)) else dtype
        dtype_lower = str(dtype).lower()
        if dtype_lower in ("franchise", "independent"):
            filters.append(f"DEALER_TYPE = '{dtype_lower.title()}'")
    if not filters:
        return sql
    base = sql.rstrip().rstrip(";")
    order_idx = base.upper().rfind("ORDER BY")
    if order_idx >= 0:
        order_clause = base[order_idx:]
        base = base[:order_idx].rstrip()
    else:
        order_clause = ""
    where_clause = " AND ".join(filters)
    result = f"{base}\nWHERE {where_clause}"
    if order_clause:
        result += f"\n{order_clause}"
    return result


def route_question(
    question: str,
    verified_queries: dict,
    cortex_generate_fn: Optional[Any] = None,
    confidence_threshold: int = 6,
) -> RouteResult:
    """Main smart routing function. Source line 781."""
    try:
        entities = _extract_entities(question)
        ranked   = _score_intent(question)
    except Exception as exc:
        logger.error("[ROUTING] Entity/intent extraction failed: %s", exc)
        sql = cortex_generate_fn("", question) if cortex_generate_fn else ""
        return RouteResult(
            source="bedrock_generated",  # Migration note: cortex_generated → bedrock_generated
            verified_query=None, sql=sql,
            intent="unknown", entities={},
            explanation="Entity extraction failed — Bedrock SQL generation used.",
        )
    if not ranked:
        sql = cortex_generate_fn("", question) if cortex_generate_fn else ""
        return RouteResult(
            source="bedrock_generated",
            verified_query=None, sql=sql,
            intent="unknown", entities=entities,
            explanation="No intent matched — Bedrock SQL generation used.",
        )
    best_intent, best_score = ranked[0]
    if best_score >= confidence_threshold:
        vq = _find_best_verified_query(best_intent, entities, verified_queries)
        if vq:
            vq_name, vq_sql = vq
            try:
                final_sql = _apply_entity_filters(vq_sql, entities, best_intent)
                needs_wrap = final_sql != vq_sql
            except Exception as exc:
                logger.warning("[ROUTING] Entity filter failed: %s", exc)
                final_sql  = vq_sql
                needs_wrap = False
            return RouteResult(
                source="verified_query" if not needs_wrap else "hybrid",
                verified_query=vq_name,
                sql=final_sql,
                intent=best_intent,
                entities=entities,
                explanation=(
                    f"Intent '{best_intent}' (score {best_score}) → verified query '{vq_name}'"
                    + (" + entity filter applied" if needs_wrap else "")
                ),
            )
    sql = cortex_generate_fn("", question) if cortex_generate_fn else ""
    return RouteResult(
        source="bedrock_generated",
        verified_query=None, sql=sql,
        intent=best_intent, entities=entities,
        explanation=f"Intent '{best_intent}' (score {best_score}) — no verified query → Bedrock SQL generation used.",
    )


def _has_specific_threshold(question: str) -> bool:
    """
    Check if question has numerical thresholds, temporal comparisons, or custom analysis.
    These bypass verified queries and trigger Bedrock SQL generation.
    Source line 7375.
    """
    q = question.lower()
    threshold_patterns = [
        r"[><=]+\s*\d+",
        r"\d+\s*[><=]",
        r"(above|below|over|under|more than|less than|at least|maximum of)\s+\d+",
        r"between\s+\d+\s+(and|to)\s+\d+",
    ]
    for p in threshold_patterns:
        if re.search(p, q, re.IGNORECASE):
            return True
    if re.search(r"(high|low|fast|slow|good|bad|strong|weak)\s+\w+\s+(but|yet|though|while|versus|compared to|vs)\s+(high|low|fast|slow|good|bad|strong|weak)", q, re.IGNORECASE):
        return True
    temporal_patterns = [
        r"\d{4}.*\d{4}",
        r"(compare.*year|compare.*period|compare.*revenue from)",
        r"(this\s+\w+\s+vs|previous\s+\w+|prior\s+\w+|last\s+\w+)",
    ]
    for p in temporal_patterns:
        if re.search(p, q, re.IGNORECASE):
            return True
    custom_patterns = [
        r"potential\s+for",
        r"identify.*that",
        r"show.*indicating",
        r"correlation\s+between",
        r"(which|what).*versus",
        r"compare.*with",
        r"(extreme|unusual|outlier|abnormal)",
    ]
    for p in custom_patterns:
        if re.search(p, q, re.IGNORECASE):
            return True
    return False


def route_verified_query_smart(question: str, model: dict) -> Optional[dict]:
    """
    Smart routing using intent detection and entity extraction. Source line 7437.
    Migration note: Cortex generation fallback replaced by Bedrock generation
    via generate_sql_with_bedrock() in call_bedrock_analyst(); here we return None
    so the caller triggers generation.
    """
    if not model:
        return None
    if _has_specific_threshold(question):
        return None
    vqs = model.get("verified_queries", [])
    vq_map = {vq.get("name"): vq.get("sql", "") for vq in vqs if isinstance(vq, dict) and "sql" in vq}
    result = route_question(question=question, verified_queries=vq_map, confidence_threshold=6)
    logger.debug("[ROUTING] Intent: %s | Source: %s | VQ: %s", result.intent, result.source, result.verified_query)
    if result.verified_query and result.source in ("verified_query", "hybrid"):
        vq_dict = next((vq for vq in vqs if vq.get("name") == result.verified_query), None)
        if vq_dict:
            if result.sql != vq_map.get(result.verified_query, ""):
                vq_dict = dict(vq_dict)
                vq_dict["sql"] = result.sql
                vq_dict["_entity_filtered"] = True
            vq_dict["_smart_router_result"] = result
            return vq_dict
    if result.source == "bedrock_generated":
        return {"name": "bedrock_generated", "sql": result.sql, "_smart_router_result": result}
    return None


def route_verified_query(question: str, model: dict) -> Optional[dict]:
    """Legacy routing wrapper. Source line 7526."""
    try:
        return route_verified_query_smart(question, model)
    except Exception as exc:
        logger.warning("[ROUTING] Smart router failed: %s — falling back", exc)
        return _route_verified_query_legacy(question, model)


def _route_verified_query_legacy(question: str, model: dict) -> Optional[dict]:
    """Keyword-based fallback. Source line 7539."""
    if not model:
        return None
    if _has_specific_threshold(question):
        return None
    q = _norm(question)
    vqs  = model.get("verified_queries", [])
    vq_by_name = {vq.get("name"): vq for vq in vqs if isinstance(vq, dict)}
    for vq in vqs:
        qq = _norm(vq.get("question", ""))
        for chunk in re.split(r"[?.]", qq):
            chunk = chunk.strip()
            if len(chunk) >= 6 and chunk in q:
                return vq
    keyword_map = {
        "declining categor|product categor declining|categor show declining": "declining_product_categories",
        "dealer health|health status|health check|overall health|dealer wellbeing": "dealer_health_scorecard",
        "top dealer|top 5|highest revenue|biggest dealers": "top_5_dealers_by_revenue",
        "profit|unprofitable|low profit|losing money": "dealer_profitability_analysis",
        "gross margin|gross profit|gpm|margin": "dealer_gross_margin",
        "growth|revenue growth|mom growth|declining dealer": "high_growth_dealers",
        "cash|ccc|cash conversion|working capital|dso|dio|dpo": "check_dealer_ccc",
        "inventory|stock|backorder|availability|out of stock": "dealer_inventory_health",
        "compare service|service across|service comparison": "compare_service_efficiency_across_dealers",
        "service|repair|turnaround|efficiency": "dealer_service_efficiency",
        "lead time|delivery|fulfillment": "lead_time_analysis_2026",
        "cost|expense|cogs|spending": "dealer_costs",
        "how many units|units sold|units by dealer|unit sales|volume by dealer": "units_sold_by_dealer",
    }
    for keys, name in keyword_map.items():
        if any(k.strip() in q for k in keys.split("|")):
            return vq_by_name.get(name)
    return None


# ============================================================================
# SQL HELPERS — no Snowflake dependency
# Source lines 7587-7781
# ============================================================================

def add_dealer_filter_to_sql(sql: str, dealer_id: str) -> str:
    """Add WHERE DEALER_NAME filter to SQL if not already present. Source line 7587."""
    if not sql or not dealer_id:
        return sql
    sql = sql.strip()
    if re.search(r"\bwhere\b.*\bdealer_name\s*=", sql, re.IGNORECASE):
        return sql
    dealer_filter   = f"DEALER_NAME = '{dealer_id}'"
    order_match     = re.search(r"\b(ORDER BY|GROUP BY|LIMIT|HAVING)\b", sql, re.IGNORECASE)
    has_where       = bool(re.search(r"\bWHERE\b", sql, re.IGNORECASE))
    if has_where:
        if order_match:
            pos = order_match.start()
            sql = sql[:pos] + f" AND {dealer_filter} " + sql[pos:]
        else:
            sql = sql.rstrip(";") + f" AND {dealer_filter};"
    else:
        if order_match:
            pos = order_match.start()
            sql = sql[:pos] + f" WHERE {dealer_filter} " + sql[pos:]
        else:
            sql = sql.rstrip(";") + f" WHERE {dealer_filter};"
    return sql


def extract_first_select_sql(text: str) -> str:
    """Extract a single SQL statement starting at first WITH or SELECT. Source line 7631."""
    if not text or not isinstance(text, str):
        return ""
    t = text.strip()

    # Llama (and other models) often wrap SQL in markdown code fences with preamble text.
    # Extract content from inside the fence block first before searching for WITH/SELECT.
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", t, re.IGNORECASE | re.DOTALL)
    if fence_match:
        t = fence_match.group(1).strip()
    else:
        # No fenced block — strip any stray backtick markers at edges
        t = re.sub(r"^```(?:sql)?", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"```$", "", t).strip()

    with_match   = re.search(r"\bWITH\b",   t, re.IGNORECASE)
    select_match = re.search(r"\bSELECT\b", t, re.IGNORECASE)
    if with_match:
        t = t[with_match.start():]
    elif select_match:
        t = t[select_match.start():]
    else:
        return ""
    if ";" in t:
        t = t.split(";", 1)[0].strip() + ";"
    return t.strip()


def _build_minimal_schema(question: str, model: dict) -> str:
    """Build focused schema context based on question keywords. Source line 7658."""
    q_lower      = question.lower()
    keyword_tables = _get_dynamic_keyword_tables()
    if not keyword_tables:
        keyword_tables = {
            "unit|volume|sold|sales": ["VW_SALES_VOLUME", "VW_SALES_PER_PRODUCT_CATEGORY"],
            "margin|profit|profitability|cogs|gross|gpm": ["VW_GROSS_PROFIT_MARGIN", "VW_DEALER_CONTRIBUTION_MARGIN"],
            "growth|revenue|trending|declining": ["VW_DEALER_REVENUE_GROWTH", "VW_GROSS_PROFIT_MARGIN"],
            "cash|ccc|working capital|dso|dio|dpo": ["VW_CASH_CONVERSION_CYCLE"],
            "service|repair|turnaround|efficiency": ["VW_AVERAGE_REPAIR_TURNAROUND_TIME"],
            "inventory|stock|backorder|availability|shortage": ["VW_STOCK_AVAILABILITY_DEALER", "VW_BACKORDER_INCIDENCE"],
            "lead time|delivery|fulfillment|order": ["VW_ORDER_LEAD_TIME"],
        }
    relevant_tables: List[str] = []
    for keyword_pattern, tables_list in keyword_tables.items():
        if any(kw.strip() in q_lower for kw in keyword_pattern.split("|")):
            relevant_tables.extend(tables_list)
    relevant_tables = list(dict.fromkeys(relevant_tables))
    if not relevant_tables:
        all_tables = [t.get("name") for t in model.get("tables", []) if isinstance(t, dict) and t.get("name")]
        relevant_tables = all_tables[:5]
    schema_lines: List[str] = []
    for table_name in relevant_tables:
        table_def = next(
            (t for t in model.get("tables", []) if isinstance(t, dict)
             and t.get("name", "").lower() == table_name.lower()),
            None,
        )
        col_defs: List[dict] = []  # [{name, type}]
        if table_def:
            # Primary: flat `columns` list with `role` and `type` fields (current YAML structure)
            for col in table_def.get("columns", []):
                n = col.get("name") or col.get("expr")
                if n:
                    col_defs.append({"name": str(n), "type": col.get("type", "")})
            # Fallback: legacy separate dimension/fact sections
            if not col_defs:
                for section in ("dimensions", "time_dimensions", "facts"):
                    for entry in table_def.get(section, []):
                        n = entry.get("name") or entry.get("expr")
                        if n:
                            col_defs.append({"name": str(n), "type": entry.get("type", "")})
        # Deduplicate by name
        seen: set = set()
        unique_defs = []
        for cd in col_defs:
            if cd["name"] not in seen:
                seen.add(cd["name"])
                unique_defs.append(cd)
        if unique_defs:
            schema_lines.append(f"\nTABLE: {table_name}")
            schema_lines.append(f"  Full path: {_DB}.{table_name.lower()}")
            schema_lines.append(f"  Recommended alias: {table_name[:2].lower()}")
            schema_lines.append("  Columns (name : type) — use exact names, filter dates with date 'YYYY-MM-DD':")
            for cd in unique_defs[:20]:
                type_hint = f" : {cd['type']}" if cd["type"] else ""
                schema_lines.append(f"    - {cd['name']}{type_hint}")
    return "\n".join(schema_lines)


def is_sql_obviously_bad(sql: str) -> Tuple[bool, str]:
    """Quick rule checks to reject common AI SQL mistakes early. Source line 7732."""
    if not sql:
        return True, "Empty SQL"
    u = sql.upper()
    if "SELECT" not in u or "FROM" not in u:
        return True, "Missing SELECT/FROM"
    if re.search(r"\bON\s+DEALER_NAME\s*=\s*DEALER_NAME\b", sql, re.IGNORECASE):
        return True, "Ambiguous join: ON DEALER_NAME = DEALER_NAME (missing table aliases)"
    # date column compared to bare integer — always a TYPE_MISMATCH in Athena
    # e.g. WHERE period_start_date = 2024  or  period_start_date = 1
    if re.search(r"\bperiod_start_date\s*[=<>!]+\s*\d+\b", sql, re.IGNORECASE):
        return True, "TYPE_MISMATCH: period_start_date is a DATE — compare with date 'YYYY-MM-DD', not an integer"
    if re.search(r"\bperiod_start_date\s*BETWEEN\s*\d+", sql, re.IGNORECASE):
        return True, "TYPE_MISMATCH: period_start_date is a DATE — use date 'YYYY-MM-DD' in BETWEEN"
    wrong_columns = {
        r"\bAVG_HOURS\b":                  "AVG_HOURS is WRONG — use AVG_TURNAROUND_HOURS",
        r"\bMARGIN_PCT\b":                 "MARGIN_PCT is WRONG — use GROSS_PROFIT_MARGIN_PCT or CONTRIBUTION_MARGIN_PCT",
        r"\bLEAD_TIME_DAYS\b":             "LEAD_TIME_DAYS is WRONG — use AVG_ORDER_LEAD_TIME_DAYS",
        r"\bCASH_CYCLE\b":                 "CASH_CYCLE is WRONG — use CCC or individual DSO/DIO/DPO",
        r"\bTURNAROUND_TIME\b":            "TURNAROUND_TIME is WRONG — use AVG_TURNAROUND_HOURS",
        r"\bSTOCK_AVAILABILITY_DEALER\b":  "STOCK_AVAILABILITY_DEALER is a TABLE name, not a column — use STOCK_AVAILABILITY_PCT",
        r"\bSTOCK_AVAILABILITY\b(?!_PCT)": "STOCK_AVAILABILITY is WRONG — use STOCK_AVAILABILITY_PCT",
    }
    for wrong_col, msg in wrong_columns.items():
        if re.search(wrong_col, sql, re.IGNORECASE):
            return True, f"Wrong column name: {msg}"
    select_clause = sql.split("FROM")[0] if "FROM" in sql else ""
    if "SELECT" in select_clause.upper():
        select_content = select_clause[select_clause.upper().find("SELECT") + 6:]
        if "." not in select_content:
            words = re.findall(r"\b([A-Z_]+)\b", select_content)
            raw_cols = [w for w in words if w not in ("FROM", "WHERE", "JOIN", "ON", "AND", "OR", "AS", "WITH")]
            if raw_cols and not re.search(r"COUNT|SUM|AVG|MIN|MAX|DISTINCT", select_content, re.IGNORECASE):
                return True, f"Unqualified columns (must use table.column): {', '.join(raw_cols[:3])}"
    return False, ""


def compile_check_sql(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Validate SQL before execution.

    Migration note: EXPLAIN USING TEXT (Snowflake) removed — Athena has no
    compile-only dry-run (per migration analysis). is_sql_obviously_bad()
    handles structural checks; Athena raises InvalidRequestException at
    execution time for syntax errors.
    """
    bad, reason = is_sql_obviously_bad(sql)
    if bad:
        return False, reason
    return True, None


# ============================================================================
# PRESCRIPTIVE HELPER — consolidated from duplicate definitions at lines
# 5339 (llama3-8b) and 7119 (llama3.1-70b). Single function using primary
# model from config.
# Migration note: CORTEX_PRESCRIPTIVE_MODEL → get_config()["bedrock"]["primary_model"]
# ============================================================================

def _bedrock_complete_prescriptive(  # Migration note: _cortex_complete_prescriptive → _bedrock_complete_prescriptive
    content: list,
    run_df_func,
    question: str,
) -> str:
    """
    Generate business-driven prescriptive insights from query data blocks.

    Migration note: both _cortex_complete_prescriptive() definitions (lines 5339
    and 7119) consolidated here. SNOWFLAKE.CORTEX.COMPLETE replaced with
    bedrock_complete(). CORTEX_PRESCRIPTIVE_MODEL (duplicate values "llama3-8b"
    and "llama3.1-70b") unified to config["bedrock"]["primary_model"].
    """
    data_parts: List[str] = []
    for block in content or []:
        if block.get("type") != "sql":
            continue
        sql = block.get("statement", "")
        if not sql.strip():
            continue
        try:
            df = run_df_func(sql)
            if df is None or df.empty:
                continue
            data_parts.append(df.head(40).to_string(index=False, max_colwidth=40))
        except Exception:
            continue
    if not data_parts:
        return ""
    data_str = "\n\n---\n\n".join(data_parts)
    if len(data_str) > 15000:
        data_str = data_str[:15000] + "\n... (truncated)"
    prompt = (
        "You are a procurement business analyst. The user asked a question and received the following data from our analytics. "
        "Provide prescriptive insights: specific recommended actions and risks based on the data. "
        "Be concrete: cite numbers, vendor names, amounts, and percentages from the data. "
        "Format as bullet points (use •). Do NOT use generic phrases like 'review the data' — give actionable recommendations.\n\n"
        f"User question: {question}\n\n"
        f"Data:\n{data_str}"
    )
    # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE
    bedrock_model = get_config()["bedrock"]["primary_model"]  # Migration note: CORTEX_PRESCRIPTIVE_MODEL → config primary_model
    try:
        text = bedrock_complete(prompt, model_id=bedrock_model)
        if text and len(text.strip()) > 20:
            return text.strip()
    except Exception:
        pass
    return ""


def _generate_prescriptive_fallback() -> str:
    """Rule-based fallback when Bedrock prescriptive is unavailable. Source line 7137."""
    return (
        "<ul>"
        "<li>Filter to dealers below median performance and compare against top quartile.</li>"
        "<li>Investigate the largest deltas by category and prioritize the biggest contributors.</li>"
        "<li>Set an owner + due date for each action and track improvement weekly.</li>"
        "</ul>"
    )


def _parse_descriptive_prescriptive(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Split analyst response into (descriptive, prescriptive) sections. Source line 5386."""
    if not text or not text.strip():
        return None, None
    text = text.strip()
    pres_markers = ("**Prescriptive**", "**Prescriptive**:", "Prescriptive:", "Prescriptive**", "\nPrescriptive:")
    idx = -1
    for m in pres_markers:
        i = text.find(m)
        if i >= 0:
            idx = i
            break
    if idx < 0:
        return None, None
    descriptive  = text[:idx].strip()
    prescriptive = text[idx:].strip()
    for m in pres_markers:
        if prescriptive.startswith(m):
            prescriptive = prescriptive[len(m):].strip().lstrip(":\n ")
            break
    for d in ("**Descriptive**", "**Descriptive**:", "Descriptive:", "Descriptive**"):
        if descriptive.startswith(d):
            descriptive = descriptive[len(d):].strip().lstrip(":\n ")
            break
    return descriptive or None, prescriptive or None


def _pick_chart_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Pick a likely categorical X and numeric Y for charts. Source lines 5325, 7084."""
    if df is None or df.empty:
        return None, None
    cat_candidates = [c for c in df.columns if str(c).upper() in ("DEALER_NAME", "VENDOR_NAME", "CATEGORY", "STATUS")]
    if not cat_candidates:
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
                cat_candidates.append(c)
                break
    num_candidates = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_candidates.append(c)
    x = cat_candidates[0] if cat_candidates else None
    y = num_candidates[0] if num_candidates else None
    if x == y:
        y = num_candidates[1] if len(num_candidates) > 1 else None
    return x, y


# ============================================================================
# SQL GENERATION
# Migration note: generate_sql_with_cortex (line 7834) → generate_sql_with_bedrock
# SNOWFLAKE.CORTEX.COMPLETE for SQL_QUERY → bedrock_complete(model_id=primary_model)
# Prompt updated to target Athena SQL syntax instead of Snowflake SQL syntax.
# ============================================================================

def generate_sql_with_bedrock(  # Migration note: generate_sql_with_cortex → generate_sql_with_bedrock
    question: str,
    yaml_content: str,
    bedrock_model: str | None = None,
) -> Tuple[str, bool, Optional[str]]:
    """
    Generate Athena SQL for *question* using the YAML semantic model as context.

    Returns (sql, ok, error_message).

    Migration note: session parameter removed (no Snowflake session needed).
    cortex_model parameter renamed to bedrock_model.
    EXPLAIN USING TEXT removed (Athena has no compile-only check; handled by
    compile_check_sql which calls is_sql_obviously_bad).
    SQL prompt updated to reference Athena syntax (date_add, date_trunc, etc.)
    instead of Snowflake syntax.
    """
    import time as _time
    cfg           = get_config()
    model_id      = bedrock_model or cfg["bedrock"]["primary_model"]  # Migration note: CORTEX_PRESCRIPTIVE_MODEL → primary_model
    yaml_model    = yaml.safe_load(yaml_content) if yaml_content else {}
    schema_ctx    = _build_minimal_schema(question, yaml_model)

    prompt = f"""You are an AWS Athena SQL expert for dealer analytics. Generate EXACT, VALID Athena SQL.

Question: {question}

==============================================================================
CRITICAL RULES — FOLLOW EXACTLY OR SQL WILL FAIL
==============================================================================

1. TABLE NAMES (for FROM/JOIN) use {_DB}. prefix:
   - VW_AVERAGE_REPAIR_TURNAROUND_TIME (alias: t)
   - VW_DEALER_CONTRIBUTION_MARGIN (alias: cm)
   - VW_GROSS_PROFIT_MARGIN (alias: gpm)
   - VW_CASH_CONVERSION_CYCLE (alias: ccc)
   - VW_ORDER_LEAD_TIME (alias: l)
   - VW_BACKORDER_INCIDENCE (alias: b)
   - VW_STOCK_AVAILABILITY_DEALER (alias: s)
   - VW_DEALER_REVENUE_GROWTH (alias: rg)
   - VW_SALES_VOLUME (alias: sv)

2. EXACT COLUMN NAMES — USE THESE EXACTLY:
   VW_AVERAGE_REPAIR_TURNAROUND_TIME: DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, AVG_TURNAROUND_HOURS
   VW_DEALER_CONTRIBUTION_MARGIN: DEALER_NAME, PERIOD_YEAR, PERIOD_MONTH, CONTRIBUTION_MARGIN_PCT
   VW_GROSS_PROFIT_MARGIN: DEALER_NAME, GROSS_PROFIT_MARGIN_PCT, TOTAL_REVENUE, PERIOD_YEAR, PERIOD_MONTH
   VW_CASH_CONVERSION_CYCLE: DEALER_NAME, DSO, DIO, DPO, CCC, PERIOD_YEAR, PERIOD_MONTH
   VW_BACKORDER_INCIDENCE: DEALER_NAME, BACKORDER_INCIDENCE_PCT, PERIOD_YEAR, PERIOD_MONTH
   VW_ORDER_LEAD_TIME: DEALER_NAME, AVG_ORDER_LEAD_TIME_DAYS, PERIOD_YEAR, PERIOD_MONTH

3. ATHENA SQL SYNTAX (not Snowflake):
   - date_add('day', -N, current_timestamp)        NOT DATEADD(...)
   - date_trunc('month', cast(col as date))        NOT DATE_TRUNC(...)
   - date_diff('day', col1, col2)                  NOT DATEDIFF(...)
   - year(current_date)                            NOT YEAR(CURRENT_DATE())
   - IF(cond, a, b)                                NOT IFF(...)
   - No QUALIFY clause (use subquery instead)

4. ALIAS RULES: ALWAYS qualify columns as alias.COLUMN_NAME.
   ✓ SELECT t.DEALER_NAME FROM ... AS t
   ✗ SELECT DEALER_NAME FROM ...

5. JOIN RULE: ALWAYS use explicit qualified columns.
   ✓ ON t.DEALER_NAME = s.DEALER_NAME
   ✗ ON DEALER_NAME = DEALER_NAME

6. MULTI-TABLE JOINS: USE CTE PATTERN to pre-aggregate before joining.

⚠️ CRITICAL: All column references MUST use table alias prefix.

==============================================================================
AVAILABLE TABLES AND COLUMNS:
{schema_ctx}

Return ONLY executable Athena SQL. No markdown, no explanations, no comments. Just valid SQL.
DO NOT ABBREVIATE COLUMN NAMES. Every SELECT must use alias.column format.""".strip()

    t0 = _time.time()
    try:
        # Migration note: bedrock_complete replaces session.sql(CORTEX.COMPLETE) for SQL_QUERY
        raw = bedrock_complete(prompt, model_id=model_id)
        print(f"[TIMING] Bedrock SQL gen: {_time.time()-t0:.2f}s", file=sys.stderr)
        sql = extract_first_select_sql(raw)
        bad, reason = is_sql_obviously_bad(sql)
        if bad:
            return sql, False, f"Rejected: {reason}"
        ok, err = compile_check_sql(sql)
        if ok:
            return sql, True, None
        return sql, False, f"Compile failed: {err}"
    except Exception as exc:
        print(f"[TIMING] Bedrock SQL gen exception after {_time.time()-t0:.2f}s: {exc}", file=sys.stderr)
        return "", False, f"Generation failed: {str(exc)[:100]}"


# ============================================================================
# GENIE LONG-TERM MEMORY
# Migration note: GenieLongTermMemory (lines 2151-2307) → GenieLongTermMemory
# - session parameter removed; DynamoDB replaces Snowflake GENIE_QUERY_HISTORY
# - CURRENT_USER() → get_current_user() from utils
# - DATEADD('day',-30,CURRENT_TIMESTAMP()) → Unix epoch cutoff
# - SNOWFLAKE.CORTEX.COMPLETE → bedrock_complete()
# ============================================================================

class GenieLongTermMemory:
    """
    Long-term memory built from DynamoDB query history.

    Migration note: replaces GenieLongTermMemory (source lines 2151-2307)
    which read from {_DB}.GENIE_QUERY_HISTORY (Snowflake).
    Now reads from DynamoDB dealer_genie_query_cache table with a scan +
    USER_NAME filter. Bedrock replaces Cortex for fact extraction.
    """

    def __init__(self, max_history: int = 20):
        self.max_history  = max_history
        self._memories: List[str] = []
        # Migration note: session.sql("SELECT CURRENT_USER()") → get_current_user()
        self._user: str = get_current_user()
        self._build_memory_from_history()

    def _build_memory_from_history(self) -> None:
        """
        Pull last N distinct questions + responses from DynamoDB and extract facts.

        Migration note: replaces Snowflake query on GENIE_QUERY_HISTORY (line 2184).
        DynamoDB scan with USER_NAME + CREATED_AT filter replaces the SQL SELECT.
        """
        if not self._user:
            return
        try:
            cfg       = get_config()
            table     = cfg["dynamodb"]["cache_table"]
            dynamo    = get_aws_session().client("dynamodb")
            # Migration note: DATEADD('day',-30,CURRENT_TIMESTAMP()) → Unix epoch cutoff
            cutoff    = int(time.time()) - 30 * 86400

            resp = dynamo.scan(
                TableName=table,
                FilterExpression="USER_NAME = :u AND CREATED_AT >= :c",
                ExpressionAttributeValues={
                    ":u": {"S": self._user},
                    ":c": {"N": str(cutoff)},
                },
                ProjectionExpression="QUESTION, RESPONSE_JSON",
                Limit=100,
            )
            items = resp.get("Items", [])[:self.max_history]

            if not items:
                return

            transcript_lines: List[str] = []
            seen: set = set()
            for item in items:
                question = item.get("QUESTION", {}).get("S", "").strip()
                if not question or question in seen:
                    continue
                seen.add(question)
                sql_hint = ""
                try:
                    raw = item.get("RESPONSE_JSON", {}).get("S", "{}")
                    resp_data = json.loads(raw)
                    sql_hint  = (resp_data.get("sql", "") or "")[:100]
                except Exception:
                    pass
                transcript_lines.append(
                    f"Q: {question}" + (f" [used: {sql_hint}]" if sql_hint else "")
                )

            if not transcript_lines:
                return

            extract_prompt = f"""You are analyzing a user's dealer analytics query history.
Extract 3-8 SHORT facts about their interests and what they've discovered.

Query history (last {len(transcript_lines)} questions):
{chr(10).join(transcript_lines)}

Rules:
- Each fact must be ONE sentence max
- Focus on: which dealers they care about, what metrics matter to them,
  patterns they've discovered, topics they frequently ask about
- Only include facts useful as future context
- If nothing worth remembering, respond NONE

Respond with facts only, one per line, no bullets:"""

            # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 2258-2261)
            bedrock_model = get_config()["bedrock"]["primary_model"]  # Migration note: CORTEX_PRESCRIPTIVE_MODEL → primary_model
            raw = bedrock_complete(extract_prompt, model_id=bedrock_model).strip()

            if not raw or raw.upper() == "NONE":
                return

            self._memories = [
                line.strip()
                for line in raw.split("\n")
                if line.strip() and line.strip().upper() != "NONE" and len(line.strip()) > 10
            ][:8]

            logger.info("[MEMORY] Built %d facts from %d history entries for %s",
                        len(self._memories), len(transcript_lines), self._user)

        except Exception as exc:
            logger.warning("[MEMORY BUILD] Failed: %s", str(exc)[:150])
            self._memories = []

    def get_memory_prefix(self) -> str:
        """Return memory context string for injection into Bedrock prompts."""
        if not self._memories:
            return ""
        lines = "\n".join(f"- {m}" for m in self._memories)
        return f"User context (learned from past sessions):\n{lines}\n\n"

    def get_all_memories(self) -> List[str]:
        return self._memories

    def refresh(self) -> None:
        self._memories = []
        self._build_memory_from_history()

    @property
    def count(self) -> int:
        return len(self._memories)


# ============================================================================
# GENIE DEBUG HELPER
# Source line 8138
# ============================================================================

def _genie_debug(msg: str) -> None:
    """Append debug messages to session state and stderr."""
    try:
        if "genie_debug" not in st.session_state:
            st.session_state.genie_debug = []
        st.session_state.genie_debug.append(msg)
        if len(st.session_state.genie_debug) > 100:
            st.session_state.genie_debug = st.session_state.genie_debug[-100:]
    except Exception:
        pass
    print(msg, file=sys.stderr)


# ============================================================================
# CONVERSATIONAL QUESTION DETECTION
# ============================================================================

_CONVERSATIONAL_RE = re.compile(
    r"^\s*("
    r"who\s+are\s+you|what\s+are\s+you|"
    r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"how\s+does\s+this\s+work|how\s+do\s+you\s+work|"
    r"help\s*|"
    r"(hi|hello|hey)\s*|"
    r"what\s+is\s+(this|dealerpulse|the\s+assistant)|"
    r"tell\s+me\s+about\s+yourself|your\s+purpose|"
    r"what\s+(topics|questions|can\s+i\s+ask)"
    r")\s*\??$",
    re.IGNORECASE,
)

_INTRO_TEXT = (
    "**Descriptive** - About DealerPulse AI Assistant\n\n"
    "I'm the DealerPulse AI Assistant — a data-driven analyst for dealer network performance. "
    "I query live data from your dealer database and provide structured analysis.\n\n"
    "**Prescriptive** - What I can help with\n\n"
    "• Top/bottom performing dealers by revenue, profit margin, or growth\n"
    "• Inventory health, stock availability, and backorder rates\n"
    "• Service turnaround times and repair efficiency\n"
    "• Cash conversion cycle (DSO, DIO, DPO) and working capital\n"
    "• Order lead times and delivery fulfillment\n"
    "• Revenue trends, forecasts, and anomaly detection\n\n"
    "**Predictive** - Example questions to try\n\n"
    "• 'Show top performing dealers'\n"
    "• 'Which dealers have the worst inventory health?'\n"
    "• 'Compare service efficiency across dealers'\n"
    "• 'Show dealer health scorecard'\n"
    "• 'What is the cash conversion cycle for Dealer 02?'"
)


def _is_conversational(question: str) -> bool:
    return bool(_CONVERSATIONAL_RE.match(question.strip()))


# ============================================================================
# LLM RESPONSE CLEANUP
# Llama models append LaTeX boxes, "The final answer is:", and meta-commentary.
# Strip all of these before returning text to the UI.
# ============================================================================

def _clean_llm_response(text: str) -> str:
    if not text:
        return text
    # Strip $\boxed{...} LaTeX constructs (may contain multiline content)
    text = re.sub(r"\$\\boxed\{.*?\}", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\\boxed\{\s*\}", "", text)
    # Strip "The final answer is:" lines only (do not strip content after them)
    text = re.sub(r"\n*The final answer is:[^\n]*", "", text, flags=re.IGNORECASE)
    # Strip Llama meta-commentary lines about formatting/instructions
    text = re.sub(
        r"^.*\b(I will remove|as per the instructions|according to the instructions|"
        r"per the instructions|the extra text|format above|formatted according)\b.*$",
        "",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    # Strip orphan trailing code fences
    text = re.sub(r"\n```\s*$", "", text).strip()
    # Collapse 4+ blank lines to 2
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ============================================================================
# MAIN GENIE ANALYST PIPELINE
# Migration note: call_cortex_analyst (line 8154) → call_bedrock_analyst
# All SNOWFLAKE.CORTEX.COMPLETE calls replaced with bedrock_complete().
# session parameter removed; athena_query() replaces run_df(session, sql).
# ============================================================================

def call_bedrock_analyst(  # Migration note: call_cortex_analyst → call_bedrock_analyst
    query_text: str,
    yaml_path:  str | None = None,
    history:    Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Full Genie analyst pipeline: cache → routing → SQL exec → Bedrock text gen.

    Migration note: session parameter removed. Athena replaces Snowflake for
    SQL execution. bedrock_complete() replaces all SNOWFLAKE.CORTEX.COMPLETE
    calls (source lines 8291, 8393, 8446, 8515, 8583).
    CORTEX_PRESCRIPTIVE_MODEL → config["bedrock"]["primary_model"].
    """
    import time as _time
    t_start   = _time.time()
    cfg       = get_config()
    # Migration note: CORTEX_PRESCRIPTIVE_MODEL → config bedrock.primary_model
    bedrock_model = cfg["bedrock"]["primary_model"]  # Migration note: CORTEX_PRESCRIPTIVE_MODEL consolidated

    # ── Build conversation context prefix ──────────────────────────────────
    context_prefix = ""
    if history:
        lines = []
        for msg in history[-6:]:
            role    = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                lines.append(f"{'Assistant' if role == 'assistant' else 'User'}: {content}")
        if lines:
            context_prefix = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    # ── Long-term memory ────────────────────────────────────────────────────
    memory_prefix = ""
    memory_obj    = st.session_state.get("genie_memory")
    if memory_obj:
        memory_prefix = memory_obj.get_memory_prefix()
    full_context = memory_prefix + context_prefix

    cache = st.session_state.genie_cache

    # ── Context-aware cache key ─────────────────────────────────────────────
    def _make_context_hash(question: str, hist: list) -> str:
        prior_user_qs = [
            m.get("content", "").strip().lower()
            for m in (hist or [])
            if m.get("role") == "user" and m.get("content", "").strip()
        ]
        prior_user_qs = [q for q in prior_user_qs if q != question.strip().lower()]
        prior         = prior_user_qs[-1] if prior_user_qs else ""
        return hashlib.md5(f"{question.strip().lower()}|{prior}".encode()).hexdigest()

    ctx_hash = _make_context_hash(query_text, history or [])
    _genie_debug(f"[CACHE] Context hash: {ctx_hash}")

    # ── Follow-up detection ─────────────────────────────────────────────────
    q_norm      = query_text.strip()
    is_followup = bool(
        history and re.search(
            r"\b(it|this|that|those|them|they|their|which one|the best|among them|of them|of those)\b",
            q_norm, re.I,
        )
    )

    # ── Conversational / meta questions — return intro, skip SQL entirely ───
    if _is_conversational(q_norm):
        _genie_debug("[ROUTING] Conversational question detected — returning intro text")
        return {
            "message":          {"content": [{"type": "text", "text": _INTRO_TEXT}]},
            "text_summary":     _INTRO_TEXT,
            "source":           "conversational",
            "response_time_ms": 0,
        }

    # ── Cache GET ───────────────────────────────────────────────────────────
    cached = cache.get(query_text, allow_semantic=not is_followup, cache_key_override=ctx_hash)
    if cached:
        cache_ms = (_time.time() - t_start) * 1000
        cached["cache_fetch_time_ms"] = cache_ms
        # Re-execute SQL if DataFrame missing (legacy entries)
        vdf = cached.get("vendors_df")
        has_df = vdf is not None and (not isinstance(vdf, pd.DataFrame) or not vdf.empty)
        if cached.get("layout") == "quick" and cached.get("sql") and not has_df:
            try:
                cached["vendors_df"] = athena_query(cached["sql"])  # Migration note: run_df → athena_query
            except Exception as exc:
                logger.warning("[CACHE RESTORE] Re-execute failed: %s", exc)
        return cached

    # ── Routing ─────────────────────────────────────────────────────────────
    model          = load_yaml_model(yaml_path)
    verified_query = route_verified_query(query_text, model)

    _genie_debug(f"[ROUTING] Question: {query_text[:60]}...")
    _genie_debug(f"[ROUTING] VQ matched: {verified_query.get('name') if verified_query else 'NONE'}")

    is_real_vq = (
        verified_query is not None
        and verified_query.get("name") not in (None, "bedrock_generated")
        and verified_query.get("sql", "").strip() != ""
    )

    # ── Follow-up context-only answer ───────────────────────────────────────
    if is_followup and not is_real_vq and full_context:
        _genie_debug("[ROUTING] Follow-up + no real VQ → context-only Bedrock answer")
        prompt = (
            f"{full_context}"
            f"Based ONLY on the previous analysis shown above, "
            f"answer this follow-up question directly.\n\n"
            f"Question: {query_text.strip()}\n\n"
            f"Be specific — reference actual dealer names and numbers from the previous data.\n"
            f"Do NOT say you need more data. Answer from what was already shown.\n\n"
            f"Format as:\n\n"
            f"**Descriptive** - Direct answer with specific dealer names and values\n\n"
            f"**Prescriptive** - 3-5 action bullets starting with •\n\n"
            f"**Predictive** - Expected outcome (1-2 sentences)"
        )
        try:
            # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 8289-8293)
            response_text    = _clean_llm_response(bedrock_complete(prompt, model_id=bedrock_model))
            response_time_ms = (_time.time() - t_start) * 1000
            response = {
                "message":          {"content": [{"type": "text", "text": response_text.strip()}]},
                "text_summary":     response_text,
                "source":           "context_followup",
                "response_time_ms": response_time_ms,
            }
            cache.set(query_text, response, response_time_ms=response_time_ms, cache_key_override=ctx_hash)
            return response
        except Exception as exc:
            _genie_debug(f"[CONTEXT FOLLOWUP] Failed: {str(exc)[:80]} — falling through")

    # ── Query consolidation ─────────────────────────────────────────────────
    if verified_query and verified_query.get("name"):
        current_dealer        = extract_dealer_name_from_question(query_text)
        consolidated_response = cache.find_by_verified_query(verified_query.get("name"))
        if consolidated_response:
            cached_dealer       = None
            has_cached_question = False
            if hasattr(cache, "_cache"):
                for entry in list(cache._cache.values()):
                    if entry.get("response") == consolidated_response:
                        cached_dealer       = extract_dealer_name_from_question(entry.get("question", ""))
                        has_cached_question = True
                        break
            if has_cached_question and current_dealer != cached_dealer:
                print(f"[CONSOLIDATION] SKIP — dealer mismatch: "
                      f"current='{current_dealer}' cached='{cached_dealer}'", file=sys.stderr)
            else:
                consolidated_response["cache_fetch_time_ms"] = (_time.time() - t_start) * 1000
                vdf = consolidated_response.get("vendors_df")
                has_df = vdf is not None and (not isinstance(vdf, pd.DataFrame) or not vdf.empty)
                if consolidated_response.get("layout") == "quick" and consolidated_response.get("sql") and not has_df:
                    try:
                        consolidated_response["vendors_df"] = athena_query(consolidated_response["sql"])  # Migration note: run_df → athena_query
                    except Exception as exc:
                        logger.warning("[CONSOLIDATION] Re-execute failed: %s", exc)
                return consolidated_response

    # ═══════════════════════════════════════════════════════════════════════
    # PATH 1 — Verified query
    # ═══════════════════════════════════════════════════════════════════════
    if verified_query and verified_query.get("sql"):
        t1  = _time.time()
        sql = verified_query["sql"].strip()
        dealer_id = extract_dealer_name_from_question(query_text)
        if dealer_id:
            sql = add_dealer_filter_to_sql(sql, dealer_id)
        try:
            df = athena_query(sql)  # Migration note: run_df → athena_query
            t2 = _time.time()
            text_summary = ""
            if not df.empty:
                try:
                    data_summary = (
                        f"Query returned {len(df)} rows:\n{df.head(5).to_string()}"
                        + (f"\n... and {len(df)-5} more rows" if len(df) > 5 else "")
                    )
                    bedrock_prompt = (
                        f"{full_context}"
                        f"You are a senior dealer business analyst. "
                        f"Answer this question based on the query data.\n\n"
                        f"Question: {query_text.strip()}\n\n"
                        f"Data Results:\n{data_summary}\n\n"
                        f"IMPORTANT: Format EXACTLY as follows:\n\n"
                        f"**Descriptive** - What the data shows\n"
                        f"2-3 sentences with key numbers and patterns.\n\n"
                        f"**Prescriptive** - Recommendations & Actions\n"
                        f"5-7 bullet points starting with •\n\n"
                        f"**Predictive** - Expected Impact (12-24 months)\n"
                        f"Financial outcomes and improvements.\n\n"
                        f"Use plain text only. Do NOT use LaTeX, $\\boxed{{...}}, or any math notation."
                    )
                    # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 8391-8395)
                    text_summary = _clean_llm_response(bedrock_complete(bedrock_prompt, model_id=bedrock_model))
                except Exception as exc:
                    logger.warning("[TEXT GEN] Failed: %s", str(exc)[:100])

            response_time_ms = (t2 - t1) * 1000
            response = {
                "layout":           "quick",
                "vendors_df":       df,
                "sql":              sql,
                "source":           "verified_query",
                "verified_name":    verified_query.get("name"),
                "text_summary":     text_summary,
                "response_time_ms": response_time_ms,
            }
            cache.set(query_text, response, response_time_ms=response_time_ms, cache_key_override=ctx_hash)
            return response

        except Exception as exc:
            return {
                "layout":           "sql_failed",
                "sql":              sql,
                "error":            f"Verified query failed: {str(exc)}",
                "source":           "verified_query",
                "response_time_ms": (_time.time() - t1) * 1000,
            }

    # ── PATH 1.5 — Context-only follow-up (no VQ matched) ──────────────────
    if not verified_query and is_followup and full_context:
        _genie_debug("[ROUTING] Follow-up with no VQ match → context-only Bedrock answer")
        prompt = (
            f"{full_context}"
            f"Based on the previous analysis above, answer this follow-up question.\n\n"
            f"Question: {query_text.strip()}\n\n"
            f"Be specific — reference dealer names and numbers from the previous data. "
            f"Format as:\n\n"
            f"**Descriptive** - Direct answer with specific values\n\n"
            f"**Prescriptive** - 3-5 action bullets starting with •\n\n"
            f"**Predictive** - Expected outcome (1-2 sentences)\n\n"
            f"Use plain text only. Do NOT use LaTeX, $\\boxed{{...}}, or any math notation."
        )
        try:
            # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 8446-8451)
            response_text    = _clean_llm_response(bedrock_complete(prompt, model_id=bedrock_model))
            response_time_ms = (_time.time() - t_start) * 1000
            response = {
                "message":          {"content": [{"type": "text", "text": response_text.strip()}]},
                "text_summary":     response_text,
                "source":           "context_followup",
                "response_time_ms": response_time_ms,
            }
            cache.set(query_text, response, response_time_ms=response_time_ms, cache_key_override=ctx_hash)
            return response
        except Exception as exc:
            _genie_debug(f"[CONTEXT FOLLOWUP] Bedrock failed: {str(exc)[:80]} — falling through")

    # ═══════════════════════════════════════════════════════════════════════
    # PATH 2 — Dynamic Bedrock SQL generation
    # Migration note: generate_sql_with_cortex → generate_sql_with_bedrock
    # ═══════════════════════════════════════════════════════════════════════
    yaml_content = ""
    try:
        model_temp = load_yaml_model(yaml_path)
        if model_temp:
            yaml_content = yaml.dump(model_temp, default_flow_style=False)
    except Exception as exc:
        logger.warning("[YAML] Error: %s", exc)

    bedrock_question = (full_context + query_text) if full_context else query_text
    gen_sql, ok, gen_err = generate_sql_with_bedrock(  # Migration note: generate_sql_with_cortex → generate_sql_with_bedrock
        question=bedrock_question,
        yaml_content=yaml_content,
        bedrock_model=bedrock_model,
    )

    if gen_sql and ok:
        t1 = _time.time()
        try:
            df = athena_query(gen_sql)  # Migration note: run_df → athena_query
            t2 = _time.time()
            text_summary = ""
            if not df.empty:
                try:
                    data_summary = (
                        f"Query returned {len(df)} rows:\n{df.head(5).to_string()}"
                        + (f"\n... and {len(df)-5} more rows" if len(df) > 5 else "")
                    )
                    bedrock_prompt = (
                        f"{full_context}"
                        f"Analyze this dealer question:\n\n"
                        f"Question: {query_text.strip()}\nData: {data_summary}\n\n"
                        f"Respond with THREE sections (ONLY these headers):\n\n"
                        f"**Descriptive** - What the data shows (2-3 sentences)\n\n"
                        f"**Prescriptive** - Recommendations (5-7 bullets with •)\n\n"
                        f"**Predictive** - Expected Impact 12-24 months (1-2 sentences)\n\n"
                        f"Use plain text only. Do NOT use LaTeX, $\\boxed{{...}}, or any math notation."
                    )
                    # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 8515-8519)
                    text_summary = _clean_llm_response(bedrock_complete(bedrock_prompt, model_id=bedrock_model))
                except Exception as exc:
                    logger.warning("[TEXT GEN] Failed: %s", str(exc)[:100])

            response_time_ms = (t2 - t1) * 1000
            response = {
                "layout":           "quick",
                "vendors_df":       df,
                "sql":              gen_sql,
                "source":           "generated_sql",
                "gen_ok":           ok,
                "gen_error":        gen_err,
                "text_summary":     text_summary,
                "response_time_ms": response_time_ms,
            }
            cache.set(query_text, response, response_time_ms=response_time_ms, cache_key_override=ctx_hash)
            return response
        except Exception as exc:
            return {
                "layout":           "sql_failed",
                "sql":              gen_sql,
                "error":            f"Generated SQL failed: {str(exc)}",
                "source":           "generated_sql",
                "gen_ok":           ok,
                "gen_error":        gen_err,
                "response_time_ms": (_time.time() - t1) * 1000,
            }

    elif gen_sql:
        return {
            "layout":           "sql_failed",
            "sql":              gen_sql,
            "error":            f"SQL validation failed: {gen_err}",
            "source":           "generated_sql",
            "gen_ok":           False,
            "gen_error":        gen_err,
            "response_time_ms": (_time.time() - t_start) * 1000,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # PATH 3 — Text-only Bedrock fallback
    # ═══════════════════════════════════════════════════════════════════════
    prompt = (
        f"{full_context}"
        f"You are a senior dealer business analyst. "
        f"Answer this question about dealer performance.\n\n"
        f"Question: {query_text.strip()}\n\n"
        f"Format EXACTLY as:\n\n"
        f"**Descriptive** - What the data shows\n"
        f"2-3 sentences with key insights.\n\n"
        f"**Prescriptive** - Recommendations & Actions\n"
        f"5-7 bullet points starting with •\n\n"
        f"**Predictive** - Expected Impact (12-24 months)\n"
        f"Financial outcomes and timeline.\n\n"
        f"Use plain text only. Do NOT use LaTeX, $\\boxed{{...}}, or any math notation."
    )
    try:
        # Migration note: bedrock_complete replaces SNOWFLAKE.CORTEX.COMPLETE (line 8583-8588)
        response_text    = _clean_llm_response(bedrock_complete(prompt, model_id=bedrock_model))
        response_time_ms = (_time.time() - t_start) * 1000

        if not response_text.strip():
            response_text = (
                "⚠️ Bedrock returned a generic response. Try more specific questions:\n"
                "• 'Which dealers are losing money?'\n"
                "• 'Show dealer health scorecard'\n"
                "• 'What are the top 5 dealers by revenue?'\n"
                f"\nYour question: '{query_text.strip()}'"
            )

        content = [{"type": "text", "text": response_text.strip()}] if response_text.strip() else []
        related = route_verified_query(query_text, model)
        if related and related.get("sql"):
            try:
                rdf      = athena_query(related["sql"].strip())  # Migration note: run_df → athena_query
                response = {
                    "message":          {"content": content},
                    "layout":           "quick",
                    "vendors_df":       rdf,
                    "text_summary":     response_text,
                    "sql":              related["sql"].strip(),
                    "related_data":     {"df": rdf, "sql": related["sql"].strip()},
                    "response_time_ms": response_time_ms,
                }
            except Exception:
                response = {"message": {"content": content}, "text_summary": response_text, "response_time_ms": response_time_ms}
        else:
            response = {"message": {"content": content}, "text_summary": response_text, "response_time_ms": response_time_ms}

        cache.set(query_text, response, response_time_ms=response_time_ms, cache_key_override=ctx_hash)
        return response

    except Exception as exc:
        return {
            "message":          {"content": [{"type": "text", "text": f"⚠️ Error: {str(exc)[:150]}"}]},
            "response_time_ms": (_time.time() - t_start) * 1000,
        }


# ============================================================================
# FORECAST PREDICTION TEXT
# Migration note: generate_forecast_prediction_text (line 8845).
# session parameter removed; athena_query replaces session.sql().to_pandas().
# SNOWFLAKE.CORTEX.COMPLETE with "mistral-7b" → bedrock_complete_mistral7b().
# SQL updated: LIMIT 12 unchanged (valid Athena syntax).
# ============================================================================

def generate_forecast_prediction_text(
    dealer_name:    str,
    forecast_result: Dict,
    anomaly_result:  Dict | None = None,
) -> Optional[str]:
    """
    Generate predictive insights about a dealer forecast using Bedrock.

    Migration note: session parameter removed. athena_query replaces
    session.sql().to_pandas(). bedrock_complete_mistral7b replaces
    SNOWFLAKE.CORTEX.COMPLETE with "mistral-7b" (source line 8940).
    """
    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        if not forecast_result or not forecast_result.get("success"):
            return None

        trend           = forecast_result.get("trend", "unknown").upper()
        change_pct      = forecast_result.get("change_percent", 0)
        confidence      = forecast_result.get("confidence", "unknown")
        mape            = forecast_result.get("mape", 0)
        recent_revenue  = forecast_result.get("recent_revenue", 0)
        forecast_weeks  = forecast_result.get("forecast_weeks", 8)

        additional_context = ""

        # Fetch margin data via Athena
        try:
            margin_sql = f"""
SELECT AVG(GROSS_PROFIT_MARGIN_PCT) AS avg_margin
FROM {_DB}.VW_GROSS_PROFIT_MARGIN
WHERE DEALER_NAME = '{dealer_name}'
LIMIT 12"""
            margin_df = athena_query(margin_sql)  # Migration note: session.sql().to_pandas() → athena_query
            if not margin_df.empty and "avg_margin" in margin_df.columns:
                margin = margin_df.at[0, "avg_margin"]
                if margin == margin:  # NaN check
                    additional_context += f"- Current profitability (margin): {float(margin):.1f}%\n"
        except Exception:
            pass

        # Fetch CCC data via Athena
        try:
            ccc_sql = f"""
SELECT AVG(CCC) AS avg_ccc
FROM {_DB}.VW_CASH_CONVERSION_CYCLE
WHERE DEALER_NAME = '{dealer_name}'
LIMIT 12"""
            ccc_df = athena_query(ccc_sql)  # Migration note: session.sql().to_pandas() → athena_query
            if not ccc_df.empty and "avg_ccc" in ccc_df.columns:
                ccc = ccc_df.at[0, "avg_ccc"]
                if ccc == ccc:
                    additional_context += f"- Cash cycle (days): {float(ccc):.0f}\n"
        except Exception:
            pass

        risk_context = ""
        if anomaly_result and anomaly_result.get("anomalies_count", 0) > 0:
            risk_context = (
                f"Performance volatility: {anomaly_result['anomalies_count']} anomalies detected "
                f"(Risk: {anomaly_result.get('risk_level', 'unknown')}). "
            )
        else:
            risk_context = "Performance is stable with no major anomalies. "

        forecast_summary = ""
        forecast_values  = forecast_result.get("forecast_values", [])
        if forecast_values and np is not None:
            avg_forecast = np.mean(forecast_values)
            forecast_summary = f"Revenue forecast average: ${avg_forecast:,.0f} over {forecast_weeks} weeks"

        prediction_prompt = f"""You are a business strategist analyzing dealer performance forecasts.

DEALER: {dealer_name}
FORECAST HORIZON: {forecast_weeks} weeks
CURRENT REVENUE: ${recent_revenue:,.0f}

FORECAST ANALYSIS:
- Trend: {trend}
- Expected Revenue Change: {change_pct:.1f}%
- {forecast_summary}
- Model Confidence: {confidence}
- Forecast Accuracy (MAPE): {mape:.1f}%

CONTEXT:
{additional_context}{risk_context}

Generate a concise 2-3 sentence prediction that:
1. Explains what the forecast means for revenue trajectory
2. Identifies 1 key business implication (cost management, cash flow, market positioning)
3. Suggests 1 immediate action if trend is negative, or opportunity if positive

Format: Clear business language, no technical jargon. Be specific with numbers where possible.""".strip()

        # Migration note: bedrock_complete_mistral7b replaces SNOWFLAKE.CORTEX.COMPLETE
        # with "mistral-7b" (source line 8940-8947)
        text = bedrock_complete_mistral7b(prediction_prompt)
        if text and len(text.strip()) > 10:
            return text.strip()

    except Exception as exc:
        logger.warning("[FORECAST PREDICTION] Error: %s", str(exc)[:100])
    return None


# ============================================================================
# DEALER NAME EXTRACTORS — no Snowflake dependency, moved as-is
# Source lines 8644-8708
# ============================================================================

def extract_dealer_name_from_question(question: str) -> Optional[str]:
    """Extract and normalise dealer name/ID from free-text question. Source line 8644."""
    if not question:
        return None
    q_lower = question.lower()
    match = re.search(r"\bdealer\s+(?:#)?(\d+)\b", q_lower)
    if match:
        dealer_id = match.group(1).strip().zfill(2)
        return f"Dealer {dealer_id}"
    match = re.search(r"\bdealer\b.*?(\d{1,3})\b", q_lower)
    if match:
        dealer_id = match.group(1).strip()
        if 1 <= int(dealer_id) <= 999:
            return f"Dealer {dealer_id.zfill(2)}"
    return None


def extract_dealer_name_from_context(
    response: Dict | None = None,
    question: str | None  = None,
) -> Optional[str]:
    """Extract dealer name from question or response dataframe. Source line 8685."""
    if question:
        dealer = extract_dealer_name_from_question(question)
        if dealer:
            return dealer
    if response and response.get("vendors_df") is not None:
        df = response["vendors_df"]
        if not df.empty:
            col = "DEALER_NAME" if "DEALER_NAME" in df.columns else ("dealer_name" if "dealer_name" in df.columns else None)
            if col and len(df) == 1:
                return str(df.iloc[0][col])
    return None


# ============================================================================
# APP ROUTING INITIALIZATION
# Migration note: initialize_app_routing (line 1570) — _semantic_run() removed
# (Snowflake-specific YAML sync); YAML loaded from S3 via load_yaml_model().
# ============================================================================

def initialize_app_routing() -> bool:
    """
    Initialize routing engine from YAML (loaded from S3 or local file).

    Migration note: replaces initialize_app_routing (source line 1570) which
    called _semantic_run() to sync Snowflake views into YAML. That sync is no
    longer needed — the YAML is maintained in S3 and loaded directly.
    """
    logger.info("[APP INIT] Loading YAML model from S3 / local file...")
    yaml_model = load_yaml_model()

    if not yaml_model or not isinstance(yaml_model, dict):
        logger.error("[APP INIT] ❌ Invalid YAML model — routing disabled")
        return False

    logger.info("[APP INIT] ✅ YAML loaded: %s (%d tables)",
                yaml_model.get("name", "unknown"), len(yaml_model.get("tables", [])))

    try:
        yaml_str = yaml.dump(yaml_model, default_flow_style=False)
        _initialize_routing_from_yaml(yaml_str)
        if _ROUTING_INITIALIZED and _INTENTS:
            logger.info("[APP INIT] ✅ Routing initialized: %d intents, %d tables",
                        len(_INTENTS), len(yaml_model.get("tables", [])))
            return True
        logger.error("[APP INIT] ❌ Routing vars not set after initialization")
        return False
    except Exception as exc:
        logger.error("[APP INIT] ❌ Routing init failed: %s", exc, exc_info=True)
        return False
