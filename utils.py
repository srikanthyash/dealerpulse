"""
utils.py — DynamoDB cache, chat persistence, long-term memory, and shared helpers.

Migration note: replaces GenieQueryCache, GenieChatPersistence, GenieLongTermMemory
(Snowflake-backed) and all shared helper functions throughout DealerFinalVersion.py.
"""
import hashlib
import json
import logging
import time
import uuid
from decimal import Decimal
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import boto3
import pandas as pd
import streamlit as st

from bedrock_client import bedrock_complete
from config_loader import get_aws_session, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_current_user() -> str:
    """
    Migration note: replaces CURRENT_USER() SQL function (source line 1733,
    2170, 2335). Athena has no CURRENT_USER() — use Streamlit session state.
    """
    return st.session_state.get("username", "anonymous")


def generate_session_id() -> str:
    """
    Migration note: replaces UUID_STRING() SQL function.
    Python-side generation per migration analysis.
    """
    return str(uuid.uuid4())


def safe_number(val, default: float = 0.0) -> float:
    """Safely convert *val* to float. Source lines 5291, 7068."""
    try:
        if val is None:
            return float(default)
        if isinstance(val, float) and val != val:   # NaN check
            return float(default)
        result = float(val)
        return result if result == result else float(default)
    except (ValueError, TypeError):
        return float(default)


def safe_int(val, default: int = 0) -> int:
    """Safely convert *val* to int. Source line 5300."""
    try:
        if val is None:
            return int(default)
        if isinstance(val, float) and val != val:
            return int(default)
        return int(val)
    except (ValueError, TypeError):
        return int(default)


def abbr_currency(val) -> str:
    """Abbreviate currency values (B / M / K). Source line 5309."""
    try:
        v = float(val)
        if abs(v) >= 1_000_000_000:
            return f"${v / 1_000_000_000:.1f}B"
        if abs(v) >= 1_000_000:
            return f"${v / 1_000_000:.1f}M"
        if abs(v) >= 1_000:
            return f"${v / 1_000:.1f}K"
        return f"${int(round(v)):,}"
    except Exception:
        return "$0"


def parse_analysis_sections(text: str):
    """
    Split an AI response into (descriptive, prescriptive, predictive) sections.
    Source line 9047.
    """
    import re as _re
    di = _re.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})descriptive(?:\*{0,2})(?:\s*[-:])?',  text, _re.IGNORECASE)
    pi = _re.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})prescriptive(?:\*{0,2})(?:\s*[-:])?', text, _re.IGNORECASE)
    ri = _re.search(r'\b(?:\d+\.?\s+)?(?:\*{0,2})predictive(?:\*{0,2})(?:\s*[-:])?',   text, _re.IGNORECASE)

    def _ext(sm, *others):
        if not sm:
            return ""
        s = sm.end()
        e = len(text)
        for o in others:
            if o and o.start() > sm.start():
                e = min(e, o.start())
        return text[s:e].strip().lstrip("*").strip()

    return _ext(di, pi, ri), _ext(pi, ri, di), _ext(ri, di, pi)


# ---------------------------------------------------------------------------
# DynamoDB helpers
# ---------------------------------------------------------------------------

def _dynamo_client() -> boto3.client:
    # Migration note: aws_session replaces snowflake_session
    return get_aws_session().client("dynamodb")


def _now_epoch() -> int:
    return int(time.time())


def _ttl_epoch(seconds: int) -> int:
    return _now_epoch() + seconds


def _to_dynamo(value):
    """Recursively convert Python types to DynamoDB-safe types."""
    if isinstance(value, bool):
        return {"BOOL": value}
    if isinstance(value, int):
        return {"N": str(value)}
    if isinstance(value, float):
        return {"N": str(Decimal(str(value)))}
    if isinstance(value, str):
        return {"S": value}
    if isinstance(value, dict):
        return {"M": {k: _to_dynamo(v) for k, v in value.items()}}
    if isinstance(value, list):
        return {"L": [_to_dynamo(i) for i in value]}
    if value is None:
        return {"NULL": True}
    return {"S": str(value)}


def _from_dynamo(item: dict):
    """Recursively deserialise a DynamoDB item to Python types."""
    if "S"    in item: return item["S"]
    if "N"    in item: return float(item["N"]) if "." in item["N"] else int(item["N"])
    if "BOOL" in item: return item["BOOL"]
    if "NULL" in item: return None
    if "M"    in item: return {k: _from_dynamo(v) for k, v in item["M"].items()}
    if "L"    in item: return [_from_dynamo(v) for v in item["L"]]
    return item


# ---------------------------------------------------------------------------
# DynamoQueryCache
# Migration note: replaces GenieQueryCache (Snowflake GENIE_QUERY_HISTORY)
# DynamoDB table: dealer_genie_query_cache (PK: QUESTION_HASH)
# ---------------------------------------------------------------------------

class DynamoQueryCache:
    """
    LRU in-memory cache with DynamoDB persistence for Genie query responses.

    Migration note: replaces GenieQueryCache backed by
    DEALER.INFORMATION_MART.GENIE_QUERY_HISTORY (Snowflake). All SQL DDL,
    PARSE_JSON, VARIANT columns, and SEARCH OPTIMIZATION are removed.
    RESPONSE_JSON stored as a plain STRING in DynamoDB (JSON-serialised dict).
    """

    SIMILARITY_THRESHOLD = 0.60
    MAX_MEMORY_SIZE = 100

    def __init__(self, ttl_seconds: int | None = None, similarity_threshold: float | None = None):
        cfg = get_config()
        self._ttl        = ttl_seconds        or cfg["dynamodb"]["cache_ttl_seconds"]
        self._threshold  = similarity_threshold or self.SIMILARITY_THRESHOLD
        self._table      = cfg["dynamodb"]["cache_table"]
        self._dynamo     = _dynamo_client()

        # In-memory LRU (same structure as source)
        self._cache: Dict[str, dict] = {}
        self._access_order: List[str] = []

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def _hash(self, question: str, prior: str = "") -> str:
        """MD5 hash of normalised question (+ prior for context-awareness). Source line ~1780."""
        combined = question.lower().strip()
        if prior:
            combined = f"{combined}|{prior.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _similarity(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

    # ------------------------------------------------------------------
    # In-memory LRU helpers
    # ------------------------------------------------------------------

    def _mem_get(self, q_hash: str) -> Optional[dict]:
        entry = self._cache.get(q_hash)
        if not entry:
            return None
        if time.time() - entry["timestamp"] > self._ttl:
            self._cache.pop(q_hash, None)
            return None
        self._access_order = [h for h in self._access_order if h != q_hash]
        self._access_order.append(q_hash)
        return entry["response"]

    def _mem_set(self, q_hash: str, question: str, response: dict, response_time_ms: float) -> None:
        if len(self._cache) >= self.MAX_MEMORY_SIZE and q_hash not in self._cache:
            oldest = self._access_order.pop(0)
            self._cache.pop(oldest, None)
        self._cache[q_hash] = {
            "response":         response,
            "timestamp":        time.time(),
            "question":         question,
            "response_time_ms": response_time_ms,
            "verified_name":    response.get("verified_name"),
        }
        if q_hash not in self._access_order:
            self._access_order.append(q_hash)

    # ------------------------------------------------------------------
    # DynamoDB persistence
    # ------------------------------------------------------------------

    def _dynamo_get(self, q_hash: str) -> Optional[dict]:
        """
        Exact hash lookup in DynamoDB.
        Migration note: replaces
          SELECT RESPONSE_JSON FROM GENIE_QUERY_HISTORY WHERE QUESTION_HASH = '...'
        """
        try:
            resp = self._dynamo.get_item(
                TableName=self._table,
                Key={"QUESTION_HASH": {"S": q_hash}},
                ProjectionExpression="RESPONSE_JSON,RESPONSE_TIME_MS",
            )
            item = resp.get("Item")
            if not item:
                return None
            raw = item["RESPONSE_JSON"]["S"]
            return json.loads(raw)
        except Exception as exc:
            logger.debug("[DynamoQueryCache] dynamo_get failed: %s", exc)
            return None

    def _dynamo_put(self, q_hash: str, question: str, response: dict, response_time_ms: float) -> None:
        """
        Persist cache entry to DynamoDB with TTL.
        Migration note: replaces
          INSERT INTO GENIE_QUERY_HISTORY ... PARSE_JSON('{json_escaped}') ...
        VARIANT → plain STRING (JSON-serialised).
        TTL attribute EXPIRES_AT is a Unix epoch int (DynamoDB TTL field).
        """
        minimal = {
            "layout":           str(response.get("layout",  "unknown"))[:50],
            "source":           str(response.get("source",  "unknown"))[:50],
            "sql":              str(response.get("sql", "") or "").replace("\n", " ")[:2000],
            "gen_ok":           bool(response.get("gen_ok", False)),
            "response_time_ms": float(response_time_ms),
            "verified_name":    response.get("verified_name", ""),
        }
        try:
            self._dynamo.put_item(
                TableName=self._table,
                Item={
                    "QUESTION_HASH":    {"S": q_hash},
                    "QUESTION":         {"S": question[:500]},
                    "RESPONSE_JSON":    {"S": json.dumps(minimal, ensure_ascii=True)},
                    "USER_NAME":        {"S": get_current_user()},
                    "RESPONSE_TIME_MS": {"N": str(response_time_ms)},
                    "CREATED_AT":       {"N": str(_now_epoch())},
                    "EXPIRES_AT":       {"N": str(_ttl_epoch(self._ttl))},
                },
            )
        except Exception as exc:
            logger.warning("[DynamoQueryCache] dynamo_put failed: %s", exc)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def find_similar(self, question: str, limit: int = 5) -> List[dict]:
        """
        Find similar cached questions (in-memory + DynamoDB).
        Migration note: replaces GenieQueryCache.find_similar_questions()
        which queried GENIE_QUERY_HISTORY with a full-table scan + Python-side
        similarity scoring.
        """
        similarities = []

        # 1. In-memory
        for entry in self._cache.values():
            if time.time() - entry["timestamp"] > self._ttl:
                continue
            score = self._similarity(question, entry.get("question", ""))
            if score >= self._threshold:
                similarities.append({
                    "question":      entry["question"],
                    "similarity_score": score,
                    "response":      entry["response"],
                    "source":        "in-memory",
                })

        # 2. DynamoDB scan (limited to 100 recent items for cost control)
        try:
            resp = self._dynamo.scan(
                TableName=self._table,
                ProjectionExpression="QUESTION,RESPONSE_JSON,RESPONSE_TIME_MS",
                Limit=100,
            )
            for item in resp.get("Items", []):
                db_q = item.get("QUESTION", {}).get("S", "")
                score = self._similarity(question, db_q)
                if score >= self._threshold:
                    raw = item.get("RESPONSE_JSON", {}).get("S", "{}")
                    similarities.append({
                        "question":         db_q,
                        "similarity_score": score,
                        "response":         json.loads(raw),
                        "source":           "dynamodb",
                    })
        except Exception as exc:
            logger.debug("[DynamoQueryCache] similarity scan failed: %s", exc)

        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:limit]

    # ------------------------------------------------------------------
    # Public get / set / find_by_verified_query
    # ------------------------------------------------------------------

    def get(self, question: str, allow_semantic: bool = True,
            cache_key_override: str = None) -> Optional[dict]:
        """
        Retrieve a cached response. Exact hash first; semantic similarity fallback.
        Migration note: replaces GenieQueryCache.get().
        """
        q_hash = cache_key_override or self._hash(question)

        # Exact — memory
        resp = self._mem_get(q_hash)
        if resp is not None:
            logger.debug("[DynamoQueryCache] Exact memory hit: %s", q_hash)
            return resp

        # Exact — DynamoDB
        resp = self._dynamo_get(q_hash)
        if resp is not None:
            logger.debug("[DynamoQueryCache] Exact DynamoDB hit: %s", q_hash)
            self._mem_set(q_hash, question, resp, resp.get("response_time_ms", 0))
            return resp

        if not allow_semantic:
            return None

        # Semantic
        matches = self.find_similar(question, limit=1)
        if matches:
            best = matches[0]
            logger.debug("[DynamoQueryCache] Semantic hit (%.0f%%): %s",
                         best["similarity_score"] * 100, best["question"][:60])
            self._mem_set(q_hash, question, best["response"], 0)
            return best["response"]

        return None

    def set(self, question: str, response: dict,
            response_time_ms: float = 0.0,
            cache_key_override: str = None) -> None:
        """
        Store response in memory and DynamoDB.
        Migration note: replaces GenieQueryCache.set() which used
        INSERT INTO ... PARSE_JSON(...).
        """
        q_hash = cache_key_override or self._hash(question)
        self._mem_set(q_hash, question, response, response_time_ms)
        self._dynamo_put(q_hash, question, response, response_time_ms)

    def find_by_verified_query(self, verified_query_name: str) -> Optional[dict]:
        """
        Find cached response for the same verified_query_name (in-memory only).
        Migration note: replaces GenieQueryCache.find_by_verified_query().
        """
        if not verified_query_name:
            return None
        for entry in self._cache.values():
            if time.time() - entry["timestamp"] > self._ttl:
                continue
            resp = entry.get("response", {})
            if (resp.get("verified_name") == verified_query_name or
                    entry.get("verified_name") == verified_query_name):
                return resp
        return None

    def get_popular_questions(self, limit: int = 10, days: int = 7) -> pd.DataFrame:
        """
        Return the most frequently asked questions from the last *days* days.

        Migration note: replaces GenieQueryCache.get_popular_questions() which ran
          SELECT QUESTION, COUNT(*) AS FREQUENCY, AVG(RESPONSE_TIME_MS), ...
          FROM GENIE_QUERY_HISTORY
          WHERE CREATED_AT >= DATEADD('day', -N, CURRENT_TIMESTAMP())
          GROUP BY QUESTION ORDER BY FREQUENCY DESC LIMIT N
        DynamoDB has no GROUP BY — scan then aggregate in Python.
        """
        from collections import defaultdict

        cutoff = _now_epoch() - days * 86400
        try:
            resp = self._dynamo.scan(
                TableName=self._table,
                FilterExpression="CREATED_AT >= :c",
                ExpressionAttributeValues={":c": {"N": str(cutoff)}},
                ProjectionExpression="QUESTION,RESPONSE_TIME_MS,CREATED_AT",
            )
            items = resp.get("Items", [])
        except Exception as exc:
            logger.warning("[DynamoQueryCache] get_popular_questions scan failed: %s", exc)
            return pd.DataFrame()

        if not items:
            return pd.DataFrame()

        groups: dict = defaultdict(lambda: {"count": 0, "rt_sum": 0.0, "last_asked": 0})
        for item in items:
            q  = item.get("QUESTION",         {}).get("S", "").strip()
            rt = float(item.get("RESPONSE_TIME_MS", {}).get("N", "0") or 0)
            ts = int(item.get("CREATED_AT",   {}).get("N", "0") or 0)
            if not q:
                continue
            groups[q]["count"]      += 1
            groups[q]["rt_sum"]     += rt
            groups[q]["last_asked"]  = max(groups[q]["last_asked"], ts)

        rows = []
        for q, agg in groups.items():
            n      = agg["count"]
            avg_rt = round(agg["rt_sum"] / n, 2) if n else 0.0
            rows.append({
                "QUESTION":             q,
                "FREQUENCY":            n,
                "AVG_RESPONSE_TIME_MS": avg_rt,
                "LAST_ASKED":           pd.Timestamp(agg["last_asked"], unit="s"),
                "TOTAL_TIME_SAVED_MS":  round(avg_rt * n, 0),
            })

        rows.sort(key=lambda x: x["FREQUENCY"], reverse=True)
        return pd.DataFrame(rows[:limit])

    def get_query_stats(self, days: int = 7) -> dict:
        """
        Return aggregate query statistics for the last *days* days.

        Migration note: replaces GenieQueryCache.get_query_stats() which ran
          SELECT COUNT(DISTINCT QUESTION), COUNT(*), COUNT(DISTINCT USER_NAME),
                 AVG/MIN/MAX(RESPONSE_TIME_MS),
                 SUM(RESPONSE_TIME_MS) / 1000.0 / 60.0 AS TOTAL_TIME_MINUTES
          FROM GENIE_QUERY_HISTORY
          WHERE CREATED_AT >= DATEADD('day', -N, CURRENT_TIMESTAMP())
        Aggregated in Python after a DynamoDB scan.
        """
        cutoff = _now_epoch() - days * 86400
        try:
            resp = self._dynamo.scan(
                TableName=self._table,
                FilterExpression="CREATED_AT >= :c",
                ExpressionAttributeValues={":c": {"N": str(cutoff)}},
                ProjectionExpression="QUESTION,USER_NAME,RESPONSE_TIME_MS",
            )
            items = resp.get("Items", [])
        except Exception as exc:
            logger.warning("[DynamoQueryCache] get_query_stats scan failed: %s", exc)
            return {}

        if not items:
            return {}

        unique_qs:    set  = set()
        unique_users: set  = set()
        rt_vals:      list = []

        for item in items:
            q    = item.get("QUESTION",         {}).get("S", "")
            user = item.get("USER_NAME",         {}).get("S", "")
            rt   = float(item.get("RESPONSE_TIME_MS", {}).get("N", "0") or 0)
            if q:
                unique_qs.add(q)
            if user:
                unique_users.add(user)
            rt_vals.append(rt)

        total = len(items)
        avg_rt = round(sum(rt_vals) / total, 2) if total else 0.0
        return {
            "unique_questions":     len(unique_qs),
            "total_queries":        total,
            "unique_users":         len(unique_users),
            "avg_response_time_ms": avg_rt,
            "min_response_time_ms": round(min(rt_vals), 2) if rt_vals else 0.0,
            "max_response_time_ms": round(max(rt_vals), 2) if rt_vals else 0.0,
            "total_time_minutes":   round(sum(rt_vals) / 1000.0 / 60.0, 2),
        }

    def stats(self) -> dict:
        """Return cache statistics. Migration note: replaces GenieQueryCache.stats()."""
        valid = {k: v for k, v in self._cache.items()
                 if time.time() - v["timestamp"] < self._ttl}
        return {
            "memory_cache_size": len(valid),
            "size":              len(valid),
            "max_size":          self.MAX_MEMORY_SIZE,
        }


# ---------------------------------------------------------------------------
# DynamoChatPersistence
# Migration note: replaces GenieChatPersistence (Snowflake GENIE_CHAT_SESSIONS)
# DynamoDB table: dealer_genie_chat_sessions (PK: SESSION_ID, SK: TURN_INDEX)
# ---------------------------------------------------------------------------

class DynamoChatPersistence:
    """
    Persists Genie conversation turns to DynamoDB so users can resume sessions.

    Migration note: replaces GenieChatPersistence backed by
    DEALER.INFORMATION_MART.GENIE_CHAT_SESSIONS (Snowflake). All CREATE TABLE
    DDL, CONSTRAINT / PRIMARY KEY syntax, and TIMESTAMP_NTZ are removed.
    DynamoDB table already exists (provisioned in infrastructure setup).
    TTL via EXPIRES_AT attribute (Unix epoch int).
    """

    RESTORE_DAYS   = 2
    MAX_TURNS_RESTORE = 40

    def __init__(self):
        cfg = get_config()
        self._table      = cfg["dynamodb"]["sessions_table"]
        self._ttl_days   = cfg["dynamodb"]["session_ttl_days"]
        self._ttl_secs   = self._ttl_days * 86400
        self._dynamo     = _dynamo_client()
        # Migration note: replaces session.sql("SELECT CURRENT_USER()") at lines 2170, 2335
        self._user       = get_current_user()

    # ------------------------------------------------------------------
    # Save a turn
    # ------------------------------------------------------------------

    def save_turn(
        self,
        session_id:    str,
        turn_index:    int,
        role:          str,
        content:       str,
        sql_used:      str = "",
        source:        str = "",
        session_label: str = "",
    ) -> None:
        """
        Write one message turn to DynamoDB.
        Migration note: replaces GenieChatPersistence.save_turn() which used
          INSERT INTO GENIE_CHAT_SESSIONS (...) VALUES (...)
        VARIANT → plain STRING attributes.
        """
        try:
            self._dynamo.put_item(
                TableName=self._table,
                Item={
                    "SESSION_ID":    {"S": session_id},
                    "TURN_INDEX":    {"N": str(turn_index)},
                    "USER_NAME":     {"S": self._user},
                    "ROLE":          {"S": role},
                    "CONTENT":       {"S": content[:4000]},
                    "SQL_USED":      {"S": (sql_used or "")[:2000]},
                    "SOURCE":        {"S": source or ""},
                    "SESSION_LABEL": {"S": session_label or ""},
                    "CREATED_AT":    {"N": str(_now_epoch())},
                    "EXPIRES_AT":    {"N": str(_ttl_epoch(self._ttl_secs))},
                },
            )
        except Exception as exc:
            logger.warning("[DynamoChatPersistence] save_turn failed: %s", exc)

    # ------------------------------------------------------------------
    # Load sessions list
    # ------------------------------------------------------------------

    def load_all_sessions(self) -> List[dict]:
        """
        Return all distinct sessions for this user from the last RESTORE_DAYS days.
        Migration note: replaces GenieChatPersistence.load_all_sessions() which used
          SELECT SESSION_ID, MAX(SESSION_LABEL), MAX(CREATED_AT), COUNT(*)
          FROM GENIE_CHAT_SESSIONS WHERE USER_NAME = '...'
          AND CREATED_AT >= DATEADD('day', -2, CURRENT_TIMESTAMP())
        DynamoDB has no GROUP BY — we aggregate in Python.
        Falls back to a FilterExpression scan if the GSI does not exist yet.
        """
        cutoff = _now_epoch() - self.RESTORE_DAYS * 86400
        items: list = []
        gsi_ok = False
        try:
            resp = self._dynamo.query(
                TableName=self._table,
                IndexName="USER_NAME-CREATED_AT-index",
                KeyConditionExpression="USER_NAME = :u AND CREATED_AT >= :c",
                ExpressionAttributeValues={
                    ":u": {"S": self._user},
                    ":c": {"N": str(cutoff)},
                },
                ProjectionExpression="SESSION_ID,SESSION_LABEL,CREATED_AT,TURN_INDEX",
            )
            items = resp.get("Items", [])
            gsi_ok = True
        except Exception as exc:
            logger.debug("[DynamoChatPersistence] GSI query failed (%s), falling back to scan", exc)

        if not gsi_ok:
            # GSI not yet created — scan the table and filter in Python
            try:
                resp = self._dynamo.scan(
                    TableName=self._table,
                    FilterExpression="USER_NAME = :u AND CREATED_AT >= :c",
                    ExpressionAttributeValues={
                        ":u": {"S": self._user},
                        ":c": {"N": str(cutoff)},
                    },
                    ProjectionExpression="SESSION_ID,SESSION_LABEL,CREATED_AT,TURN_INDEX",
                )
                items = resp.get("Items", [])
            except Exception as exc:
                logger.warning("[DynamoChatPersistence] load_all_sessions failed: %s", exc)
                return []

        # Aggregate by SESSION_ID in Python (replaces GROUP BY)
        sessions: Dict[str, dict] = {}
        for item in items:
            sid   = item.get("SESSION_ID", {}).get("S", "")
            label = item.get("SESSION_LABEL", {}).get("S", "")
            ts    = int(item.get("CREATED_AT", {}).get("N", "0"))
            if sid not in sessions or ts > sessions[sid]["_last_ts"]:
                sessions.setdefault(sid, {"_last_ts": 0, "turn_count": 0})
                sessions[sid]["session_id"]    = sid
                sessions[sid]["session_label"] = label or "Previous chat"
                sessions[sid]["_last_ts"]      = max(sessions[sid]["_last_ts"], ts)
            sessions[sid]["turn_count"] = sessions[sid].get("turn_count", 0) + 1

        now = _now_epoch()
        result = []
        for s in sessions.values():
            age_h = (now - s["_last_ts"]) / 3600
            result.append({
                "session_id":    s["session_id"],
                "session_label": s["session_label"],
                "age_hours":     age_h,
                "turn_count":    s["turn_count"],
            })
        result.sort(key=lambda x: x["age_hours"])
        return result

    # ------------------------------------------------------------------
    # Load messages for one session
    # ------------------------------------------------------------------

    def load_session_messages(self, session_id: str) -> List[dict]:
        """
        Fetch all turns for *session_id* ordered by TURN_INDEX.
        Migration note: replaces GenieChatPersistence.load_session_messages()
          SELECT ROLE, CONTENT, ... FROM GENIE_CHAT_SESSIONS
          WHERE SESSION_ID = '...' ORDER BY TURN_INDEX LIMIT 40
        """
        try:
            resp = self._dynamo.query(
                TableName=self._table,
                KeyConditionExpression="SESSION_ID = :s",
                ExpressionAttributeValues={":s": {"S": session_id}},
                ScanIndexForward=True,
                Limit=self.MAX_TURNS_RESTORE,
            )
            messages = []
            for item in resp.get("Items", []):
                messages.append({
                    "role":      item.get("ROLE",     {}).get("S", ""),
                    "content":   item.get("CONTENT",  {}).get("S", ""),
                    "timestamp": pd.Timestamp(int(item.get("CREATED_AT", {}).get("N", "0")), unit="s"),
                    "response":  None,
                    "source":    item.get("SOURCE",   {}).get("S", ""),
                    "sql_used":  item.get("SQL_USED",  {}).get("S", ""),
                })
            return messages
        except Exception as exc:
            logger.warning("[DynamoChatPersistence] load_session_messages failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Purge old sessions
    # ------------------------------------------------------------------

    def purge_old_sessions(self, keep_days: int | None = None) -> None:
        """
        Delete expired turns (DynamoDB TTL handles this automatically via
        EXPIRES_AT). This method is a no-op kept for API compatibility.
        Migration note: replaces
          DELETE FROM GENIE_CHAT_SESSIONS WHERE CREATED_AT < DATEADD('day',-N,...)
        DynamoDB TTL is set on every put_item via EXPIRES_AT — no manual deletes needed.
        """
        logger.debug("[DynamoChatPersistence] purge_old_sessions: TTL handles expiry automatically.")


# ---------------------------------------------------------------------------
# GenieLongTermMemory
# Migration note: replaces GenieLongTermMemory (Snowflake + Cortex)
# Now uses DynamoDB query history + Bedrock for fact extraction.
# ---------------------------------------------------------------------------

class GenieLongTermMemory:
    """
    Extract durable facts from the user's past Genie questions and inject
    them as context into future AI prompts.

    Migration note: replaces GenieLongTermMemory which queried
    DEALER.INFORMATION_MART.GENIE_QUERY_HISTORY via Snowflake and called
    SNOWFLAKE.CORTEX.COMPLETE for extraction.
    Now queries DynamoDB dealer_genie_query_cache and calls bedrock_complete().
    """

    def __init__(self, max_history: int = 20):
        cfg = get_config()
        self._table        = cfg["dynamodb"]["cache_table"]
        self._dynamo       = _dynamo_client()
        self._user         = get_current_user()
        self._max_history  = max_history
        self._memories: List[str] = []
        self._build_memory_from_history()

    def _build_memory_from_history(self) -> None:
        """
        Pull recent questions from DynamoDB and extract facts via Bedrock.
        Migration note: replaces Snowflake query + CORTEX.COMPLETE call.
        DATEADD('day',-30,...) → Python epoch arithmetic.
        """
        cutoff = _now_epoch() - 30 * 86400
        try:
            resp = self._dynamo.scan(
                TableName=self._table,
                FilterExpression="USER_NAME = :u AND CREATED_AT >= :c",
                ExpressionAttributeValues={
                    ":u": {"S": self._user},
                    ":c": {"N": str(cutoff)},
                },
                ProjectionExpression="QUESTION,RESPONSE_JSON,CREATED_AT",
                Limit=self._max_history * 3,
            )
            items = resp.get("Items", [])
        except Exception as exc:
            logger.debug("[GenieLongTermMemory] DynamoDB scan failed: %s", exc)
            return

        # De-duplicate by question text (replaces ROW_NUMBER() OVER PARTITION BY in source)
        seen: Dict[str, dict] = {}
        for item in items:
            q   = item.get("QUESTION",      {}).get("S", "").strip()
            ts  = int(item.get("CREATED_AT", {}).get("N", "0"))
            raw = item.get("RESPONSE_JSON", {}).get("S", "{}")
            if q and (q not in seen or ts > seen[q]["ts"]):
                seen[q] = {"ts": ts, "raw": raw}

        if not seen:
            return

        lines = []
        for q, meta in list(seen.items())[: self._max_history]:
            sql_hint = ""
            try:
                resp_data = json.loads(meta["raw"])
                sql_hint  = (resp_data.get("sql") or "")[:100]
            except Exception:
                pass
            lines.append(f"Q: {q}" + (f" [used: {sql_hint}]" if sql_hint else ""))

        transcript = "\n".join(lines)
        extract_prompt = (
            "You are analyzing a user's dealer analytics query history.\n"
            "Extract 3-8 SHORT facts about their interests and what they've discovered.\n\n"
            f"Query history (last {len(lines)} questions):\n{transcript}\n\n"
            "Rules:\n"
            "- Each fact must be ONE sentence max\n"
            "- Focus on: which dealers they care about, what metrics matter to them,\n"
            "  patterns they've discovered, topics they frequently ask about\n"
            "- Only include facts useful as future context\n"
            "- If nothing worth remembering, respond NONE\n\n"
            "Respond with facts only, one per line, no bullets:"
        )

        # Migration note: replaces session.sql(SNOWFLAKE.CORTEX.COMPLETE) at line 2259
        raw_text = bedrock_complete(extract_prompt)
        if not raw_text or raw_text.strip().upper() == "NONE":
            return

        self._memories = [
            line.strip()
            for line in raw_text.split("\n")
            if line.strip() and line.strip().upper() != "NONE" and len(line.strip()) > 10
        ][:8]

        logger.debug("[GenieLongTermMemory] Built %d facts for %s", len(self._memories), self._user)

    def get_memory_prefix(self) -> str:
        """Return context string for injection into Bedrock prompts."""
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


# ---------------------------------------------------------------------------
# Streamlit-cached singletons
# Callers should import these functions rather than instantiating classes directly.
# ---------------------------------------------------------------------------

@st.cache_resource
def get_query_cache() -> DynamoQueryCache:
    """Return the shared DynamoQueryCache instance."""
    return DynamoQueryCache()


@st.cache_resource
def get_chat_persistence() -> DynamoChatPersistence:
    """Return the shared DynamoChatPersistence instance."""
    return DynamoChatPersistence()
