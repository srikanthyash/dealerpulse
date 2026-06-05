"""
athena_client.py — all Athena query execution for DealerPulse.

Migration note: replaces session.sql(...).to_pandas() and session.sql(...).collect()
throughout DealerFinalVersion.py.
"""
import io
import logging
import time

import boto3
import pandas as pd

from config_loader import get_aws_session, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SQL dialect translation helpers
# Migration note: translate every Snowflake-specific function to Athena/Presto
# ---------------------------------------------------------------------------

def _translate_sql(sql: str) -> str:
    """
    Apply Snowflake → Athena SQL dialect translations in-place.

    Covers every transformation listed in the migration analysis:
      DATEADD / DATE_TRUNC / DATEDIFF / IFF / ::DATE cast /
      TIMESTAMP_NTZ / YEAR(CURRENT_DATE()) / UUID_STRING() /
      CURRENT_USER() / ADD SEARCH OPTIMIZATION / EXPLAIN USING TEXT
    """
    import re

    # Strip Snowflake-style catalog prefix: dealer_db.dealer_db.TABLE → dealer_db.TABLE
    # Athena catalog is AwsDataCatalog; the first "dealer_db" segment is not a valid Athena catalog.
    sql = re.sub(r"\bdealer_db\.dealer_db\.", "dealer_db.", sql, flags=re.IGNORECASE)

    # DATEADD('unit', n, expr)  →  date_add('unit', n, expr)
    sql = re.sub(r"\bDATEADD\s*\(", "date_add(", sql, flags=re.IGNORECASE)

    # DATE_TRUNC('UNIT', expr::DATE)  →  date_trunc('unit', cast(expr as date))
    # Case-sensitive: only match Snowflake uppercase DATE_TRUNC, never re-translate
    # already-lowercased date_trunc (which would corrupt cast(...as date) expressions).
    def _trunc_sub(m):
        unit = m.group(1)
        col = m.group(2).strip()
        col = re.sub(r"::\s*DATE", "", col, flags=re.IGNORECASE).strip()
        return f"date_trunc('{unit.lower()}', cast({col} as date))"
    sql = re.sub(
        r"\bDATE_TRUNC\s*\(\s*'([^']+)'\s*,\s*([^)]+)\)",
        _trunc_sub,
        sql,
    )

    # DATEDIFF('unit', col1, col2)  →  date_diff('unit', col1, col2)
    sql = re.sub(r"\bDATEDIFF\s*\(", "date_diff(", sql, flags=re.IGNORECASE)

    # IFF(cond, a, b)  →  IF(cond, a, b)
    sql = re.sub(r"\bIFF\s*\(", "IF(", sql, flags=re.IGNORECASE)

    # 'value'::DATE  →  cast('value' as date)
    sql = re.sub(
        r"'([^']+)'\s*::\s*DATE",
        r"cast('\1' as date)",
        sql,
        flags=re.IGNORECASE,
    )

    # col::DATE  →  cast(col as date)   (bare identifier cast)
    sql = re.sub(
        r"\b(\w+)\s*::\s*DATE\b",
        r"cast(\1 as date)",
        sql,
        flags=re.IGNORECASE,
    )

    # TIMESTAMP_NTZ  →  TIMESTAMP
    sql = re.sub(r"\bTIMESTAMP_NTZ\b", "TIMESTAMP", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bTIMESTAMP_LTZ\b", "TIMESTAMP", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bTIMESTAMP_TZ\b",  "TIMESTAMP", sql, flags=re.IGNORECASE)

    # YEAR(CURRENT_DATE())  →  year(current_date)
    sql = re.sub(r"\bYEAR\s*\(\s*CURRENT_DATE\s*\(\s*\)\s*\)", "year(current_date)", sql, flags=re.IGNORECASE)

    # CURRENT_TIMESTAMP()  →  current_timestamp   (Athena uses no parens)
    sql = re.sub(r"\bCURRENT_TIMESTAMP\s*\(\s*\)", "current_timestamp", sql, flags=re.IGNORECASE)

    # CURRENT_DATE()  →  current_date
    sql = re.sub(r"\bCURRENT_DATE\s*\(\s*\)", "current_date", sql, flags=re.IGNORECASE)

    # UUID_STRING()  →  remove (caller should generate uuid in Python)
    sql = re.sub(r"\bUUID_STRING\s*\(\s*\)", "''", sql, flags=re.IGNORECASE)

    # CURRENT_USER()  →  '' (caller substitutes from session state)
    sql = re.sub(r"\bCURRENT_USER\s*\(\s*\)", "''", sql, flags=re.IGNORECASE)

    # ADD SEARCH OPTIMIZATION  →  remove entirely
    sql = re.sub(r"\bADD\s+SEARCH\s+OPTIMIZATION\b[^;]*", "", sql, flags=re.IGNORECASE)

    # EXPLAIN USING TEXT <sql>  →  remove (caught by InvalidRequestException handler)
    sql = re.sub(r"\bEXPLAIN\s+USING\s+TEXT\b", "", sql, flags=re.IGNORECASE)

    # Date string literals in BETWEEN → date 'YYYY-MM-DD'
    # Athena won't implicitly cast varchar to date; Snowflake did.
    sql = re.sub(
        r"\bBETWEEN\s*'(\d{4}-\d{2}-\d{2})'\s*AND\s*'(\d{4}-\d{2}-\d{2})'",
        r"BETWEEN date '\1' AND date '\2'",
        sql,
        flags=re.IGNORECASE,
    )
    # Date string literals in comparison operators → date 'YYYY-MM-DD'
    sql = re.sub(
        r"(>=|<=|<>|!=|(?<![<>!])=|(?<![<>])>(?!=)|(?<![<>])<(?!=))\s*'(\d{4}-\d{2}-\d{2})'",
        r"\1 date '\2'",
        sql,
        flags=re.IGNORECASE,
    )

    # Legacy fully-qualified refs (DEALER.INFORMATION_MART.VW_xxx) → {database}.VW_xxx
    # No schema layer in Athena; views live directly in the database.
    cfg = get_config()
    db = cfg["athena"]["database"]
    sql = re.sub(
        r"\bDEALER_DB\.",
        f"{db}.",
        sql,
        flags=re.IGNORECASE,
    )

    # PRIMARY KEY / CONSTRAINT DDL  →  remove (DynamoDB/Athena don't use DDL keys)
    sql = re.sub(
        r"\bPRIMARY\s+KEY\s*\([^)]*\)",
        "",
        sql,
        flags=re.IGNORECASE,
    )

    return sql


# ---------------------------------------------------------------------------
# Core client
# ---------------------------------------------------------------------------

class AthenaClient:
    """
    Thin wrapper around boto3 Athena that runs a query and returns a DataFrame.

    Migration note: replaces all session.sql(query).to_pandas() call sites.
    """

    def __init__(self):
        cfg = get_config()
        self._database  = cfg["athena"]["database"]
        self._workgroup = cfg["athena"]["workgroup"]
        self._s3_output = cfg["athena"]["s3_output"]
        self._poll_interval = cfg["athena"]["poll_interval_seconds"]
        self._poll_max_wait = cfg["athena"]["poll_max_wait_seconds"]

        # Migration note: aws_session replaces snowflake_session
        session: boto3.Session = get_aws_session()
        self._client = session.client("athena")
        self._s3     = session.client("s3")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_query(self, sql: str) -> pd.DataFrame:
        """
        Execute *sql* against Athena and return results as a DataFrame.

        Migration note:
          replaces  session.sql(sql).to_pandas()
          replaces  session.sql(sql).collect()  (returns DataFrame instead of list of Row)

        Applies Snowflake → Athena SQL dialect translations automatically.
        Uses exponential backoff polling up to poll_max_wait_seconds.
        """
        translated = _translate_sql(sql)
        execution_id = self._start_query(translated)
        self._wait_for_query(execution_id)
        return self._fetch_results(execution_id)

    def run_query_collect(self, sql: str) -> list:
        """
        Convenience wrapper that returns rows as a list of dicts.
        Migration note: replaces session.sql(sql).collect() where callers iterate rows.
        """
        df = self.run_query(sql)
        return df.to_dict(orient="records")

    def get_glue_column_comments(self, table_name: str) -> dict:
        """
        Return {column_name: comment} for *table_name* via Glue Data Catalog.

        Migration note: replaces INFORMATION_SCHEMA.COLUMNS COMMENT lookup
        (line 1143-1149 in source). Athena information_schema has no COMMENT
        column — Glue API is the correct replacement.
        """
        cfg = get_config()
        schema = cfg["athena"]["database"]
        glue = get_aws_session().client("glue")
        try:
            resp = glue.get_table(DatabaseName=schema, Name=table_name.lower())
            cols = resp["Table"]["StorageDescriptor"]["Columns"]
            return {c["Name"]: c.get("Comment", "") for c in cols}
        except Exception as exc:
            logger.warning("[athena_client] Glue column comment lookup failed for %s: %s", table_name, exc)
            return {}

    def list_views(self, prefix: str = "vw_") -> list:
        """
        Return view names from Glue Data Catalog filtered by name prefix.

        Migration note: replaces INFORMATION_SCHEMA.VIEWS query — Athena
        information_schema is not available; Glue API is the correct source.
        """
        cfg  = get_config()
        db   = cfg["athena"]["database"]
        glue = get_aws_session().client("glue")
        views = []
        try:
            paginator = glue.get_paginator("get_tables")
            for page in paginator.paginate(DatabaseName=db):
                for tbl in page.get("TableList", []):
                    name = tbl.get("Name", "")
                    if name.lower().startswith(prefix.lower()):
                        views.append(name)
        except Exception as exc:
            logger.warning("[athena_client] Glue list_views failed: %s", exc)
        return sorted(views)

    def list_columns(self, view_name: str) -> pd.DataFrame:
        """
        Return column metadata for *view_name* via Glue Data Catalog.

        Migration note: replaces INFORMATION_SCHEMA.COLUMNS query — Athena
        information_schema is not available; Glue API is the correct source.
        Column comments are also available via StorageDescriptor (same API call).
        """
        cfg  = get_config()
        db   = cfg["athena"]["database"]
        glue = get_aws_session().client("glue")
        try:
            resp = glue.get_table(DatabaseName=db, Name=view_name.lower())
            cols = resp["Table"]["StorageDescriptor"]["Columns"]
            return pd.DataFrame([
                {"column_name": c["Name"], "data_type": c.get("Type", "")}
                for c in cols
            ])
        except Exception as exc:
            logger.warning("[athena_client] Glue list_columns failed for %s: %s", view_name, exc)
            return pd.DataFrame(columns=["column_name", "data_type"])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_query(self, sql: str) -> str:
        response = self._client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": self._database},
            WorkGroup=self._workgroup,
            ResultConfiguration={"OutputLocation": self._s3_output},
        )
        qid = response["QueryExecutionId"]
        logger.debug("[athena_client] Started query %s", qid)
        return qid

    def _wait_for_query(self, execution_id: str) -> None:
        """
        Poll with exponential backoff until the query succeeds or fails.
        Raises RuntimeError on FAILED/CANCELLED, TimeoutError on max wait exceeded.
        """
        elapsed   = 0.0
        wait      = float(self._poll_interval)
        max_wait  = float(self._poll_max_wait)

        while elapsed < max_wait:
            time.sleep(wait)
            elapsed += wait

            resp   = self._client.get_query_execution(QueryExecutionId=execution_id)
            state  = resp["QueryExecution"]["Status"]["State"]

            if state == "SUCCEEDED":
                logger.debug("[athena_client] Query %s succeeded after %.1fs", execution_id, elapsed)
                return

            if state in ("FAILED", "CANCELLED"):
                reason = resp["QueryExecution"]["Status"].get("StateChangeReason", "")
                # Migration note: replaces EXPLAIN USING TEXT error handling —
                # Athena raises InvalidRequestException for unsupported syntax.
                raise RuntimeError(
                    f"Athena query {execution_id} {state}: {reason}"
                )

            # Exponential backoff, cap at 30 s per interval
            wait = min(wait * 1.5, 30.0)

        raise TimeoutError(
            f"Athena query {execution_id} did not complete within {max_wait}s"
        )

    def _fetch_results(self, execution_id: str) -> pd.DataFrame:
        """
        Stream query results from S3 as CSV and return a DataFrame.
        Streaming avoids the 1000-row pagination limit of get_query_results().
        """
        resp = self._client.get_query_execution(QueryExecutionId=execution_id)
        s3_path = resp["QueryExecution"]["ResultConfiguration"]["OutputLocation"]

        # s3://bucket/prefix/execution_id.csv
        s3_path = s3_path.lstrip("s3://")
        bucket, _, key = s3_path.partition("/")

        try:
            obj = self._s3.get_object(Bucket=bucket, Key=key)
            df  = pd.read_csv(io.BytesIO(obj["Body"].read()))
            logger.debug("[athena_client] Fetched %d rows for %s", len(df), execution_id)
            return df
        except Exception as exc:
            logger.warning("[athena_client] Could not read S3 result for %s: %s", execution_id, exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Module-level singleton
# Callers import athena_query() rather than instantiating AthenaClient directly.
# ---------------------------------------------------------------------------

_client: AthenaClient | None = None


def _get_client() -> AthenaClient:
    global _client
    if _client is None:
        _client = AthenaClient()
    return _client


def athena_query(sql: str) -> pd.DataFrame:
    """
    Execute *sql* and return a DataFrame.

    Migration note: top-level replacement for session.sql(sql).to_pandas()
    throughout the codebase.
    """
    return _get_client().run_query(sql)


def athena_collect(sql: str) -> list:
    """
    Execute *sql* and return rows as a list of dicts.

    Migration note: replacement for session.sql(sql).collect()
    throughout the codebase.
    """
    return _get_client().run_query_collect(sql)


def get_glue_column_comments(table_name: str) -> dict:
    """Module-level alias for AthenaClient.get_glue_column_comments()."""
    return _get_client().get_glue_column_comments(table_name)


def list_views(prefix: str = "vw_") -> list:
    """Module-level alias for AthenaClient.list_views()."""
    return _get_client().list_views(prefix)


def list_columns(view_name: str) -> pd.DataFrame:
    """Module-level alias for AthenaClient.list_columns()."""
    return _get_client().list_columns(view_name)
