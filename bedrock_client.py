"""
bedrock_client.py — all Amazon Bedrock AI calls and S3 YAML file operations.

Migration note: replaces every
  session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?,?) AS RESPONSE", params=[model, prompt])
call site (16 total) and every
  session.file.get_stream / session.file.put_stream  (@DEALER.BUSINESS_VAULT.DEALER_STAGE)
call site throughout DealerFinalVersion.py.
"""
import json
import logging

import boto3
import yaml

from config_loader import get_aws_session, get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model families — each requires a different request/response payload shape
# ---------------------------------------------------------------------------

def _is_claude(model_id: str) -> bool:
    return "anthropic." in model_id

def _is_llama(model_id: str) -> bool:
    return "meta." in model_id

def _is_mistral(model_id: str) -> bool:
    return "mistral." in model_id


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

def _build_body(model_id: str, prompt: str, max_tokens: int, temperature: float) -> bytes:
    """
    Build the JSON request body for invoke_model.

    Migration note: replaces params=[MODEL, PROMPT] in every
    SNOWFLAKE.CORTEX.COMPLETE(?,?) call. Each model family expects a
    different payload schema.
    """
    if _is_claude(model_id):
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    elif _is_llama(model_id):
        payload = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
        }
    elif _is_mistral(model_id):
        payload = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        # Generic fallback — attempt Claude-style body
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
    return json.dumps(payload).encode("utf-8")


def _parse_response(model_id: str, raw: dict) -> str:
    """
    Extract the text response from invoke_model's response body.

    Migration note: replaces tdf.at[0,"RESPONSE"] / df.at[0,"R"] /
    df.at[0,"SQL_QUERY"] everywhere in the codebase.
    """
    body = json.loads(raw["body"].read())

    if _is_claude(model_id):
        # {"content": [{"type": "text", "text": "..."}], ...}
        contents = body.get("content", [])
        parts = [c.get("text", "") for c in contents if c.get("type") == "text"]
        return "\n".join(parts).strip()

    if _is_llama(model_id):
        # {"generation": "...", ...}
        return (body.get("generation") or "").strip()

    if _is_mistral(model_id):
        # {"outputs": [{"text": "..."}]}
        outputs = body.get("outputs", [])
        return (outputs[0].get("text", "") if outputs else "").strip()

    # Fallback: try Claude shape
    contents = body.get("content", [])
    parts = [c.get("text", "") for c in contents if c.get("type") == "text"]
    return "\n".join(parts).strip()


# ---------------------------------------------------------------------------
# Core Bedrock client
# ---------------------------------------------------------------------------

class BedrockClient:
    """
    Wraps boto3 bedrock-runtime for all Cortex → Bedrock AI calls.

    Migration note: replaces get_snowflake_connection() + session.sql(CORTEX.COMPLETE)
    pattern used at 16 call sites throughout DealerFinalVersion.py.
    """

    def __init__(self):
        cfg = get_config()
        self._primary_model = cfg["bedrock"]["primary_model"]    # Migration note: was CORTEX_PRESCRIPTIVE_MODEL (primary)
        self._model_llama8b  = cfg["bedrock"]["model_llama8b"]   # Migration note: was "llama3-8b"
        self._model_llama70b = cfg["bedrock"]["model_llama70b"]  # Migration note: was "llama3.1-70b"
        self._model_mistral7b = cfg["bedrock"]["model_mistral7b"]# Migration note: was "mistral-7b"
        self._max_tokens  = cfg["bedrock"]["max_tokens"]
        self._temperature = cfg["bedrock"]["temperature"]

        # S3 config for YAML stage operations
        self._s3_bucket       = cfg["s3"]["bucket"]
        self._config_prefix   = cfg["s3"]["config_prefix"]
        self._yaml_filename   = cfg["s3"]["yaml_filename"]

        # Migration note: aws_session replaces snowflake_session
        session: boto3.Session = get_aws_session()
        self._runtime = session.client("bedrock-runtime")
        self._s3      = session.client("s3")

    # ------------------------------------------------------------------
    # Public AI interface
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        model_id: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """
        Call Bedrock and return the text response.

        Migration note: replaces all 16 call sites of the form:
          session.sql("SELECT SNOWFLAKE.CORTEX.COMPLETE(?,?) AS RESPONSE",
                      params=[model, prompt]).to_pandas()
          tdf.at[0,"RESPONSE"]

        model_id defaults to bedrock.primary_model from config.
        """
        model_id    = model_id    or self._primary_model
        max_tokens  = max_tokens  or self._max_tokens
        temperature = temperature if temperature is not None else self._temperature

        body = _build_body(model_id, prompt, max_tokens, temperature)
        try:
            raw = self._runtime.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            text = _parse_response(model_id, raw)
            logger.debug("[bedrock_client] complete(%s) → %d chars", model_id, len(text))
            return text
        except Exception as exc:
            logger.warning("[bedrock_client] invoke_model failed (%s): %s", model_id, exc)
            return ""

    # Convenience aliases matching source model references -----------------

    def complete_llama8b(self, prompt: str) -> str:
        """Migration note: replaces CORTEX.COMPLETE with 'llama3-8b' (lines 5289, 13908)."""
        return self.complete(prompt, model_id=self._model_llama8b)

    def complete_llama70b(self, prompt: str) -> str:
        """Migration note: replaces CORTEX.COMPLETE with 'llama3.1-70b' (line 7062)."""
        return self.complete(prompt, model_id=self._model_llama70b)

    def complete_mistral7b(self, prompt: str) -> str:
        """Migration note: replaces CORTEX.COMPLETE with 'mistral-7b' (line 8940)."""
        return self.complete(prompt, model_id=self._model_mistral7b)

    # ------------------------------------------------------------------
    # S3 YAML stage operations
    # Migration note: replaces session.file.get_stream / put_stream on
    # @DEALER.BUSINESS_VAULT.DEALER_STAGE (lines 425, 851, 957, 3617)
    # ------------------------------------------------------------------

    def load_yaml_from_s3(self, filename: str | None = None) -> dict:
        """
        Load a YAML file from the S3 config prefix.

        Migration note: replaces
          session.file.get_stream("@DEALER.BUSINESS_VAULT.DEALER_STAGE/<file>")
        """
        key = self._config_prefix + (filename or self._yaml_filename)
        try:
            obj = self._s3.get_object(Bucket=self._s3_bucket, Key=key)
            content = obj["Body"].read().decode("utf-8")
            logger.debug("[bedrock_client] Loaded YAML from s3://%s/%s", self._s3_bucket, key)
            return yaml.safe_load(content) or {}
        except Exception as exc:
            logger.warning("[bedrock_client] Could not load s3://%s/%s: %s", self._s3_bucket, key, exc)
            return {}

    def save_yaml_to_s3(self, data: dict, filename: str | None = None) -> bool:
        """
        Save a dict as YAML to the S3 config prefix.

        Migration note: replaces
          session.file.put_stream(stream, "@DEALER.BUSINESS_VAULT.DEALER_STAGE/<file>")
        """
        key = self._config_prefix + (filename or self._yaml_filename)
        try:
            body = yaml.dump(data, default_flow_style=False, allow_unicode=True)
            self._s3.put_object(
                Bucket=self._s3_bucket,
                Key=key,
                Body=body.encode("utf-8"),
                ContentType="application/x-yaml",
            )
            logger.debug("[bedrock_client] Saved YAML to s3://%s/%s", self._s3_bucket, key)
            return True
        except Exception as exc:
            logger.warning("[bedrock_client] Could not save s3://%s/%s: %s", self._s3_bucket, key, exc)
            return False


# ---------------------------------------------------------------------------
# Module-level singleton + convenience functions
# Callers use bedrock_complete() rather than instantiating BedrockClient.
# ---------------------------------------------------------------------------

_client: BedrockClient | None = None


def _get_client() -> BedrockClient:
    global _client
    if _client is None:
        _client = BedrockClient()
    return _client


def bedrock_complete(
    prompt: str,
    model_id: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Call Bedrock and return the text response.

    Migration note: primary drop-in replacement for all 16
    SNOWFLAKE.CORTEX.COMPLETE call sites.
    """
    return _get_client().complete(prompt, model_id=model_id, max_tokens=max_tokens, temperature=temperature)


def bedrock_complete_llama8b(prompt: str) -> str:
    """Migration note: replaces CORTEX.COMPLETE with 'llama3-8b' (lines 5289, 13908)."""
    return _get_client().complete_llama8b(prompt)


def bedrock_complete_llama70b(prompt: str) -> str:
    """Migration note: replaces CORTEX.COMPLETE with 'llama3.1-70b' (line 7062)."""
    return _get_client().complete_llama70b(prompt)


def bedrock_complete_mistral7b(prompt: str) -> str:
    """Migration note: replaces CORTEX.COMPLETE with 'mistral-7b' (line 8940)."""
    return _get_client().complete_mistral7b(prompt)


def load_yaml_from_s3(filename: str | None = None) -> dict:
    """
    Load YAML config from S3.

    Migration note: replaces session.file.get_stream(@DEALER.BUSINESS_VAULT.DEALER_STAGE/...)
    """
    return _get_client().load_yaml_from_s3(filename)


def save_yaml_to_s3(data: dict, filename: str | None = None) -> bool:
    """
    Save YAML config to S3.

    Migration note: replaces session.file.put_stream(stream, @DEALER.BUSINESS_VAULT.DEALER_STAGE/...)
    """
    return _get_client().save_yaml_to_s3(data, filename)
