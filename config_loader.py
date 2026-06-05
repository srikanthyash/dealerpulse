import copy
import json
import logging
import os

import boto3
import streamlit as st

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Every key that must exist after merging config.json with Secrets Manager.
# Format: "section.key"
_REQUIRED_KEYS = [
    "aws.region",
    "aws.secrets_manager_key",
    "athena.database",
    "athena.workgroup",
    "athena.s3_output",
    "athena.poll_interval_seconds",
    "athena.poll_max_wait_seconds",
    "bedrock.primary_model",     # Migration note: was CORTEX_COMPLETE model name constant
    "bedrock.model_llama8b",
    "bedrock.model_llama70b",
    "bedrock.model_mistral7b",
    "bedrock.max_tokens",
    "bedrock.temperature",
    "s3.bucket",
    "s3.config_prefix",
    "s3.yaml_filename",
    "dynamodb.cache_table",
    "dynamodb.sessions_table",
    "dynamodb.cache_ttl_seconds",
    "dynamodb.session_ttl_days",
]


def _load_file() -> dict:
    with open(_CONFIG_PATH) as fh:
        return json.load(fh)


def _fetch_secrets(base_cfg: dict) -> dict:
    """
    Pull overrides from AWS Secrets Manager.

    The secret is expected to be a JSON object whose top-level keys are
    section names matching config.json (e.g. {"athena": {"database": "..."}}).
    Returns {} when Secrets Manager is unreachable — local config.json is
    used as-is in that case (local dev path).
    """
    secret_name = base_cfg.get("aws", {}).get("secrets_manager_key", "")
    region = base_cfg.get("aws", {}).get("region", "us-east-1")

    if not secret_name:
        return {}

    try:
        # Migration note: aws_session replaces snowflake_session
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response["SecretString"])
    except Exception as exc:
        logging.debug(
            "[config_loader] Secrets Manager unavailable (%s); "
            "falling back to config.json values.",
            exc,
        )
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _validate(cfg: dict) -> None:
    """Raise ValueError listing every missing required key."""
    missing = []
    for dotted in _REQUIRED_KEYS:
        section, _, leaf = dotted.partition(".")
        if leaf not in cfg.get(section, {}):
            missing.append(dotted)
    if missing:
        raise ValueError(
            "DealerPulse config is incomplete. "
            "Add the following keys to config.json or the Secrets Manager secret "
            f"'{cfg.get('aws', {}).get('secrets_manager_key', '<unset>')}': "
            + ", ".join(missing)
        )


@st.cache_resource
def get_config() -> dict:
    """
    Return the fully-merged, validated application config.

    Load order:
      1. config.json  (baseline / local-dev values)
      2. AWS Secrets Manager secret named by aws.secrets_manager_key
         (overrides config.json; silently skipped when unreachable)

    Raises ValueError on startup if any required key is absent.
    """
    base = _load_file()
    secrets = _fetch_secrets(base)
    cfg = _deep_merge(base, secrets)
    _validate(cfg)
    return cfg


@st.cache_resource
def get_aws_session() -> boto3.Session:
    """
    Return a reusable boto3 Session.

    Migration note: replaces get_snowflake_connection() / SNOWFLAKE_AVAILABLE block.
    Credentials come from the boto3 default chain:
      env vars → ~/.aws/credentials → EC2 instance profile / IAM role.
    """
    cfg = get_config()
    region = cfg["aws"]["region"]
    logging.info("[config_loader] Creating boto3 Session (region=%s).", region)
    # Migration note: aws_session replaces snowflake_session
    return boto3.Session(region_name=region)
