"""Settings, paths, and tunable constants for the whole package, in one place.

Tunable values (models, retrieval weights, limits) are plain constants.
Values that differ per machine or deployment, such as the OpenAI API key,
come from environment variables or the git-ignored API.env file through
``Settings``.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ================================================================
# PATHS
# ================================================================

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
EVAL_FILE = DATA_DIR / "eval" / "100.json"
CACHE_DIR = DATA_DIR / "cache"

ENV_FILE = PROJECT_ROOT / "API.env"
INDEX_HTML = PACKAGE_DIR / "static" / "index.html"

LEGAL_DIRECTORIES = [
    "bmg_sections",
    "aufenthg_sections",
    "sgb_sections",
    "aufenthv_sections",
    "vab_sections",
]

# ================================================================
# MODELS
# ================================================================

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.6"

MAX_COMPLETION_TOKENS_EXPANSION = 200
MAX_COMPLETION_TOKENS_ANSWER = 800

# ================================================================
# CHUNKING
# ================================================================

MAX_CHUNK = 750
MIN_CHUNK = 150
CHUNK_OVERLAP = 100

# ================================================================
# RETRIEVAL
# ================================================================

TOP_K = 12
MAX_PER_SOURCE = 2

SEMANTIC_WEIGHT = 0.70
BM25_WEIGHT = 0.30
BM25_ORIGINAL_WEIGHT = 0.60
BM25_EXPANDED_WEIGHT = 0.40

BM25_K1 = 1.5
BM25_B = 0.75

EMBED_BATCH_SIZE = 100
FINGERPRINT_LENGTH = 16

# ================================================================
# API
# ================================================================

MAX_QUESTION_CHARS = 500
DAILY_REQUEST_LIMIT = 200


class Settings(BaseSettings):
    """Deployment settings read from environment variables or API.env.

    Real environment variables take precedence over API.env, so Docker
    (``--env-file``) and Render (dashboard variables) work unchanged.

    Attributes:
        openai_api_key: The OpenAI API key, or None when not configured.
            Stored as a SecretStr so it never appears in logs or reprs.
    """

    model_config = SettingsConfigDict(
        env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Load the settings once and reuse them.

    Returns:
        The process-wide Settings instance.
    """
    return Settings()


# ================================================================
# LOGGING
# ================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
