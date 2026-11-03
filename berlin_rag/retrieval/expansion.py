"""Query expansion: rewrite a question into German legal keywords for BM25.

Students often ask in English or in everyday words, while the statutes use
German legal terms. The expansion lets keyword search find those terms.
"""

import logging

from ..config import CHAT_MODEL, MAX_COMPLETION_TOKENS_EXPANSION
from ..openai_client import get_client

logger = logging.getLogger(__name__)

NO_EXPANSION = ""

EXPANSION_PROMPT = (
    "Map the following user question to German administrative and legal "
    "search keywords.\n\n"
    "Include relevant German legal terms, statutory sections, and "
    "administrative terminology when appropriate.\n\n"
    "Examples:\n"
    "- cost/fee -> Gebühren kostenfrei\n"
    "- lost paper/proof -> Meldebescheinigung Meldebestätigung\n"
    "- fine/penalty -> Ordnungswidrigkeit Bußgeld § 54\n"
    "- hotel/hostel -> Beherbergungsstätte § 29\n"
    "- move/change address -> Anmeldung Ummeldung § 17 § 21\n\n"
    "Question: {question}\n\n"
    "Output ONLY space-separated German keywords."
)


def expand_query(question: str) -> str:
    """Rewrite a question into space-separated German legal keywords.

    Expansion is optional: if the model call fails for any reason, search
    continues with the original question only.

    Args:
        question: The user's question, in English or German.

    Returns:
        The German keywords, or an empty string if the expansion failed.
    """
    try:
        response = get_client().chat.completions.create(
            model=CHAT_MODEL,
            max_completion_tokens=MAX_COMPLETION_TOKENS_EXPANSION,
            messages=[
                {
                    "role": "user",
                    "content": EXPANSION_PROMPT.format(question=question),
                }
            ],
        )
        return response.choices[0].message.content.strip()

    # Expansion is optional: any failure falls back to the original question.
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query expansion failed: %s", exc)
        return NO_EXPANSION
