import re

import pytest

from berlin_rag.citations import cited_files
from berlin_rag.config import MAX_PER_SOURCE, TOP_K
from berlin_rag.generation.answer import load_answer_generator
from berlin_rag.generation.prompts import is_refusal

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def answer_generator():
    return load_answer_generator()


def test_search(answer_generator):
    hits = answer_generator.retriever.search(
        "How many days do I have to register my address?"
    )
    sources = [chunk["source"] for chunk, *_ in hits]
    scores = [final for _, final, _, _ in hits]

    assert len(hits) == TOP_K
    assert max(sources.count(source) for source in sources) <= MAX_PER_SOURCE
    assert scores == sorted(scores, reverse=True)
    assert any("bmg" in source and "_17" in source for source in sources)


def test_ask_answers_a_covered_question(answer_generator):
    text, _ = answer_generator.ask(
        "How long do I have to register my address in Berlin?"
    )

    assert not is_refusal(text)
    assert re.search(r"\b(14|two weeks|zwei Wochen)\b", text, re.I)
    assert cited_files(text)


def test_ask_refuses_an_uncovered_question(answer_generator):
    text, _ = answer_generator.ask("How much is the Rundfunkbeitrag for students?")
    assert is_refusal(text)
