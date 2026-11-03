import pytest

from berlin_rag.citations import PAGE_NAMES, cited_files, readable, strip_citations
from berlin_rag.config import CORPUS_DIR


@pytest.fixture
def answer():
    return (
        "You may work 140 days [aufenthg_en_16b.txt]. "
        "Student jobs are excluded [lea_studium.md; bmg_de_17.txt]."
    )


def test_cited_files(answer):
    expected_result = {"aufenthg_en_16b.txt", "lea_studium.md", "bmg_de_17.txt"}
    assert cited_files(answer) == expected_result


@pytest.mark.parametrize(
    "text, expected_result",
    [
        ("Three [a_b.md, c_d.txt, e.txt] here.", {"a_b.md", "c_d.txt", "e.txt"}),
        ("No citation at all.", set()),
    ],
)
def test_cited_files_edge_cases(text, expected_result):
    assert cited_files(text) == expected_result


def test_strip_citations(answer):
    expected_result = "You may work 140 days. Student jobs are excluded."
    assert strip_citations(answer) == expected_result


def test_strip_citations_leaves_plain_text_alone():
    assert strip_citations("Nothing to strip here.") == "Nothing to strip here."


@pytest.mark.parametrize(
    "filename, expected_result",
    [
        ("bmg_de_17.txt", "BMG § 17"),
        ("bmg_en_23a.txt", "BMG § 23a"),
        ("aufenthg_de_16b.txt", "AufenthG § 16b"),
        ("aufenthg_en_16b.txt", "AufenthG § 16b"),
        ("aufenthv_de_47.txt", "AufenthV § 47"),
        ("sgb_de_199a.txt", "SGB V § 199a"),
        ("vab_a16b.txt", "VAB A 16b — LEA-Verfahrenshinweise"),
        ("lea_studium.md", "LEA — Aufenthaltserlaubnis zum Studium"),
        ("unknown_file.md", "unknown_file.md"),
    ],
)
def test_readable(filename, expected_result):
    assert readable(filename) == expected_result


def test_page_names_match_corpus_pages():
    expected_result = {path.name for path in CORPUS_DIR.glob("*.md")}
    assert set(PAGE_NAMES) == expected_result
