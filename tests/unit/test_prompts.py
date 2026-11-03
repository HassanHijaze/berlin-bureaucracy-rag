import pytest

from berlin_rag.generation.prompts import PROMPT, REFUSAL_TEXT, is_refusal


def test_prompt_contains_refusal_text():
    assert REFUSAL_TEXT in PROMPT


@pytest.mark.parametrize(
    "answer, expected_result",
    [
        ("The sources I have do not cover this.", True),
        ("  the sources I have do not cover this\n", True),
        ('"The sources I have do not cover this."', True),
        ("The sources I have do not cover this. Try the Bürgeramt.", True),
        ("Private plans do not cover dental care [tk_studenten_ausland.md].", False),
        ("You must register within 14 days [bmg_en_17.txt].", False),
        ("", False),
    ],
)
def test_is_refusal(answer, expected_result):
    assert is_refusal(answer) is expected_result
