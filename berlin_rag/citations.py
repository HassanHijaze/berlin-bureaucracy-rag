"""Citation handling for generated answers.

The language model cites its sources inline as bracketed filenames, for
example ``[bmg_de_17.txt]`` or ``[lea_termine.md; aufenthg_en_16b.txt]``.
This module extracts those filenames, converts them into human-readable
labels for the web interface, and removes the raw markers from the answer
text so that sources can be displayed separately.
"""

import re

CITATION_MARKER_RE = re.compile(
    r"\s*\[[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt)"
    r"(?:\s*[;,]\s*[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt))*\]",
    re.IGNORECASE,
)
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
SEPARATOR_RE = re.compile(r"[;,]")
SOURCE_FILE_RE = re.compile(r"\.(?:txt|md)$", re.IGNORECASE)
SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([.,;:!?])")
EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")

PAGE_NAMES: dict[str, str] = {
    "anmeldung_hauptwohnung.md": "Berlin.de — Anmeldung einer Wohnung",
    "anmeldung_hauptwohnung_leicht.md": "Berlin.de — Anmeldung (Leichte Sprache)",
    "anmeldung_nebenwohnung.md": "Berlin.de — Anmeldung einer Nebenwohnung",
    "lea_studium.md": "LEA — Aufenthaltserlaubnis zum Studium",
    "lea_aufenthaltserlaubnis_studium.md": "LEA — Studium (Merkblatt)",
    "lea_krankenversicherung.md": "LEA — Krankenversicherung",
    "lea_termine.md": "LEA — Termine",
    "lea_erloeschen.md": "LEA — Erlöschen des Aufenthaltstitels",
    "lea_visum_national.md": "LEA — Einreise mit nationalem Visum",
    "lea_kurzaufenthalt_90tage.md": "LEA — Kurzaufenthalt bis 90 Tage",
    "visum_studium.md": "Auswärtiges Amt — Visum zum Studium",
    "gkv_versicherte.md": "Bundesgesundheitsministerium — GKV",
    "tk_studenten_ausland.md": "TK — Studierende aus dem Ausland",
    "krankenkasse_studium.md": "TK — Krankenversicherung für Studierende",
}
"""Display names for administrative pages, keyed by filename."""

LAW_ABBREVIATIONS: dict[str, str] = {
    "bmg": "BMG",
    "aufenthg": "AufenthG",
    "aufenthv": "AufenthV",
    "sgb": "SGB V",
}
"""Official abbreviations for statutes, keyed by filename prefix."""


def readable(source: str) -> str:
    """Convert a corpus filename into a human-readable source label.

    Administrative pages are looked up in ``PAGE_NAMES``. Statute files
    follow the pattern ``<law>_<language>_<section>`` and are rendered as
    the law's abbreviation and section number.

    Args:
        source: Filename of a corpus document.

    Returns:
        A display label, or the filename itself if no rule applies.

    Examples:
        >>> readable("bmg_de_17.txt")
        'BMG § 17'
        >>> readable("aufenthg_en_16b.txt")
        'AufenthG § 16b'
        >>> readable("vab_a16b.txt")
        'VAB A 16b — LEA-Verfahrenshinweise'
    """
    if source in PAGE_NAMES:
        return PAGE_NAMES[source]

    parts = source.replace(".txt", "").replace(".md", "").split("_")

    if parts[0] in LAW_ABBREVIATIONS and len(parts) >= 3:
        return f"{LAW_ABBREVIATIONS[parts[0]]} § {parts[2]}"

    if parts[0] == "vab" and len(parts) >= 2:
        return f"VAB A {parts[1].lstrip('a')} — LEA-Verfahrenshinweise"

    return source


def cited_files(text: str) -> set[str]:
    """Extract the filenames cited in an answer.

    Handles single citations such as ``[a.txt]`` and grouped citations
    separated by semicolons or commas, such as ``[a.txt; b.md]``.

    Args:
        text: The generated answer.

    Returns:
        The set of cited ``.txt`` and ``.md`` filenames.
    """
    files: set[str] = set()

    for citation in BRACKET_RE.findall(text):
        for item in SEPARATOR_RE.split(citation):
            item = item.strip()
            if SOURCE_FILE_RE.search(item):
                files.add(item)

    return files


def strip_citations(text: str) -> str:
    """Remove inline citation markers from an answer.

    The web interface lists sources separately, so the bracketed filenames
    are removed and the surrounding whitespace is tidied.

    Args:
        text: The generated answer.

    Returns:
        The answer without citation markers.
    """
    text = CITATION_MARKER_RE.sub("", text)
    text = SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", text)
    text = EXCESS_BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()