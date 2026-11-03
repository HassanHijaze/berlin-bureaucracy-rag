"""Source-aware chunking for German statutes and Berlin administrative pages."""

import re

from ..config import CHUNK_OVERLAP, MAX_CHUNK, MIN_CHUNK

BERLIN_HEADINGS = [
    "Verfahrensablauf",
    "Voraussetzungen",
    "Erforderliche Unterlagen",
    "Formulare",
    "Gebühren",
    "Rechtsgrundlagen",
    "Durchschnittliche Bearbeitungszeit",
    "Weiterführende Informationen",
]


def split_hard(
    text: str, max_size: int = MAX_CHUNK, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into fixed-size pieces that overlap by ``overlap`` characters.

    Args:
        text: The text to split.
        max_size: Maximum length of each piece, in characters.
        overlap: Number of characters shared by consecutive pieces.

    Returns:
        The pieces in order. A text no longer than ``max_size`` is returned whole.
    """
    if len(text) <= max_size:
        return [text]

    pieces = []
    step = max_size - overlap
    start = 0

    while start < len(text):
        pieces.append(text[start : start + max_size])
        start += step

    return pieces


def chunk_legal(text: str, max_size: int = MAX_CHUNK) -> list[str]:
    """Split a statute at paragraph markers like (1), (2), keeping the header.

    Every piece after the first starts with the statute's first line in
    brackets, so a chunk still says which section it belongs to.

    Args:
        text: The statute section, with its title on the first line.
        max_size: Maximum length of each chunk, in characters.

    Returns:
        The chunks in order. Paragraphs longer than ``max_size`` are split hard.
    """
    if len(text) <= max_size:
        return [text]

    lines = text.split("\n")
    header = lines[0] if lines else ""

    pieces = []
    current = ""

    for part in re.split(r"\n(?=\(\d+\))", text):
        if current and len(current) + len(part) > max_size:
            pieces.append(f"[{header}]\n{current}" if pieces else current)
            current = part
        else:
            current += ("\n" if current else "") + part

    if current:
        pieces.append(f"[{header}]\n{current}" if pieces else current)

    final_pieces = []
    for piece in pieces:
        if len(piece) <= max_size:
            final_pieces.append(piece)
        else:
            final_pieces.extend(split_hard(piece, max_size=max_size))

    return final_pieces


def chunk_berlin(text: str, max_size: int = MAX_CHUNK) -> list[str]:
    """Split a Berlin administrative page at its standard section headings.

    Args:
        text: The page content.
        max_size: Maximum length of each chunk, in characters.

    Returns:
        One chunk per section, in order. Sections longer than ``max_size``
        are split hard.
    """
    pattern = "|".join(re.escape(heading) for heading in BERLIN_HEADINGS)

    parts = [
        part.strip() for part in re.split(rf"\n(?={pattern})", text) if part.strip()
    ]
    if not parts:
        parts = [text]

    output = []
    for part in parts:
        if len(part) <= max_size:
            output.append(part)
        else:
            output.extend(split_hard(part, max_size=max_size))

    return output


def merge_small(chunks: list[dict], min_size: int = MIN_CHUNK) -> list[dict]:
    """Merge tiny chunks into the previous chunk from the same source.

    A chunk like "Gebühren / keine" (a heading and one word) is too short to
    be found by search on its own, so it is appended to its neighbour when
    the result still fits.

    Args:
        chunks: Chunks with ``text``, ``source`` and ``index`` fields.
        min_size: Chunks shorter than this are merged when possible.

    Returns:
        New chunk dicts with indices renumbered per source. The input is
        not modified.
    """
    output = []

    for chunk in chunks:
        can_merge = (
            output
            and len(chunk["text"]) < min_size
            and output[-1]["source"] == chunk["source"]
            and len(output[-1]["text"]) + 1 + len(chunk["text"]) <= MAX_CHUNK
        )
        if can_merge:
            output[-1]["text"] += "\n" + chunk["text"]
        else:
            output.append(dict(chunk))

    # Renumber chunk indices per source after merging.
    counts = {}
    for chunk in output:
        index = counts.get(chunk["source"], 0)
        chunk["index"] = index
        counts[chunk["source"]] = index + 1

    return output
