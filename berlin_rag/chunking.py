"""Source-aware chunking for German statutes and Berlin administrative pages."""

import re

MAX_CHUNK = 750
MIN_CHUNK = 150

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


def split_hard(text, max_size=MAX_CHUNK, overlap=100):
    """Split text into fixed-size pieces that overlap by `overlap` characters."""
    if len(text) <= max_size:
        return [text]

    pieces = []
    step = max_size - overlap
    start = 0

    while start < len(text):
        pieces.append(text[start:start + max_size])
        start += step

    return pieces


def chunk_legal(text, max_size=MAX_CHUNK):
    """Split a statute at paragraph markers like (1), (2), keeping the header."""
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


def chunk_berlin(text, max_size=MAX_CHUNK):
    """Split a Berlin administrative page at its standard section headings."""
    pattern = "|".join(re.escape(heading) for heading in BERLIN_HEADINGS)

    parts = [
        part.strip()
        for part in re.split(rf"\n(?={pattern})", text)
        if part.strip()
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


def merge_small(chunks, min_size=MIN_CHUNK):
    """Merge tiny chunks into the previous chunk from the same source."""
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