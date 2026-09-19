"""
FastAPI service for the Berlin bureaucracy RAG.

Run:
    uvicorn api:app --reload --port 8000

Open:
    http://127.0.0.1:8000
    http://127.0.0.1:8000/docs
    http://127.0.0.1:8000/health

The corpus is built once at startup, not per request.
"""

import re
import time
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import System, K


# ================================================================
# CONFIG
# ================================================================

MAX_QUESTION_CHARS = 500
DAILY_REQUEST_LIMIT = 200


# ================================================================
# CITATION HELPERS
# ================================================================

CITE_RE = re.compile(
    r"\s*\[[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt)"
    r"(?:\s*[;,]\s*[A-Za-z0-9_ äöüÄÖÜß.-]+?\.(?:md|txt))*\]",
    re.IGNORECASE,
)


PAGE_NAMES = {
    "anmeldung_hauptwohnung.md":
        "Berlin.de — Anmeldung einer Wohnung",

    "anmeldung_hauptwohnung_leicht.md":
        "Berlin.de — Anmeldung (Leichte Sprache)",

    "anmeldung_nebenwohnung.md":
        "Berlin.de — Anmeldung einer Nebenwohnung",

    "lea_studium.md":
        "LEA — Aufenthaltserlaubnis zum Studium",

    "lea_aufenthaltserlaubnis_studium.md":
        "LEA — Studium (Merkblatt)",

    "lea_krankenversicherung.md":
        "LEA — Krankenversicherung",

    "lea_termine.md":
        "LEA — Termine",

    "lea_erloeschen.md":
        "LEA — Erlöschen des Aufenthaltstitels",

    "lea_visum_national.md":
        "LEA — Einreise mit nationalem Visum",

    "lea_kurzaufenthalt_90tage.md":
        "LEA — Kurzaufenthalt bis 90 Tage",

    "visum_studium.md":
        "Auswärtiges Amt — Visum zum Studium",

    "gkv_versicherte.md":
        "Bundesgesundheitsministerium — GKV",

    "tk_studenten_ausland.md":
        "TK — Studierende aus dem Ausland",

    "krankenkasse_studium.md":
        "TK — Krankenversicherung für Studierende",
}


LAWS = {
    "bmg": "BMG",
    "aufenthg": "AufenthG",
    "aufenthv": "AufenthV",
    "sgb": "SGB V",
}


def readable(source: str) -> str:
    """Turn a filename into a human-readable source label."""

    if source in PAGE_NAMES:
        return PAGE_NAMES[source]

    stem = (
        source
        .replace(".txt", "")
        .replace(".md", "")
    )

    parts = stem.split("_")

    # Example:
    # bmg_de_17 -> BMG § 17
    # aufenthg_en_16b -> AufenthG § 16b

    if parts[0] in LAWS and len(parts) >= 3:
        return (
            f"{LAWS[parts[0]]} § {parts[2]}"
        )

    # Example:
    # vab_a16b -> VAB A 16b
    if parts[0] == "vab" and len(parts) >= 2:
        return (
            f"VAB A {parts[1].lstrip('a')} "
            f"— LEA-Verfahrenshinweise"
        )

    return source


def cited_files(text: str):
    """Extract cited filenames from [file.txt] markers."""

    citations = re.findall(
        r"\[([^\]]+)\]",
        text,
    )

    files = set()

    for citation in citations:
        for item in re.split(
            r"[;,]",
            citation,
        ):
            item = item.strip()

            if re.search(
                r"\.(?:txt|md)$",
                item,
                re.IGNORECASE,
            ):
                files.add(item)

    return files


def strip_citations(text: str) -> str:
    """
    Remove inline [filename] markers because sources are shown
    separately in the frontend.
    """

    text = CITE_RE.sub("", text)

    # Remove spaces before punctuation.
    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text,
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ================================================================
# APP
# ================================================================

app = FastAPI(
    title="Berlin bureaucracy RAG",
    description=(
        "Answers questions about German registration and residence law "
        "for international students in Berlin, from cited sources. "
        "Declines when the sources do not cover the question. "
        "Not legal advice."
    ),
    version="1.0",
)


# ================================================================
# SYSTEM
# ================================================================

# Built ONCE when the FastAPI process starts.
system = System()

state = {
    "count": 0,
    "started": time.time(),
}


# ================================================================
# MODELS
# ================================================================

class Question(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
        examples=[
            "How long do I have to register my address?"
        ],
    )


class Source(BaseModel):
    file: str
    label: str


class Answer(BaseModel):
    answer: str
    refused: bool
    sources: List[Source]
    elapsed_seconds: float


# ================================================================
# ENDPOINT: ASK
# ================================================================

@app.post(
    "/ask",
    response_model=Answer,
)
def ask(q: Question) -> Answer:
    """Answer one question from the corpus."""

    question = q.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="The question is empty.",
        )

    if state["count"] >= DAILY_REQUEST_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Request limit reached for this instance.",
        )

    state["count"] += 1

    started = time.time()

    try:
        text, hits = system.ask(question)

    except Exception as exc:
        # IMPORTANT:
        # Print the complete underlying error to the terminal.
        # This makes BadRequestError debugging much easier.
        print()
        print("=" * 70)
        print("MODEL ERROR")
        print("=" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print("=" * 70)
        print()

        raise HTTPException(
            status_code=502,
            detail=(
                f"The model call failed: "
                f"{type(exc).__name__}"
            ),
        )

    if not text or not text.strip():
        text = (
            "The sources I have do not cover this."
        )

    refused = (
        "do not cover"
        in text.lower()
    )

    cited_sources = cited_files(text)

    files = sorted(cited_sources)

    return Answer(
        answer=strip_citations(text),
        refused=refused,
        sources=(
            []
            if refused
            else [
                Source(
                    file=f,
                    label=readable(f),
                )
                for f in files
            ]
        ),
        elapsed_seconds=round(
            time.time() - started,
            2,
        ),
    )


# ================================================================
# ENDPOINT: HEALTH
# ================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "documents": len(system.documents),
        "chunks": len(system.chunks),
        "k": K,
        "requests_served": state["count"],
        "uptime_seconds": round(
            time.time() - state["started"]
        ),
    }


# ================================================================
# ENDPOINT: HOME
# ================================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return PAGE


# ================================================================
# FRONTEND
# ================================================================

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Berlin Bureaucracy RAG</title>
<style>
:root{
  --bg:#f6f8fb;--panel:#fff;--ink:#12213a;--body:#445169;--muted:#7b8798;
  --line:#e1e7ef;--accent:#1f63d7;--accent-dark:#174fae;--good:#1d7a50;
  --good-soft:#edf8f2;--danger:#a13c3c;--danger-soft:#fff2f2;
  --shadow:0 18px 45px rgba(26,45,77,.10)
}
*{box-sizing:border-box}
body{
  margin:0;min-height:100vh;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
  color:var(--body);
  background:radial-gradient(circle at 8% 0%,#eaf2ff 0,transparent 30%),
             radial-gradient(circle at 92% 8%,#eef9ff 0,transparent 28%),var(--bg);
  -webkit-font-smoothing:antialiased
}
.shell{width:min(980px,calc(100% - 32px));margin:0 auto;padding:28px 0 58px}
.topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:48px}
.brand{display:flex;align-items:center;gap:12px}
.brand-mark{width:40px;height:40px;display:grid;place-items:center;border-radius:12px;color:#fff;background:linear-gradient(135deg,#1f63d7,#3c8cff);box-shadow:0 9px 24px rgba(31,99,215,.25);font-weight:800}
.brand-title{color:var(--ink);font-size:14px;font-weight:800}
.brand-subtitle{margin-top:3px;color:var(--muted);font-size:11.5px}
.github-link{padding:9px 13px;border:1px solid var(--line);border-radius:10px;background:rgba(255,255,255,.76);color:var(--body);text-decoration:none;font-size:13px;font-weight:700;transition:.18s ease}
.github-link:hover{transform:translateY(-1px);background:#fff;color:var(--ink)}
.hero{max-width:790px;margin:0 auto;text-align:center}
.badge{display:inline-flex;align-items:center;gap:7px;padding:7px 11px;margin-bottom:17px;border-radius:999px;background:var(--good-soft);color:var(--good);font-size:12px;font-weight:800}
.badge-dot{width:7px;height:7px;border-radius:50%;background:var(--good)}
h1{margin:0;color:var(--ink);font-size:clamp(34px,6vw,54px);line-height:1.04;letter-spacing:-.038em;font-weight:850}
.hero p{max-width:700px;margin:17px auto 0;color:var(--muted);font-size:16px;line-height:1.7}
.search-card{max-width:840px;margin:30px auto 0;padding:18px;border:1px solid rgba(214,223,235,.95);border-radius:20px;background:rgba(255,255,255,.91);box-shadow:var(--shadow);backdrop-filter:blur(12px)}
form{display:flex;gap:10px}
input{min-width:0;flex:1;height:56px;padding:0 17px;border:1px solid #d6dee9;border-radius:12px;background:#fff;color:var(--ink);font:inherit;font-size:15.5px;transition:border-color .15s,box-shadow .15s}
input::placeholder{color:#a4aebb}
input:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 4px rgba(31,99,215,.10)}
#btn{height:56px;padding:0 25px;border:0;border-radius:12px;background:linear-gradient(135deg,var(--accent),#347ce8);color:#fff;font:inherit;font-size:14.5px;font-weight:800;cursor:pointer;box-shadow:0 8px 18px rgba(31,99,215,.20);transition:.18s ease}
#btn:hover:not(:disabled){transform:translateY(-1px);background:linear-gradient(135deg,var(--accent-dark),var(--accent))}
#btn:disabled{opacity:.55;cursor:default;box-shadow:none}
.examples-label{margin:16px 2px 9px;color:var(--muted);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.08em}
.examples{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.example-btn{appearance:none;min-height:54px;padding:10px 12px;border:1px solid var(--line);border-radius:10px;background:#fff;color:var(--body);text-align:left;font:inherit;font-size:12.5px;line-height:1.4;cursor:pointer;transition:.16s ease}
.example-btn:hover{border-color:#b8cae6;background:#f8fbff;color:var(--accent-dark)}
.meta{max-width:840px;margin:14px auto 0;display:flex;justify-content:center;gap:18px;flex-wrap:wrap;color:var(--muted);font-size:12px}
.meta strong{color:var(--body)}
#out{max-width:840px;margin:27px auto 0}
#out:empty{display:none}
.card{padding:26px;border:1px solid var(--line);border-radius:18px;background:var(--panel);box-shadow:var(--shadow)}
.card-header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px}
.card-title{color:var(--ink);font-size:12px;font-weight:850;text-transform:uppercase;letter-spacing:.08em}
.status-pill{padding:6px 9px;border-radius:999px;background:var(--good-soft);color:var(--good);font-size:11px;font-weight:800}
.status-pill.refusal{background:#f1f3f6;color:#6e7683}
.answer{color:var(--body);font-size:15.5px;line-height:1.75}
.answer p{margin:0 0 14px}.answer p:last-child{margin-bottom:0}
.answer b,.answer strong{color:var(--ink);font-weight:750}
.answer ul{margin:0 0 14px;padding-left:21px}.answer li{margin-bottom:7px}
.refused{color:#6d7581;font-style:italic}
.sources{margin-top:22px;padding-top:18px;border-top:1px solid var(--line)}
.sources-label{display:block;margin-bottom:10px;color:var(--muted);font-size:11px;font-weight:850;text-transform:uppercase;letter-spacing:.08em}
.source-grid{display:flex;flex-wrap:wrap;gap:8px}
.source-chip{display:inline-flex;align-items:center;gap:7px;padding:7px 10px;border:1px solid #d9e3ef;border-radius:9px;background:#f8fbff;color:#35516f;font-size:12px;font-weight:700}
.source-dot{width:6px;height:6px;border-radius:50%;background:var(--accent)}
.timing{margin-top:10px;text-align:right;color:#9ca6b5;font-size:11.5px}
.thinking{padding:18px 20px;border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.84);color:var(--muted);font-size:14px;box-shadow:0 10px 28px rgba(26,45,77,.06)}
.thinking::after{content:'';display:inline-block;width:1.2em;animation:dots 1.15s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
.error{padding:16px 18px;border:1px solid #f0cccc;border-radius:12px;background:var(--danger-soft);color:var(--danger);font-size:14px}
.footer{max-width:840px;margin:30px auto 0;padding-top:18px;border-top:1px solid rgba(214,222,232,.85);color:var(--muted);text-align:center;font-size:11.5px;line-height:1.6}
@media(max-width:720px){.shell{width:min(100% - 22px,980px);padding-top:18px}.topbar{margin-bottom:34px}.brand-subtitle{display:none}form{flex-direction:column}#btn{width:100%}.examples{grid-template-columns:1fr}.card{padding:20px}}
</style>
</head>
<body>
<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">B</div>
      <div>
        <div class="brand-title">Berlin Bureaucracy RAG</div>
        <div class="brand-subtitle">Source-grounded answers for international students</div>
      </div>
    </div>
    <a class="github-link" href="https://github.com/HassanHijaze/berlin-bureaucracy-rag" target="_blank" rel="noopener">GitHub</a>
  </header>

  <main>
    <section class="hero">
      <div class="badge"><span class="badge-dot"></span>Grounded in official sources</div>
      <h1>Berlin bureaucracy,<br>made easier to navigate.</h1>
      <p>
        Ask questions about address registration, student residence permits,
        student working rights and health insurance. Answers are generated only
        from the project's curated legal and administrative corpus.
      </p>
    </section>

    <section class="search-card">
      <form onsubmit="go(event)">
        <input id="q" maxlength="500" placeholder="Ask a question about living or studying in Berlin..." autofocus>
        <button id="btn" type="submit">Ask</button>
      </form>

      <div class="examples-label">Try an example</div>
      <div class="examples">
        <button class="example-btn" type="button" onclick="fill(this)">What happens if I register late?</button>
        <button class="example-btn" type="button" onclick="fill(this)">How many days can I work as a student?</button>
        <button class="example-btn" type="button" onclick="fill(this)">My permit expires before my appointment</button>
      </div>
    </section>

    <div class="meta">
      <span><strong>Retrieval:</strong> Semantic + BM25</span>
      <span><strong>Top-K:</strong> 12</span>
      <span><strong>Scope:</strong> Berlin / international students</span>
    </div>

    <div id="out"></div>

    <footer class="footer">
      Not legal advice. The system declines questions that are not supported by its sources.
      Verify important decisions against the cited law or official administrative guidance.
    </footer>
  </main>
</div>

<script>
const out = document.getElementById('out');
const btn = document.getElementById('btn');
const box = document.getElementById('q');

function fill(el){box.value=el.textContent.trim();go();}

async function go(event){
  if(event){event.preventDefault();}
  const q=box.value.trim();
  if(!q){return;}

  btn.disabled=true;
  btn.textContent='Searching...';
  out.innerHTML='<div class="thinking">Searching the legal and administrative sources</div>';

  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});

    if(!r.ok){
      const err=await r.json().catch(()=>({}));
      out.innerHTML='<div class="error">'+esc(err.detail||('Request failed ('+r.status+')'))+'</div>';
      btn.disabled=false;
      btn.textContent='Ask';
      return;
    }

    const d=await r.json();
    let html='<div class="card">';

    html+='<div class="card-header">'+
      '<div class="card-title">Answer</div>'+
      '<div class="status-pill'+(d.refused?' refusal':'')+'">'+
      (d.refused?'Not covered':'Source-grounded')+'</div></div>';

    html+='<div class="answer'+(d.refused?' refused':'')+'">'+format(d.answer)+'</div>';

    if(d.sources.length){
      html+='<div class="sources"><span class="sources-label">Sources used</span><div class="source-grid">'+
        d.sources.map(s=>'<div class="source-chip"><span class="source-dot"></span>'+esc(s.label)+'</div>').join('')+
        '</div></div>';
    }

    html+='</div>';
    html+='<div class="timing">Answered in '+d.elapsed_seconds+' seconds</div>';
    out.innerHTML=html;
  }catch(e){
    out.innerHTML='<div class="error">Could not reach the service.</div>';
  }

  btn.disabled=false;
  btn.textContent='Ask';
}

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function format(text){
  const blocks=esc(text).split(/\\n\\s*\\n/);
  return blocks.map(block=>{
    const lines=block.split('\\n').map(l=>l.trim()).filter(Boolean);
    const bulleted=lines.length&&lines.every(l=>/^[-*•]\\s+|^\\d+\\.\\s+/.test(l));
    if(bulleted){
      return '<ul>'+lines.map(l=>'<li>'+bold(l.replace(/^[-*•]\\s+|^\\d+\\.\\s+/,''))+'</li>').join('')+'</ul>';
    }
    return '<p>'+bold(lines.join(' '))+'</p>';
  }).join('');
}

function bold(s){
  return s.replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>').replace(/__(.+?)__/g,'<b>$1</b>');
}
</script>
</body>
</html>
"""
