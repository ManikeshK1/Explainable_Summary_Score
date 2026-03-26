"""
local_grader.py — ExplainGrade Local Runner
============================================
Grades student summaries (.xlsx) against a Google Meet transcript (.docx)
using real sentence-transformers for better semantic accuracy than the browser.

Usage:
    python local_grader.py --docx "transcript.docx" --xlsx "summaries.xlsx"
    python local_grader.py --docx "transcript.docx" --xlsx "summaries.xlsx" --max-score 10 --output "results.csv"

XLSX must have columns (flexible name matching):
    emailAddress  (or email / mail)
    name          (or student_name)
    summary       (or answer / response)

Requirements:
    pip install -r requirements.txt
"""

import argparse
import sys
import os
import re
import csv
import math
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# Dependency check helper
# ─────────────────────────────────────────────────────────────────

def _require(pkg, install_name=None):
    try:
        return __import__(pkg)
    except ImportError:
        name = install_name or pkg
        print(f"\n❌  Missing package '{name}'. Run:  pip install {name}\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ExplainGrade — local student summary grader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--docx",      required=True, help="Path to the teacher transcript (.docx)")
    p.add_argument("--xlsx",      required=True, help="Path to the student summaries (.xlsx)")
    p.add_argument("--max-score", type=float, default=5.0, help="Maximum score per student (default: 5)")
    p.add_argument("--output",    default="grading_results.csv", help="Output CSV file (default: grading_results.csv)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# File readers
# ─────────────────────────────────────────────────────────────────

def read_docx(path: str) -> str:
    """Extract raw text from a .docx file."""
    docx = _require("docx", "python-docx")
    doc = docx.Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def read_xlsx(path: str):
    """Read an .xlsx file and return a list of row dicts."""
    pd = _require("pandas")
    _require("openpyxl")    # pandas needs this as its xlsx backend
    df = pd.read_excel(path, engine="openpyxl")
    df = df.fillna("")
    return df.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────
# Script refinement (mirrors JS refineTeacherScript logic)
# ─────────────────────────────────────────────────────────────────

STOP_WORDS = set("""
a an the and or but if in on at to for of with by from is are was were be been
being have has had do does did will would could should may might shall
i you he she it we they me him her us them my your his its our their
um uh ah like basically so well anyway yes no okay ok right yeah yep sure mhmm
""".split())

# Common verbs to exclude from technical term extraction (mirrors JS COMMON_VERBS)
COMMON_VERBS = {
    'use', 'used', 'uses', 'using', 'make', 'made', 'makes', 'making',
    'get', 'gets', 'got', 'give', 'gives', 'gave', 'take', 'takes', 'took',
    'come', 'goes', 'went', 'see', 'seen', 'look', 'looked', 'looks',
    'know', 'knew', 'show', 'shows', 'shown', 'need', 'needs', 'want',
    'let', 'help', 'call', 'called', 'keep', 'kept', 'put',
    'seem', 'seems', 'felt', 'feel', 'tell', 'told', 'mean', 'means',
    'provide', 'provides', 'include', 'includes', 'contain', 'contains',
    'allow', 'allows', 'require', 'requires', 'become', 'becomes',
    'refer', 'refers', 'result', 'results', 'occur', 'occurs',
    'describe', 'describes', 'define', 'defines', 'explain', 'explains',
    'involve', 'involves', 'perform', 'performs', 'produce', 'produces',
    'increase', 'decrease', 'change', 'changes', 'lead', 'leads',
    'work', 'works', 'move', 'moves', 'form', 'forms', 'pass', 'passes',
    'start', 'starts', 'begin', 'begins', 'end', 'ends', 'cause', 'causes',
}

# Global spaCy model (loaded once on first call, None if not installed)
_spacy_nlp = None
_spacy_available = None  # tri-state: None=unchecked, True=loaded, False=unavailable


def _get_spacy():
    """Lazy-load spaCy en_core_web_sm. Returns (nlp, True) or (None, False)."""
    global _spacy_nlp, _spacy_available
    if _spacy_available is not None:
        return _spacy_nlp, _spacy_available
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        _spacy_available = True
        print("✅  spaCy en_core_web_sm loaded — using POS-based technical noun extraction.")
    except (ImportError, OSError):
        _spacy_available = False
        print("ℹ   spaCy not found / model not installed — using enhanced keyword filter (≈ spaCy accuracy).")
        print("    To install: pip install spacy && python -m spacy download en_core_web_sm\n")
    return _spacy_nlp, _spacy_available


def extract_technical_terms_spacy(text: str, nlp) -> list:
    """
    spaCy-based technical noun extraction (mirrors Colab notebook logic exactly):
    - Keeps NOUN and PROPN tokens only
    - Removes stopwords, punctuation, digits, short tokens (<3 chars)
    - Removes DATE/TIME entities (timestamps, etc.)
    - Deduplicates (ordered set via dict)
    """
    doc = nlp(text)
    seen = {}
    for token in doc:
        if token.is_punct or token.is_space or token.is_digit:
            continue
        if token.pos_ not in ("NOUN", "PROPN"):
            continue
        lemma = token.lemma_.lower()
        if len(lemma) < 3:
            continue
        if token.is_stop:
            continue
        if token.tag_ == "UH":       # interjection
            continue
        if token.ent_type_ in ("DATE", "TIME"):
            continue
        if lemma not in seen:
            seen[lemma] = True
    return list(seen.keys())


def extract_technical_terms_fallback(text: str) -> list:
    """
    Fallback technical term extractor (no spaCy needed).
    Approximates NOUN/PROPN filtering:
    - Tokens ≥ 4 chars (filters short function words)
    - Not in STOP_WORDS
    - Not in COMMON_VERBS
    - Deduplicates
    """
    words = re.findall(r"[a-z]{4,}", text.lower())
    seen = {}
    for w in words:
        if w in STOP_WORDS or w in COMMON_VERBS:
            continue
        if w not in seen:
            seen[w] = True
    return list(seen.keys())


def extract_technical_terms(text: str) -> list:
    """Extract technical nouns from text. Uses spaCy if available, else fallback."""
    nlp, available = _get_spacy()
    if available:
        return extract_technical_terms_spacy(text, nlp)
    return extract_technical_terms_fallback(text)



def extract_keyphrases(text: str, n: int = 25):
    """Simple TF-based keyphrase extractor (no KeyBERT needed for the refiner)."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    freq = {}
    for w in words:
        if w not in STOP_WORDS:
            freq[w] = freq.get(w, 0) + 1
    # bigrams
    for i in range(len(words) - 1):
        bg = words[i] + " " + words[i + 1]
        if words[i] not in STOP_WORDS and words[i + 1] not in STOP_WORDS:
            freq[bg] = freq.get(bg, 0) + 1
    return sorted(freq, key=lambda k: -freq[k])[:n]


def refine_transcript(raw: str) -> str:
    """Remove timestamps, speaker names, filler lines from a Meet transcript."""
    lines_in = raw.split("\n")
    refined = []

    for i, line in enumerate(lines_in):
        line = line.strip()
        if not line:
            continue
        # Skip bare timestamps like "18:10" or "[00:15]"
        if re.fullmatch(r"\[?\d{1,2}:\d{2}(:\d{2})?\]?", line):
            continue
        # Skip speaker name if next line is a timestamp
        if i + 1 < len(lines_in) and re.fullmatch(r"\[?\d{1,2}:\d{2}(:\d{2})?\]?", lines_in[i + 1].strip()):
            continue
        # Strip inline timestamp prefix
        line = re.sub(r"^\[?\d{1,2}:\d{2}(:\d{2})?\]?\s*", "", line)
        # Strip speaker prefix "FirstName LastName: "
        if re.match(r"^[A-Z][a-zA-Z\s]+:", line):
            line = re.sub(r"^.*?:(.*)", r"\1", line).strip()
        # Skip system messages
        if re.search(r"joined the meeting|left the meeting|Attendees|Agentic AI Class", line, re.I):
            continue
        # Skip short fillers
        if len(line.split()) < 4 and re.fullmatch(
            r"(yes|no|okay|ok|right|yeah|yep|sure|mhmm)\.?", line, re.I
        ):
            continue
        if line:
            refined.append(line)

    clean = " ".join(refined)

    # Extractive summarisation: keep top 35% most informative sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9])", clean) if len(s.strip()) > 5]

    if len(sentences) > 8:
        anchors = set(extract_keyphrases(clean, 25))

        def score_sent(sent):
            lower = sent.lower()
            sc = sum(1 for a in anchors if a in lower)
            filler_count = len(re.findall(
                r"\b(i|you|he|she|it|we|they|me|him|her|us|them|um|uh|ah|like|basically|so|well|anyway)\b",
                lower
            ))
            sc -= filler_count * 0.5
            wc = len(sent.split())
            if 8 < wc < 30:
                sc += 0.5
            return sc

        scored = sorted(enumerate(sentences), key=lambda x: -score_sent(x[1]))
        top_n = max(5, math.ceil(len(sentences) * 0.35))
        top_idxs = sorted(i for i, _ in scored[:top_n])
        clean = " ".join(sentences[i] for i in top_idxs)

    return clean.strip()


# ─────────────────────────────────────────────────────────────────
# Sentence-transformers semantic grader
# ─────────────────────────────────────────────────────────────────

def load_model():
    """Download / load the all-MiniLM-L6-v2 sentence-transformer model."""
    SentenceTransformer = _require("sentence_transformers", "sentence-transformers").SentenceTransformer
    print("⬇  Loading sentence-transformers model (downloads ~90 MB on first run)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅  Model loaded.\n")
    return model


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"[a-z]{2,}", a.lower()))
    sb = set(re.findall(r"[a-z]{2,}", b.lower()))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def levenshtein_norm(a: str, b: str) -> float:
    """Normalised Levenshtein similarity (0–1)."""
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    # clip to 300 chars for speed
    a, b = a[:300], b[:300]
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j - 1] + 1, dp[j] + 1, prev[j - 1] + cost)
    dist = dp[lb]
    return 1.0 - dist / max(la, lb)


# def tf_cosine(ref: str, stu: str) -> float:
#     """TF-vector cosine similarity between two texts."""
#     ref_words = re.findall(r"[a-z]{2,}", ref.lower())
#     stu_words = re.findall(r"[a-z]{2,}", stu.lower())
#     vocab = set(ref_words) | set(stu_words)
#     if not vocab:
#         return 0.0
#     vocab = list(vocab)
#     ref_tf = {w: ref_words.count(w) for w in vocab}
#     stu_tf = {w: stu_words.count(w) for w in vocab}
#     v1 = [ref_tf.get(w, 0) for w in vocab]
#     v2 = [stu_tf.get(w, 0) for w in vocab]
#     return cosine_similarity(v1, v2)
# # C:\Users\deii\Desktop\cloud\local_grader.py

def tf_cosine(ref: str, stu: str) -> float:
    """
    FIX: Removed math.log(3/(1+doc_count)) which was causing Zero-IDF.
    Switched to raw Term Frequency Cosine Similarity.
    """
    import collections
    import math
    
    toks1 = _all_tokens(ref)
    toks2 = _all_tokens(stu)
    if not toks1 or not toks2: return 0.0

    tf1 = collections.Counter(toks1)
    tf2 = collections.Counter(toks2)
    all_terms = set(tf1) | set(tf2)

    dot = sum(tf1.get(t, 0) * tf2.get(t, 0) for t in all_terms)
    mag1 = math.sqrt(sum(v*v for v in tf1.values()))
    mag2 = math.sqrt(sum(v*v for v in tf2.values()))

    if mag1 == 0 or mag2 == 0: return 0.0
    return dot / (mag1 * mag2)

def normalized_word_count(ref: str, stu: str) -> float:
    ref_words = set(w for w in re.findall(r"[a-z]{3,}", ref.lower()) if w not in STOP_WORDS)
    stu_words = set(w for w in re.findall(r"[a-z]{3,}", stu.lower()) if w not in STOP_WORDS)
    if not ref_words:
        return 0.0
    return min(1.0, len(stu_words & ref_words) / len(ref_words))


def grade_answer(model, ref_embedding, ref: str, stu: str, max_score: float):
    """
    Two-stage grading: Colab-inspired rule-based floor + PMC12171532 NLP similarity.

    Stage 1 — Technical Term Coverage (Colab notebook approach)
        • Extract unique technical nouns from the reference (spaCy if available, else fallback)
        • Score = matched_technical_terms / total_technical_terms × max_score
        • Identical to the Colab: "Summary matched X key technical terms out of Y"

    Stage 2 — PMC12171532 NLP similarity score (unchanged)
    Final   = min(max_score, stage1 + stage2)
    """
    if not stu.strip():
        return {"stage1": 0.0, "stage2": 0.0, "final": 0.0, "matched_terms": 0, "total_terms": 0}

    _require("numpy")
    # ── Semantic embedding similarity (real model) ──────────────
    stu_embedding = model.encode(stu, convert_to_numpy=True)
    sem_sim = float(cosine_similarity(ref_embedding.tolist(), stu_embedding.tolist()))

    # ── Stage 1: Technical Term Coverage (Colab notebook approach) ──
    # Step 1: Extract unique technical nouns from reference
    ref_terms = extract_technical_terms(ref)          # list of unique lemmas/words
    total_ref_terms = len(ref_terms)

    if total_ref_terms == 0:
        stage1 = 0.0
        matched_count = 0
    else:
        # Step 2: All tokens of the student answer (for matching)
        stu_tokens = set(re.findall(r"[a-z]{3,}", stu.lower()))

        # Step 3: How many reference technical terms appear in student answer?
        matched_terms = [t for t in ref_terms if t in stu_tokens]
        matched_count = len(matched_terms)

        # Step 4: Coverage ratio → stage1 score
        coverage_ratio = matched_count / total_ref_terms         # 0–1
        stage1 = round(coverage_ratio * max_score, 3)

    # ── Stage 2: NLP similarity score ───────────────────────────
    s_j = jaccard(ref, stu)
    s_e = levenshtein_norm(ref[:300], stu[:300])
    s_c = tf_cosine(ref, stu)
    s_w = normalized_word_count(ref, stu)
    s_tf = sem_sim

    c_nlp = max(0.0, min(1.0, 0.15 * s_j + 0.05 * s_e + 0.15 * s_c + 0.15 * s_w))
    C = max(0.0, min(1.0, 0.5 * s_tf + 0.5 * c_nlp))

    if s_tf < 0.2:
        F = 0.0
    elif s_tf >= 0.9 and s_w >= 0.85:
        F = 1.0
    else:
        F = C

    stage2 = F * max_score

    final = min(max_score, stage1 + stage2)

    return {
        "stage1": round(stage1, 3),
        "stage2": round(stage2, 3),
        "final":  round(final,  3),
        "semantic": round(sem_sim, 3),
        "jaccard":  round(s_j, 3),
    }


# ─────────────────────────────────────────────────────────────────
# Column detection
# ─────────────────────────────────────────────────────────────────

def find_col(keys, *patterns):
    for pat in patterns:
        for k in keys:
            if re.search(pat, k, re.I):
                return k
    return None


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Validate paths
    for f, label in [(args.docx, "--docx"), (args.xlsx, "--xlsx")]:
        if not Path(f).exists():
            print(f"\n❌  File not found for {label}: {f}\n")
            sys.exit(1)

    print("=" * 60)
    print("  ExplainGrade — Local Grader")
    print("=" * 60)

    # 1. Read and refine transcript
    print(f"\n📄  Reading transcript: {args.docx}")
    raw_script = read_docx(args.docx)
    print(f"    → {len(raw_script):,} characters read.")

    print("🧠  Refining transcript (extractive summarisation)…")
    reference = refine_transcript(raw_script)
    print(f"    → Refined to {len(reference):,} characters / ~{len(reference.split())} words.")

    if len(reference) < 20:
        print("\n⚠  Refined transcript is very short — check the .docx format.")
        sys.exit(1)

    # 2. Read summaries
    print(f"\n📊  Reading summaries: {args.xlsx}")
    rows = read_xlsx(args.xlsx)
    print(f"    → {len(rows)} student rows found.")

    if not rows:
        print("\n⚠  No data rows found in the XLSX file.")
        sys.exit(1)

    keys = list(rows[0].keys())
    col_email   = find_col(keys, r"email", r"mail")
    col_name    = find_col(keys, r"name", r"student")
    col_summary = find_col(keys, r"summary", r"answer", r"response")

    if not col_email or not col_summary:
        print(f"\n❌  Could not locate required columns.\n    Found columns: {keys}")
        print("    Expected: emailAddress / email, name, summary / answer / response\n")
        sys.exit(1)

    print(f"    → Columns: email='{col_email}', name='{col_name}', summary='{col_summary}'")

    # 3. Load model
    model = load_model()

    # 4. Encode reference once (for speed)
    print("🔢  Encoding reference answer…")
    numpy = _require("numpy")
    ref_embedding = model.encode(reference, convert_to_numpy=True)

    # 5. Grade each student
    print(f"⚡  Grading {len(rows)} students (max score = {args.max_score})…\n")
    results = []

    for i, row in enumerate(rows, 1):
        email   = str(row.get(col_email, "Unknown")).strip() or "Unknown"
        name    = str(row.get(col_name,  "N/A")).strip()      if col_name else "N/A"
        summary = str(row.get(col_summary, "")).strip()

        if not summary:
            results.append({
                "email": email, "name": name,
                "stage1": 0.0, "stage2": 0.0, "final": 0.0,
                "semantic": 0.0, "jaccard": 0.0,
                "note": "No summary provided"
            })
            print(f"  [{i:>3}/{len(rows)}]  {email[:40]:<40}  Score: 0.00 / {args.max_score}  (no summary)")
            continue

        res = grade_answer(model, ref_embedding, reference, summary, args.max_score)
        results.append({
            "email": email, "name": name, **res, "note": ""
        })
        bar = "█" * int((res["final"] / args.max_score) * 20)
        print(f"  [{i:>3}/{len(rows)}]  {email[:40]:<40}  Score: {res['final']:.2f} / {args.max_score}  |{bar:<20}|")

    # 6. Sort by score descending
    results.sort(key=lambda r: -r["final"])

    # 7. Export CSV
    out_path = args.output
    fieldnames = ["email", "name", "final", "stage1", "stage2", "semantic", "jaccard", "note"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 60}")
    print(f"✅  Done! {len(results)} students graded.")
    print(f"📁  Results saved to: {os.path.abspath(out_path)}")

    # Quick summary stats
    scores = [r["final"] for r in results]
    if scores:
        avg = sum(scores) / len(scores)
        high = max(scores)
        low  = min(scores)
        print(f"\n📈  Score summary  (max={args.max_score})")
        print(f"    Average : {avg:.2f}")
        print(f"    Highest : {high:.2f}")
        print(f"    Lowest  : {low:.2f}")
    print()


if __name__ == "__main__":
    main()
