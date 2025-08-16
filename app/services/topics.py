# app/services/topics.py

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from typing import List, Tuple

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
import spacy
from spellchecker import SpellChecker
from nltk.stem.snowball import SnowballStemmer
from transformers import pipeline

from anthropic import Anthropic
from app.config import USE_LLM_TOPICS, ANTHROPIC_API_KEY


# -----------------------------
# Global initializers
# -----------------------------
# Optional Anthropic client
_anthropic: Anthropic | None = (
    Anthropic(api_key=ANTHROPIC_API_KEY) if (USE_LLM_TOPICS and ANTHROPIC_API_KEY) else None
)
_LLM_MODEL = "claude-sonnet-4-20250514"  # set to a model available to you

# Local summarizer (optional, used in both Anthropic and fallback paths)
# OK on CPU for short texts; try/except in case model can't be loaded.
try:
    _summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    _summarizer = None

# Classic components
_kw_model = KeyBERT()
_emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
_stemmer = SnowballStemmer("english")
_spell = SpellChecker()
# Keep parser ON (noun_chunks require it). Disable only NER for speed.
_nlp = spacy.load("en_core_web_sm", disable=["ner"])


# -----------------------------
# Anthropic LLM path (topics + summary)
# -----------------------------
def _llm_topics_and_summary(text: str, top_k: int = 5) -> Tuple[List[str], List[float], str]:
    """
    Ask Claude for K distinct topics (2–5 word noun phrases), weights summing to 1,
    and a 2–3 sentence summary. Returns (topics, weights, summary).
    Raises if Anthropic is not configured or output is invalid.
    """
    if not _anthropic:
        raise RuntimeError("Anthropic client not configured")

    system_msg = (
        "You are a careful NLP assistant. Extract discussion topics and write a short summary.\n"
        "Rules:\n"
        f"- Return EXACTLY {top_k} distinct topics.\n"
        "- Topics must be concise noun phrases (2-5 words), avoid verbs where possible.\n"
        "- Do not include commas inside a single topic.\n"
        "- Provide topic_weights that are non-negative and sum to 1.0 (within 1e-6).\n"
        "- Provide a fluent 2-3 sentence abstractive summary.\n"
        "- Respond ONLY with a compact JSON object: {\"topics\": [...], \"topic_weights\": [...], \"summary\": \"...\"}.\n"
        "- No extra text, no Markdown.\n"
    )
    user_payload = {
        "instruction": (
            f"From the following transcript, extract exactly {top_k} distinct conversation topics "
            "and a 2-3 sentence summary. Topics must be 2-5 word noun phrases, no commas inside a topic."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "topics": {"type": "array", "items": {"type": "string"}},
                "topic_weights": {"type": "array", "items": {"type": "number"}},
                "summary": {"type": "string"},
            },
            "required": ["topics", "topic_weights", "summary"],
        },
        "transcript": text,
    }

    resp = _anthropic.messages.create(
        model=_LLM_MODEL,
        max_tokens=1000,
        temperature=0.2,
        system=system_msg,
        messages=[{"role": "user", "content": json.dumps(user_payload)}],
    )

    content = resp.content[0].text if resp.content and hasattr(resp.content[0], "text") else ""
    data = json.loads(content)

    topics = [t.strip() for t in data["topics"]]
    weights = [float(w) for w in data["topic_weights"]]
    summary = (data["summary"] or "").strip()

    # Enforce size == top_k
    if len(topics) != top_k or len(weights) != top_k:
        topics = (topics + topics[:top_k])[:top_k]
        weights = (weights + [1.0 / top_k] * top_k)[:top_k]

    # Sanitize topics (no commas, max 6 tokens, Title Case-ish)
    clean_topics = []
    for t in topics:
        t = " ".join(t.replace(",", " ").split())
        toks = t.split()
        t = " ".join(toks[:6])
        t = " ".join(w if w.isupper() else w.capitalize() for w in t.split())
        clean_topics.append(t)
    topics = clean_topics

    # Renormalize weights
    s = sum(weights) or 1.0
    weights = [w / s for w in weights]

    # If we have a local summarizer, optionally refine summary a bit
    if _summarizer and len(summary.split()) < 30 and len(text) > 120:
        try:
            summary = _summarizer(
                text, max_length=220, min_length=80, do_sample=False, truncation=True
            )[0]["summary_text"].strip()
        except Exception:
            pass

    return topics, weights, summary


# -----------------------------
# Fallback summarizer (BART) or TF-IDF
# -----------------------------
def _llm_summarize(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if _summarizer:
        try:
            return _summarizer(
                text, max_length=220, min_length=80, do_sample=False, truncation=True
            )[0]["summary_text"].strip()
        except Exception:
            pass
    # Extractive fallback (TF-IDF top 1–2 sentences)
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return ""
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(sentences)
    scores = X.mean(axis=1).A.ravel()
    top_idxs = scores.argsort()[::-1][:2]
    return ". ".join(sentences[i] for i in sorted(top_idxs)) + "."


# -----------------------------
# Classic topic pipeline helpers (no hardcoding)
# -----------------------------
def _normalize_topic(s: str) -> str:
    """
    Algorithmic normalization for bucketing (no hardcoded maps):
      - lowercase
      - keep alphanumerics, tokenize
      - drop stopwords and short tokens
      - stem tokens with Snowball
      - de-duplicate stems while preserving order
    """
    s = s.lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    tokens = [t for t in tokens if len(t) > 2 and t not in ENGLISH_STOP_WORDS]
    stems = [_stemmer.stem(t) for t in tokens]
    seen, kept = set(), []
    for t in stems:
        if t not in seen:
            kept.append(t)
            seen.add(t)
    return " ".join(kept)


def _merge_similar_topics(topics: List[str], weights: List[float], keep_k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Merge normalized duplicates by summing weights, keep the best original phrase,
    then renormalize and keep exactly top_k.
    """
    buckets = defaultdict(lambda: {"label": None, "label_w": 0.0, "weight": 0.0})
    for label, w in zip(topics, weights):
        key = _normalize_topic(label)
        if not key:
            continue
        b = buckets[key]
        b["weight"] += float(w)
        if float(w) > b["label_w"]:
            b["label"] = label
            b["label_w"] = float(w)

    merged_topics, merged_weights = [], []
    for _, v in buckets.items():
        merged_topics.append(v["label"] or "")
        merged_weights.append(v["weight"])

    total = sum(merged_weights) or 1.0
    merged_weights = [w / total for w in merged_weights]

    order = sorted(range(len(merged_topics)), key=lambda i: merged_weights[i], reverse=True)[:keep_k]
    merged_topics = [merged_topics[i] for i in order]
    merged_weights = [merged_weights[i] for i in order]

    total_k = sum(merged_weights) or 1.0
    merged_weights = [w / total_k for w in merged_weights]
    return merged_topics, merged_weights


def _semantic_merge(phrases: List[str], weights: List[float], threshold: float = 0.82, keep_k: int = 5) -> Tuple[List[str], List[float]]:
    """
    Cluster paraphrases using sentence embeddings; label = highest-weight phrase in cluster.
    """
    if not phrases:
        return [], []
    embs = _emb_model.encode(phrases, normalize_embeddings=True)
    used = [False] * len(phrases)
    merged_labels, merged_weights = [], []

    for i in range(len(phrases)):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(phrases)):
            if used[j]:
                continue
            sim = float(np.dot(embs[i], embs[j]))
            if sim >= threshold:
                used[j] = True
                group.append(j)
        idx = max(group, key=lambda k: weights[k])
        merged_labels.append(phrases[idx])
        merged_weights.append(sum(weights[k] for k in group))

    total = sum(merged_weights) or 1.0
    merged_weights = [w / total for w in merged_weights]
    order = sorted(range(len(merged_labels)), key=lambda i: merged_weights[i], reverse=True)[:keep_k]
    merged_labels = [merged_labels[i] for i in order]
    merged_weights = [merged_weights[i] for i in order]

    total_k = sum(merged_weights) or 1.0
    merged_weights = [w / total_k for w in merged_weights]
    return merged_labels, merged_weights


def _edit_dist(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j - 1], dp[j])
            prev, dp[j] = dp[j], cur
    return dp[n]


def _spellfix_token(tok: str) -> str:
    """
    Conservative spell fix to avoid 'Mars' -> 'Mar' type errors.
    Accept only small edits (<=1) and similar length.
    """
    t = tok.strip()
    if not (t.isalpha() and len(t) > 2):
        return t

    cand = _spell.candidates(t)
    if not cand:
        return t
    corr = _spell.correction(t)
    if not corr or corr == t:
        return t

    dist = _edit_dist(t.lower(), corr.lower())
    if dist <= 1 and abs(len(corr) - len(t)) <= 2:
        return corr
    return t


def _canonicalize_label(label: str) -> str:
    """
    Automatic label cleanup (no hardcoded rules):
      - lower, basic punctuation->space, split hyphens
      - conservative per-token spellfix
      - prefer noun chunks; fallback to POS-filtered NOUN/PROPN/ADJ lemmas
      - remove stopwords/short tokens; de-duplicate while preserving order
      - Title Case (preserve ALL-CAPS tokens)
    """
    s = re.sub(r"[^a-zA-Z0-9\s\-]", " ", label.lower()).replace("-", " ").strip()
    if not s:
        return ""

    toks = [_spellfix_token(t) for t in s.split()]
    doc = _nlp(" ".join(toks))

    # 1) try noun chunks
    chunks = []
    for nc in doc.noun_chunks:
        words = [t for t in nc if not t.is_stop and len(t.text) > 2]
        if words:
            chunks.append(" ".join(t.text for t in words))
    candidate = " ".join(chunks).strip()

    # 2) fallback: POS-filtered NOUN/PROPN/ADJ lemmas
    if not candidate:
        kept, seen = [], set()
        for t in doc:
            if t.is_stop or len(t.text) <= 2:
                continue
            if t.pos_ not in {"NOUN", "PROPN", "ADJ"}:
                continue
            lemma = (t.lemma_ or t.text).strip()
            if len(lemma) <= 2:
                continue
            if lemma not in seen:
                kept.append(lemma)
                seen.add(lemma)
        candidate = " ".join(kept).strip()
    if not candidate:
        candidate = " ".join(toks)

    candidate = re.sub(r"\s+", " ", candidate).strip()
    pretty = " ".join(w if w.isupper() else w.capitalize() for w in candidate.split())
    return pretty


# -----------------------------
# Public API
# -----------------------------
def extract_topics_and_summary(text: str, top_k: int = 5) -> Tuple[List[str], List[float], str]:
    """
    Preferred path: Anthropic (Claude) returns clean topics + weights + summary.
    Fallback: KeyBERT + merges + canonicalize + LLM (BART) or TF-IDF summary.
    Always returns exactly top_k topics and weights that sum to 1.0.
    """
    text = (text or "").strip()
    if not text:
        return [], [], ""

    # 1) Anthropic path (if enabled)
    if _anthropic:
        try:
            topics, weights, summary = _llm_topics_and_summary(text, top_k=top_k)
            if len(topics) != top_k or len(weights) != top_k:
                topics = (topics + topics[:top_k])[:top_k]
                weights = (weights + [1.0 / top_k] * top_k)[:top_k]
            s = sum(weights) or 1.0
            weights = [w / s for w in weights]
            return topics, weights, summary
        except Exception:
            # fall back silently
            pass

    # 2) Fallback: classic KeyBERT pipeline
    candidates = _kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(2, 3),
        stop_words="english",
        top_n=25,
        use_mmr=True,
        diversity=0.7,
    )
    raw_topics = [t for t, _ in candidates]
    raw_scores = np.array([w for _, w in candidates], dtype=float)

    if raw_scores.size == 0:
        return [], [], _llm_summarize(text)

    total = float(raw_scores.sum())
    weights = (raw_scores / total) if total > 0 else np.full(len(raw_topics), 1.0 / max(1, len(raw_topics)))
    weights = weights.tolist()

    # merge (text-normalized)
    topics, weights = _merge_similar_topics(raw_topics, weights, keep_k=top_k)
    # semantic merge
    topics, weights = _semantic_merge(topics, weights, threshold=0.82, keep_k=top_k)
    # canonicalize labels
    topics = [_canonicalize_label(t) for t in topics]
    # final merge & enforce size
    topics, weights = _merge_similar_topics(topics, weights, keep_k=top_k)

    if len(topics) < top_k:
        used = set(t.lower() for t in topics)
        for t, _ in candidates:
            if len(topics) >= top_k:
                break
            if t.lower() in used:
                continue
            topics.append(_canonicalize_label(t))
            used.add(t.lower())
        weights = [1.0 / len(topics)] * len(topics)

    if len(topics) > top_k:
        topics = topics[:top_k]
        weights = weights[:top_k]

    s = sum(weights) or 1.0
    weights = [w / s for w in weights]

    summary = _llm_summarize(text)
    return topics, weights, summary
