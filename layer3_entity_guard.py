"""Lightweight entity-anchor consistency checks for Layer 3 retrieval."""

import math
import re
from dataclasses import dataclass
from typing import Dict, Mapping, Set, Tuple


SENTENCE_STARTERS = {
    "what", "where", "who", "whom", "whose", "when", "which", "why", "how",
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "can", "could", "will", "would", "should", "may", "might", "has", "have",
    "tell", "give", "list", "show", "name", "state", "find", "get", "explain",
    "define", "describe", "in", "on", "at", "for", "from", "to", "by", "with",
    "of", "about", "under", "during", "according", "please", "provide",
}
NON_ENTITY_ACRONYMS = {"id"}
NON_ENTITY_TITLECASE = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
}

HARD_ID_RE = re.compile(
    r"\b(?:"
    r"[A-Za-z]+[-_.]\d+[A-Za-z0-9._-]*"
    r"|\d+[-_][A-Za-z]+[A-Za-z0-9._-]*"
    r"|[A-Za-z]+\d+[A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*"
    r")\b"
)
QUOTED_RE = re.compile(r"[\"']([^\"']{2,40})[\"']")
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*")
QUARTER_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(?:19\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class AnchorSet:
    hard_ids: Set[str]
    proper_nouns: Set[str]
    all_tokens: Set[str]


def _normalize_token(token: str) -> str:
    token = token.strip().lower()
    if token.endswith(("'s", "’s")):
        token = token[:-2]
    return token


def extract_entity_anchors(text: str) -> AnchorSet:
    """Extract coded IDs, quoted literals, and cased proper-noun anchors."""
    hard_ids = {
        normalized
        for match in HARD_ID_RE.finditer(text)
        if (normalized := _normalize_token(match.group(0))) not in {"q1", "q2", "q3", "q4"}
    }
    cleaned = re.sub(r"['’]s\b", "", text)
    raw_tokens = TOKEN_RE.findall(cleaned)
    all_tokens: Set[str] = set()
    for token in raw_tokens:
        normalized = _normalize_token(token)
        all_tokens.add(normalized)
        if "-" in normalized or "_" in normalized or "." in normalized:
            all_tokens.update(part for part in re.split(r"[-_.]", normalized) if part)

    alpha_tokens = [token for token in raw_tokens if any(char.isalpha() for char in token)]
    is_all_caps = len(alpha_tokens) >= 3 and all(token.isupper() for token in alpha_tokens)
    proper_nouns: Set[str] = set()
    if not is_all_caps:
        for index, token in enumerate(raw_tokens):
            normalized = _normalize_token(token)
            if normalized in hard_ids or len(normalized) <= 1 or normalized.isdigit():
                continue
            is_title = token[0].isupper() and any(char.islower() for char in token[1:])
            is_acronym = token.isupper() and len(token) >= 2 and token.isalpha()
            if is_acronym:
                if normalized not in NON_ENTITY_ACRONYMS:
                    proper_nouns.add(normalized)
            elif (
                is_title
                and normalized not in NON_ENTITY_TITLECASE
                and not (index == 0 and normalized in SENTENCE_STARTERS)
            ):
                proper_nouns.add(normalized)

    for quoted_span in QUOTED_RE.findall(cleaned):
        for token in TOKEN_RE.findall(quoted_span):
            normalized = _normalize_token(token)
            if len(normalized) > 1 and normalized not in SENTENCE_STARTERS:
                proper_nouns.add(normalized)

    return AnchorSet(hard_ids, proper_nouns, all_tokens)


def normalized_idf(token: str, df_live: Mapping[str, int], n_live: int) -> float:
    """Return scale-normalized IDF; unseen (df=0) terms score 1.0."""
    if n_live <= 0:
        return 1.0
    df = df_live.get(token, 0)
    return math.log((n_live + 1.0) / (df + 1.0)) / math.log(n_live + 1.0)


def has_temporal_conflict(query: str, candidate: str) -> Tuple[bool, str]:
    """Reject only when both sides specify disjoint quarters or years."""
    query_quarters = {f"q{match.group(1)}" for match in QUARTER_RE.finditer(query)}
    candidate_quarters = {f"q{match.group(1)}" for match in QUARTER_RE.finditer(candidate)}
    if query_quarters and candidate_quarters and query_quarters.isdisjoint(candidate_quarters):
        return True, f"quarter_conflict:{sorted(query_quarters)}!={sorted(candidate_quarters)}"

    query_years = set(YEAR_RE.findall(query))
    candidate_years = set(YEAR_RE.findall(candidate))
    if query_years and candidate_years and query_years.isdisjoint(candidate_years):
        return True, f"year_conflict:{sorted(query_years)}!={sorted(candidate_years)}"
    return False, ""


def high_idf_anchors(
    anchors: Set[str],
    df_live: Mapping[str, int],
    n_live: int,
    tau_nidf: float = 0.60,
) -> Set[str]:
    return {token for token in anchors if normalized_idf(token, df_live, n_live) >= tau_nidf}


def entity_anchor_guard(
    query: str,
    cand_key: str,
    df_live: Mapping[str, int],
    n_live: int,
    tau_nidf: float = 0.60,
    structured_key: bool = True,
) -> Tuple[bool, Dict]:
    """Check temporal conflicts, hard IDs, and bidirectional entity coverage."""
    query_anchors = extract_entity_anchors(query)
    key_anchors = extract_entity_anchors(cand_key)

    missing_query_ids = query_anchors.hard_ids - key_anchors.all_tokens
    missing_key_ids = key_anchors.hard_ids - query_anchors.all_tokens
    if missing_query_ids or missing_key_ids:
        return False, {
            "reason": "hard_id_mismatch",
            "missing_q_ids": sorted(missing_query_ids),
            "missing_k_ids": sorted(missing_key_ids),
        }

    temporal_conflict, temporal_detail = has_temporal_conflict(query, cand_key)
    if temporal_conflict:
        return False, {"reason": temporal_detail.split(":", 1)[0], "detail": temporal_detail}

    query_high = high_idf_anchors(query_anchors.proper_nouns, df_live, n_live, tau_nidf)
    missing_query_props = query_high - key_anchors.all_tokens
    if missing_query_props:
        return False, {
            "reason": "forward_entity_mismatch",
            "q_high": sorted(query_high),
            "missing_q_prop": sorted(missing_query_props),
        }

    if structured_key:
        key_high = high_idf_anchors(key_anchors.proper_nouns, df_live, n_live, tau_nidf)
        if key_high:
            overlap = key_high & query_anchors.all_tokens
            if not query_anchors.proper_nouns:
                missing_key_props = key_high - query_anchors.all_tokens
                if missing_key_props:
                    return False, {
                        "reason": "reverse_lowercase_entity_mismatch",
                        "k_high": sorted(key_high),
                        "missing_k_prop": sorted(missing_key_props),
                    }
            elif not overlap:
                return False, {
                    "reason": "reverse_zero_anchor_overlap",
                    "k_high": sorted(key_high),
                }

    return True, {"reason": "pass", "q_high": sorted(query_high)}


def entity_guard_validation_cases():
    """Entity and temporal checks using a synthetic N=100 live DF snapshot."""
    n_live = 100
    df_live = {
        "project": 25,
        "dr": 25,
        "elena": 2,
        "kestrel": 1,
        "orion": 1,
        "rostova": 1,
        "vance": 1,
    }
    cases = [
        ("Forgotten proper noun", "What is the budget for Project Kestrel?", "What is the budget for Project Orion?", False),
        ("Lowercase forgotten noun", "what is project kestrel's budget?", "What is the budget for Project Orion?", False),
        ("Lowercase person mismatch", "what is elena vance's badge number?", "What is Dr. Elena Rostova's badge number?", False),
        ("Retained unseen paraphrase", "Tell me Project Kestrel's secret passcode", "What is the access code for Project Kestrel?", True),
        ("Retained lowercase paraphrase", "tell me project kestrel's secret passcode", "What is the access code for Project Kestrel?", True),
        ("Retained surname only", "What is Vance's badge number?", "What is Dr. Elena Vance's badge number?", True),
        ("Conflicting quarters", "What is the Q3 2026 revenue forecast for Graphene?", "What is the Q4 2026 revenue forecast for Graphene?", False),
        ("Quarter to month paraphrase", "Tell me the codename assigned to the Q3 2026 Graphene lot Lot-023.", "What is the batch identifier for the October 2026 Graphene Lot-023?", True),
        ("Quarter to undated paraphrase", "What is the Q3 2026 forecast?", "What is the forecast?", True),
        ("Conflicting years", "What is the 2025 Graphene revenue forecast?", "What is the 2026 Graphene revenue forecast?", False),
    ]
    return [
        {
            "name": name,
            "query": query,
            "key": key,
            "expected": expected,
            "allowed": entity_anchor_guard(query, key, df_live, n_live)[0],
            "diagnostics": entity_anchor_guard(query, key, df_live, n_live)[1],
        }
        for name, query, key, expected in cases
    ]


def print_entity_guard_validation() -> bool:
    """Print the Tier 2C entity/temporal matrix and return whether it passed."""
    rows = entity_guard_validation_cases()
    print("\nTIER 2C: generalized entity-anchor guard checks (synthetic N=100 DF)")
    print(f"{'Case':<31} | {'Expected':<8} | {'Actual':<8} | Result | Guard path")
    print("-" * 100)
    for row in rows:
        expected = "PASS" if row["expected"] else "BLOCK"
        actual = "PASS" if row["allowed"] else "BLOCK"
        outcome = "OK" if row["expected"] == row["allowed"] else "FAIL"
        print(f"{row['name']:<31} | {expected:<8} | {actual:<8} | {outcome:<6} | {row['diagnostics']['reason']}")
    passed = all(row["expected"] == row["allowed"] for row in rows)
    print(f"Tier 2C guard checks: {sum(row['expected'] == row['allowed'] for row in rows)}/{len(rows)} passed")
    return passed
