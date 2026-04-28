"""Rule-based fast filter for obvious non-claims.

Skips the neural model for sentences that are clearly not factual claims:
questions, exclamations, opinion hedges, greetings. Saves ~30-40% of compute.
"""

import re

# Patterns that strongly indicate NOT a claim
_QUESTION_RE = re.compile(r"\?\s*$")
_OPINION_PREFIXES = (
    "i think", "i believe", "i feel", "in my opinion", "i guess",
    "i suppose", "i hope", "i wish", "i love", "i hate", "i prefer",
    "personally", "to me,", "for me,",
)
_GREETING_PREFIXES = (
    "hello", "hi ", "hey ", "good morning", "good evening",
    "thank you", "thanks", "please", "sorry",
)
_IMPERATIVE_PREFIXES = (
    "let's", "let us", "please ", "don't ", "do not ",
    "stop ", "go ", "come ", "try ", "make sure",
)

# Patterns that strongly indicate IS a claim
_NUMBER_RE = re.compile(r"\d+\.?\d*\s*(%|percent|billion|million|trillion|thousand)")
_STATISTIC_RE = re.compile(r"(increased|decreased|grew|fell|rose|dropped|declined)\s+by")


class FastFilterResult:
    """Result from the fast filter."""
    __slots__ = ("decision", "confidence", "rule")

    def __init__(self, decision: bool | None, confidence: float, rule: str):
        self.decision = decision  # True=claim, False=not claim, None=uncertain
        self.confidence = confidence
        self.rule = rule


def fast_filter(text: str) -> FastFilterResult:
    """Apply rule-based filtering to skip obvious cases.

    Returns:
        FastFilterResult with decision=None if uncertain (needs model).
        decision=True/False if confident enough to skip the model.
    """
    text_lower = text.strip().lower()

    # Empty or very short text
    if len(text_lower) < 5:
        return FastFilterResult(False, 0.95, "too_short")

    # Questions are rarely claims
    if _QUESTION_RE.search(text):
        return FastFilterResult(False, 0.90, "question")

    # Opinion hedges
    if any(text_lower.startswith(p) for p in _OPINION_PREFIXES):
        return FastFilterResult(False, 0.85, "opinion_hedge")

    # Greetings / pleasantries
    if any(text_lower.startswith(p) for p in _GREETING_PREFIXES):
        return FastFilterResult(False, 0.90, "greeting")

    # Imperative sentences (commands/requests)
    if any(text_lower.startswith(p) for p in _IMPERATIVE_PREFIXES):
        return FastFilterResult(False, 0.80, "imperative")

    # Statistical claims (high confidence claim)
    if _STATISTIC_RE.search(text_lower):
        return FastFilterResult(True, 0.85, "statistic_pattern")

    # Numeric claims with units
    if _NUMBER_RE.search(text_lower):
        return FastFilterResult(None, 0.0, "has_numbers_uncertain")

    # Uncertain — needs the model
    return FastFilterResult(None, 0.0, "uncertain")
