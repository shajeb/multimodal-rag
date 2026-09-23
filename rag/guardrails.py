"""Heuristic input checks plus prompt separation; not a security guarantee."""
import re

INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions"
    r"|(?:reveal|print|show|expose)\s+(?:your\s+)?(?:system\s+prompt|api\s*key|secrets)"
    r"|you\s+are\s+now\s+(?:a|an)\s+"
    r"|<\|(?:system|assistant|im_start)\|>",
    re.IGNORECASE,
)


def validate_question(text):
    if not text.strip() or len(text) > 8000:
        raise ValueError("Question must contain 1-8000 characters")
    if INJECTION.search(text):
        raise ValueError("The request contains an instruction-override or secret-extraction pattern")


def safe_context(text):
    return not INJECTION.search(text)


def citation_labels(answer):
    """Read single or comma-grouped evidence labels, such as [S1, S2]."""
    groups = re.findall(r"\[(S\d+(?:\s*,\s*S\d+)*)\]", answer)
    return {label for group in groups for label in re.findall(r"S(\d+)", group)}
