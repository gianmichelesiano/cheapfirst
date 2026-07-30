"""Verifica della risposta. ACCEPT, REVISE o FAIL."""

from enum import Enum
from .classifier import TaskSignature
import re


class Verdict(Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    FAIL = "FAIL"


def verify_response(response: dict, sig: TaskSignature) -> Verdict:
    """Verifica se la risposta e accettabile con euristiche migliorate."""
    text = response.get("text", "")
    if not text or not text.strip():
        return Verdict.REVISE

    if sig.task == "code":
        return _verify_code(text)
    elif sig.task == "translation":
        return _verify_translation(text)
    elif sig.task == "factual":
        return _verify_factual(text)
    elif sig.task == "math":
        return _verify_math(text)
    else:
        return _verify_general(text)


def _verify_code(text: str) -> Verdict:
    """Verifica codice: controlla presenza di codice e parentesi bilanciate."""
    has_code = bool(re.search(r"```|def |class |function |import |const |let |var ", text))
    if not has_code:
        return Verdict.REVISE

    code_blocks = re.findall(r"```[\s\S]*?```", text)
    if not code_blocks:
        if len(text) < 2000:
            if not _balanced(text, "{", "}"):
                return Verdict.REVISE
            if not _balanced(text, "(", ")"):
                return Verdict.REVISE
    else:
        for block in code_blocks:
            clean = re.sub(r"```\w*\n?", "", block)
            if not _balanced(clean, "{", "}"):
                return Verdict.REVISE
            if not _balanced(clean, "(", ")"):
                return Verdict.REVISE

    return Verdict.ACCEPT


def _balanced(text: str, open_c: str, close_c: str) -> bool:
    count = 0
    for ch in text:
        if ch == open_c:
            count += 1
        elif ch == close_c:
            count -= 1
        if count < 0:
            return False
    return count == 0


def _verify_translation(text: str) -> Verdict:
    stripped = text.strip()
    if len(stripped) < 5:
        return Verdict.REVISE
    if len(stripped.split()) == 1 and len(stripped) < 20:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_factual(text: str) -> Verdict:
    stripped = text.strip()
    if len(stripped) < 30:
        return Verdict.REVISE
    if len(stripped.split()) < 5:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_math(text: str) -> Verdict:
    stripped = text.strip()
    has_numbers = bool(re.search(r'\d+', text))
    has_math = bool(re.search(r'[=≈≠≤≥±√∫∑∏∞]|\\frac|\\sum|\\int', text))
    if not has_numbers and not has_math:
        return Verdict.REVISE
    if len(stripped) < 10:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_general(text: str) -> Verdict:
    stripped = text.strip()
    if len(stripped) < 15:
        return Verdict.REVISE
    return Verdict.ACCEPT
