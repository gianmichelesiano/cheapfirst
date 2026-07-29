"""Verifica della risposta. ACCEPT, REVISE o FAIL."""

from enum import Enum
from .classifier import TaskSignature
import re


class Verdict(Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    FAIL = "FAIL"


def verify_response(response: dict, sig: TaskSignature) -> Verdict:
    """Verifica se la risposta è accettabile.

    Usa controlli strutturati in base al task type.
    """
    text = response.get("text", "")
    if not text:
        return Verdict.REVISE

    # Controllo base: risposta non vuota
    if len(text.strip()) < 5:
        return Verdict.REVISE

    # Controlli per tipo di task
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
    """Verifica codice: controlla sintassi di base."""
    # Deve contenere almeno codice
    has_code = bool(re.search(r"```|def |class |function |import |const |let |var ", text))
    if not has_code:
        return Verdict.REVISE

    # Controllo base: parentesi bilanciate
    opens = text.count("{")
    closes = text.count("}")
    if opens > 0 and opens != closes:
        return Verdict.REVISE

    opens_p = text.count("(")
    closes_p = text.count(")")
    if opens_p > 0 and opens_p != closes_p:
        return Verdict.REVISE

    return Verdict.ACCEPT


def _verify_translation(text: str) -> Verdict:
    """Verifica traduzione: controlla che non sia la lingua originale."""
    # Se la risposta è molto corta e sembra identica all'input, potrebbe essere un errore
    if len(text.strip()) < 3:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_factual(text: str) -> Verdict:
    """Verifica risposta fattuale."""
    # Controlla che la risposta sia sostanziosa
    if len(text.strip()) < 20:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_math(text: str) -> Verdict:
    """Verifica risposta matematica."""
    # Deve contenere numeri o formule
    has_numbers = bool(re.search(r'\d+', text))
    has_math = bool(re.search(r'=|≈|≠|≤|≥|±|√|∫|∑|∏|∞', text))
    if not has_numbers and not has_math and len(text) < 50:
        return Verdict.REVISE
    return Verdict.ACCEPT


def _verify_general(text: str) -> Verdict:
    """Verifica risposta generale."""
    if len(text.strip()) < 10:
        return Verdict.REVISE
    return Verdict.ACCEPT
