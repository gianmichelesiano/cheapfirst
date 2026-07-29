"""
Classificatore euristico (zero cost, zero chiamate LLM).

Classifica il task in base a pattern regex, calcola difficoltà e confidenza.
"""

from dataclasses import dataclass
import re


@dataclass
class TaskSignature:
    task: str          # code|math|creative|factual|translation|analysis|general
    difficulty: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    caps: list[str] = None       # ["multimodal", "128k", ...]
    sensitive: bool = False      # PII detection
    freshness: bool = False      # news/time-sensitive


# ── Pattern per tipo di task ─────────────────────────────────────────────
CODE_RE = re.compile(
    r"```|\b(function|class|def |import |const |let |var |async |await|"
    r"return|npm |pip |regex|stack.?trace|exception|compile|null pointer|"
    r"segfault|typescript|python|javascript|rust|golang)\b|"
    r"\.(ts|js|py|rs|go|java|cpp|sql)\b",
    re.I,
)

MATH_RE = re.compile(
    r"\b(integral|derivative|equation|theorem|prove|proof|matrix|"
    r"probability|calculus|algebra|factorial|modulo|summation)\b|"
    r"[0-9]\s*[+\-*/^]\s*[0-9]|\\\\frac|\\\\sum|\\\\int",
    re.I,
)

TRANSLATE_RE = re.compile(
    r"\b(translate|translation|traduci|traduis|traduce|"
    r"in (french|spanish|german|arabic|chinese|japanese|"
    r"portuguese|italian|russian|korean))\b",
    re.I,
)

FACTUAL_RE = re.compile(
    r"\b(who|what|when|where|which|capital of|how many|"
    r"define|definition of|meaning of)\b",
    re.I,
)

FRESH_RE = re.compile(
    r"\b(today|todays|latest|current|currently|right now|"
    r"this (week|month|year)|breaking|news|recent|as of|202[6-9]|live)\b",
    re.I,
)

HARD_RE = re.compile(
    r"\b(prove|derive|design|architect|optimi[sz]e|refactor|"
    r"analy[sz]e|explain why|step by step|trade-?offs?|"
    r"complex|end-to-end|production-grade|edge cases?|formal)\b",
    re.I,
)

MEDIUM_RE = re.compile(
    r"\b(explain|solve|implement|compute|describe|compare|"
    r"difference between|how (do|does|to)|write a|walk through|"
    r"step|fix|debug|error|stack.?trace|bug|review)\b",
    re.I,
)

SENSITIVE_RE = re.compile(
    r"\b(password|passwd|secret|api[_-]?key|private key|"
    r"ssn|credit card)\b|sk-[a-zA-Z0-9]{16,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)

CREATIVE_RE = re.compile(
    r"\b(brainstorm|creative|idea|suggest|imagine|"
    r"write a story|poem|essay|design|art|invent)\b",
    re.I,
)


def count_matches(regex: re.Pattern, text: str) -> int:
    """Conta le occorrenze di un pattern nel testo."""
    return len(regex.findall(text))


def classify(messages: list[dict]) -> TaskSignature:
    """Classifica un messaggio utente e restituisce un TaskSignature."""
    # Prende l'ultimo messaggio utente
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return TaskSignature(task="general", difficulty=0.3, confidence=0.3)

    text = user_msgs[-1]["content"]
    if isinstance(text, list):
        # Messaggio multimodale: prende solo la parte testuale
        parts = [p for p in text if isinstance(p, dict) and p.get("type") == "text"]
        text = " ".join(p["text"] for p in parts) if parts else ""
    else:
        text = str(text)

    lower = text.lower()
    chars = len(text)

    # ── Rilevamento task type (con priorità) ──
    is_code = bool(CODE_RE.search(text))
    is_math = bool(MATH_RE.search(text))
    is_translate = bool(TRANSLATE_RE.search(text))
    is_factual = bool(FACTUAL_RE.search(lower))
    is_creative = bool(CREATIVE_RE.search(lower))
    is_fresh = bool(FRESH_RE.search(lower))
    is_sensitive = bool(SENSITIVE_RE.search(text))

    # Assegnazione task (priorità decrescente)
    if is_translate:
        task = "translation"
    elif is_code:
        task = "code"
    elif is_math:
        task = "math"
    elif is_factual and chars < 400:
        task = "factual"
    elif is_creative and not is_code and not is_math:
        task = "creative"
    else:
        task = "general"

    # ── Calcolo difficoltà ──
    d = 0.22  # base
    if is_code:
        d += 0.22
    if is_math:
        d += 0.28
    if is_creative:
        d += 0.15

    hard_count = count_matches(HARD_RE, text)
    medium_count = count_matches(MEDIUM_RE, text)

    d += min(0.5, hard_count * 0.20)
    d += min(0.28, medium_count * 0.12)

    # Bonus per multi-domanda / multi-richiesta
    multi = min(1.0, (count_matches(re.compile(r"\?"), text) +
                      count_matches(re.compile(r"\b(and|also|then)\b", re.I), lower)) / 4)
    d += 0.20 * multi

    # Bonus per lunghezza
    d += 0.15 * min(1.0, chars / 1200)

    difficulty = max(0.0, min(1.0, d))

    # Calcolo confidenza
    # Match forte: pattern specifico trovato
    if task == "code" and ("```" in text or "def " in text or "class " in text or "function" in text):
        confidence = 0.90
    elif task == "code" and hard_count > 0:
        confidence = 0.75
    elif task == "translation":
        confidence = 0.85
    elif task == "math" and is_math:
        confidence = 0.85
    elif task == "factual":
        confidence = 0.80
    elif hard_count > 2:
        confidence = 0.85
    elif medium_count > 2:
        confidence = 0.70
    elif hard_count > 0:
        confidence = 0.65
    elif medium_count > 0:
        confidence = 0.55
    else:
        confidence = 0.40  # ambiguo

    # Capacità
    caps = []
    if any(m.get("type") == "image_url" for m in user_msgs[-1].get("content", [])
           if isinstance(m, dict)):
        caps.append("multimodal")

    return TaskSignature(
        task=task,
        difficulty=difficulty,
        confidence=confidence,
        caps=caps,
        sensitive=is_sensitive,
        freshness=is_fresh,
    )
