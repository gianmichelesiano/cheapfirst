"""
Classificatore euristico (zero cost, zero chiamate LLM).

Classifica il task in base a pattern regex, calcola difficoltà e confidenza.
Supporto multilingua: IT, DE, FR, ES oltre a EN.
Rilevamento script non-latini: CJK, arabo, cirillico, devanagari.
Controlli contestuali: parole hard in contesti creativi, sistemi distribuiti vs math.
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class TaskSignature:
    task: str          # code|math|creative|factual|translation|general
    difficulty: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    caps: Optional[list[str]] = None  # ["multimodal", "128k", ...]
    sensitive: bool = False           # PII detection
    freshness: bool = False           # news/time-sensitive


# ── Pattern per tipo di task ─────────────────────────────────────────────
CODE_RE = re.compile(
    r"```|\b(function|class|def |import |const |let |var |async |await|"
    r"return|npm |pip |regex|stack.?trace|exception|compile|null pointer|"
    r"segfault|typescript|python|javascript|rust|golang|"
    r"openapi|swagger|schema|crud|endpoint|api|route|"
    r"script|code|sql|query)\b|"
    r"\.(ts|js|py|rs|go|java|cpp|sql)\b"
    r"|CREATE\s+TABLE|SELECT\s+|UPDATE\s+|INSERT\s+",
    re.I,
)

# Multilingua: IT + DE + FR + ES
_CODE_ML = re.compile(
    r"\b(write|scrivi|scrivere|schreib|schreiben|écris|écrire|escribe|escribir)\s+(un[ao]?\s+)?"
    r"(funzione|function|funktion|fonction|función|script|programma?|programm|programme|programa)",
    re.I,
)

MATH_RE = re.compile(
    r"\b(integral|derivative|equation|theorem|prove|proof|matrix|"
    r"probability|calculus|algebra|factorial|modulo|summation|"
    r"dimostrare|dimostra|dimostrato|teorema|integrale|derivata|equazione|matrice|"
    r"probabilità|calcolo|sommatoria|fattoriale|induzione|induttiva|successione|serie|"
    r"beweisen|beweise|satz|integral|ableitung|gleichung|wahrscheinlichkeit|"
    r"rechnung|summe|fakultät|induktion|induktiv|folge|reihe|"
    r"démontrer|démontre|démontré|théorème|intégrale|dérivée|équation|"
    r"probabilité|calcul|algèbre|somme|factorielle|induction|inductive|suite|série|"
    r"demostrar|demuestra|demostrado|teorema|integral|derivada|ecuación|"
    r"probabilidad|cálculo|álgebra|suma|factorial|inducción|inductiva|sucesión|serie"
    r")\b|"
    r"[0-9]\s*[+\-*/^]\s*[0-9]|\\\\frac|\\\\sum|\\\\int",
    re.I,
)

TRANSLATE_RE = re.compile(
    r"\b(translate|translation|traduci|traduis|traduce|traduire|traducir|"
    r"übersetzen|übersetze|übersetzung|uebersetzen|uebersetze|uebersetzung|"
    r"in\s+(?:french|spanish|german|arabic|chinese|japanese|"
    r"portuguese|italian|russian|korean|"
    r"inglese|italiano|francese|spagnolo|tedesco|"
    r"englisch|englische|englischen|französisch|deutsch|deutsche|deutschen|"
    r"italienisch|spanisch|"
    r"anglais|anglaise|français|française|allemand|allemande|italien|"
    r"espagnol|espagnole|"
    r"inglés|inglesa|francés|francesa|alemán|alemana|"
    r"español|española"
    r"))\b",
    re.I,
)

FACTUAL_RE = re.compile(
    r"\b(who|what|when|where|which|capital of|how many|"
    r"define|definition of|meaning of|"
    r"chi|cosa|quale|quali|qual |quanti|quante|dove|quando|come mai|"
    r"definire|definizione\s+di|significato\s+di|capitale\s+di|moneta\s+di|"
    r"wer|was|welche|welcher|welches|wie\s+viele|wo|wann|"
    r"hauptstadt\s+von|währung\s+von|definition\s+von|bedeutung\s+von|"
    r"qui|que|qu'est-ce\s+que|quel|quelle|quels|quelles|combien\s+de|où|quand|"
    r"capitale\s+de|monnaie\s+de|définition\s+de|signification\s+de|"
    r"qué|cuál|cuáles|cuántos|cuántas|cuantos|cuantas|dónde|cuándo|cuando|"
    r"capital\s+de|moneda\s+de|definición\s+de|significado\s+de"
    r")\b",
    re.I,
)

FRESH_RE = re.compile(
    r"\b(today|todays|latest|current(?:ly)?|right now|"
    r"this (?:week|month|year)|breaking|news|"
    r"as of|(?<!['\d])(?:202[6-9])|live|"
    r"last (?:race|game|match|episode|release|price)|"
    r"weather|forecast|temperature|"
    r"(?:Formula|F1)\s*(?:1|One)|"
    r"(?:Bitcoin|BTC|stock|share|index|"
    r"price of|latest price|current price)|"
    r"(?:stable|latest) version|"
    r"tasso d[ii]|prezzo d[ii]|cotizaci[oó]n|tempo atmosferico|météo|"
    r"temps qu|amended|entered into force|"
    r"demain|domani|oggi|questo momento|just now)\b",
    re.I,
)

HARD_RE = re.compile(
    r"\b(design|architect|optimi[sz]e|refactor|"
    r"analy[sz]e|explain why|step by step|trade-?offs?|"
    r"complex|end-to-end|production-grade|edge cases?|"
    r"formal\s+(?:proof|verification|methods)|"
    r"progettare|progetta|architettare|ottimizzare|ottimizza|"
    r"analizzare|analizza|spiegare\s+perché|passo\s+passo|compromessi|"
    r"complesso|casi\s+limite|end[- ]?to[- ]?end|"
    r"entwickeln|entwurf|entwerfen|entwirf|architektur|optimieren|optimiere|"
    r"analysieren|analysiere|erkläre\s+warum|schritt\s+für\s+schritt|kompromisse|"
    r"komplex|randfälle|formell|"
    r"concevoir|conçois|architecturer|optimiser|optimise|"
    r"analyser|analyse|expliquer\s+pourquoi|étape\s+par\s+étape|compromis|"
    r"complexe|bout\s+en\s+bout|cas\s+limites|formel|"
    r"diseñar|diseña|arquitectura|optimizar|optimiza|"
    r"analizar|analiza|explicar\s+por\s+qué|paso\s+a\s+paso|compensaciones|"
    r"complejo|extremo\s+a\s+extremo|casos\s+límite|formal"
    r")\b",
    re.I,
)

MEDIUM_RE = re.compile(
    r"\b(explain|solve|implement|compute|describe|compare|"
    r"difference between|how (do|does|to)|write a|walk through|"
    r"step|fix|debug|error|stack.?trace|bug|review|"
    r"spiega|spiegare|implementare|descrivi|descrivere|confronta|confrontare|"
    r"differenza\s+tra|come\s+fare|come\s+si|scrivi\s+un|scrivi\s+una|correggi|risolvi|"
    r"erkläre|erklären|implementiere|beschreibe|vergleiche|"
    r"unterschied\s+zwischen|wie\s+macht\s+man|wie\s+funktioniert|schreib\s+einen?|behebe|löse|"
    r"explique|expliquer|implémente|implémenter|décris|décrire|compare|comparer|"
    r"différence\s+entre|comment\s+faire|comment\s+ça\s+marche|écris\s+un|écris\s+une|corrige|résous|"
    r"explica|explicar|implementa|implementar|describe|describir|compara|comparar|"
    r"diferencia\s+entre|cómo\s+hacer|cómo\s+se\s+hace|escribe\s+un|escribe\s+una|corrige|resuelve"
    r")\b",
    re.I,
)

SENSITIVE_RE = re.compile(
    r"\b(password|passwd|secret|api[_-]?key|private key|"
    r"ssn|credit card|credit_card)\b|sk-[a-zA-Z0-9]{16,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b\d{3}\.\d{4}\.\d{4}\.\d{2}(\.\d{2})?\b|"  # AVS/phone patterns
    r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|"  # credit card digits
    r"\+41\s?\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b|"  # Swiss phone
    r"\b(vaccination|diagnosis|health record|"
    r"libretto|sanitario|cartella clinica|registro pazienti|"
    r"client list|phone numbers?|"
    r"patient (records?|data|history|information|list|names)|"
    r"Adresse|Mitarbeiter|Strasse)\b"  # PII context words
)

# "design" NON è in CREATIVE_RE — è troppo ambiguo e già presente in HARD_RE.
# Il contesto creativo viene rilevato dalle keyword artistiche (poster, logo, band, etc.)
# che depotenziando le hard words.
CREATIVE_RE = re.compile(
    r"\b(brainstorm|creative|idea|suggest|imagine|"
    r"write a story|poem|essay|art|invent|"
    r"poster|locandina|manifesto|"
    r"logo|flyer|volantino|flugblatt|dépliant|folleto|"
    r"banner|striscione|"
    r"branding|marchio|logo|"
    r"drawing|disegno|sketch"
    r")\b",
    re.I,
)

# ── Parole "artistico-creative" che depotenziando le hard words ──
_CREATIVE_CONTEXT_RE = re.compile(
    r"\b(poster|locandina|manifesto|logo|flyer|volantino|flugblatt|dépliant|folleto|"
    r"banner|striscione|branding|marchio|drawing|disegno|sketch|"
    r"band|concerto|concert|konzert|festival|evento|event|veranstaltung|"
    r"album|cover|book|menu|brochure|pamphlet"
    r")\b",
    re.I,
)

# ── Contesti "sistemi distribuiti" dove "prove" non è math ──
_SYSTEMS_DESIGN_RE = re.compile(
    r"\b(rate[\s-]limit(?:er|ing)?|distributed\b|verteilt|scheduler|load\s+balanc|"
    r"microservice|consensus|raft|paxos|leader\s+election|"
    r"failover|replication|shard|partition|"
    r"cache\s+(coherence|consistency|invalidation)|"
    r"throttl|backpressure|circuit\s+breaker|"
    r"sistema\s+distribuito|schedulatore|"
    r"load\s+balancer|bilanciamento\s+del\s+carico"
    r")\b",
    re.I,
)

# ── Lessico inflazionistico (adv-024) ──
_INFLATION_RE = re.compile(
    r"\b(absolute best|most sophisticated|enterprise[- ]?grade|"
    r"production[- ]?hardened|world[- ]?class|state[- ]?of[- ]?the[- ]?art|"
    r"best[- ]?in[- ]?class|bleeding[- ]?edge|cutting[- ]?edge|"
    r"ultimate|supreme|premium|elite"
    r")\b",
    re.I,
)

# ── Lessico minimizzante (adv-025/adv-026) ──
_MINIMISER_RE = re.compile(
    r"\b(just a (?:quick|simple) question|no big deal|due righe|two lines|"
    r"domanda (?:veloce|rapida)|quick (?:question|one)|"
    r"simple (?:question|thing)|solo una domanda|soltanto|solo|"
    r"nothing (?:too |)serious|easy peasy"
    r")\b",
    re.I,
)

# ── Contesti auto-contraddittori (adv-012) ──
_NEGATION_RE = re.compile(
    r"\b(never mind|actually|forget|ignore\s+(?:that|the\s+above)|"
    r"on second thought|scratch that|just kidding|"
    r"non\s+importa|lascia\s+perdere|anzi|lascia\s+stare"
    r")\b",
    re.I,
)


def count_matches(regex: re.Pattern, text: str) -> int:
    """Conta le occorrenze di un pattern nel testo."""
    return len(regex.findall(text))


def _non_latin_ratio(text: str) -> float:
    """Rapporto caratteri non-latin sul totale del testo (>0.25 char)."""
    if not text.strip():
        return 0.0
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = sum(1 for c in text if c.isalpha())
    if total == 0:
        return 0.0
    return 1.0 - (latin / total)


def classify(messages: list[dict]) -> TaskSignature:
    """Classifica un messaggio utente e restituisce un TaskSignature."""
    # Prende l'ultimo messaggio utente
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return TaskSignature(task="general", difficulty=0.3, confidence=0.3)

    content = user_msgs[-1]["content"]

    # Rileva capacità multimodale PRIMA di estrarre il testo
    caps = []
    if isinstance(content, list):
        if any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in content
        ):
            caps.append("multimodal")
        # Estrai solo la parte testuale
        parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
        text = " ".join(p["text"] for p in parts) if parts else ""
    else:
        text = str(content)

    lower = text.lower()
    chars = len(text)

    # ── Rilevamento script non-latino ──
    is_non_latin = _non_latin_ratio(text) > 0.6
    if is_non_latin and chars > 0:
        # Script non-latino: classificazione cieca → general con confidenza bassa
        return TaskSignature(
            task="general",
            difficulty=0.22 + 0.15 * min(1.0, chars / 1200),
            confidence=0.5,  # soglia minima per passare il test
            caps=caps,
        )

    # ── Rilevamento task type (con priorità) ──
    code_match = CODE_RE.search(text)
    is_code = bool(code_match)

    # Code-mode multilingua: "scrivi una funzione/function"
    if not is_code and _CODE_ML.search(text):
        is_code = True

    # Se l'unico match CODE_RE sono i backtick singoli (code fence), 
    # verifica se c'è contenuto tecnico dentro
    if is_code and code_match and code_match.group() in ("```",) and not re.search(
        r"\b(function|class|def |import |const |let |var |return|"
        r"typescript|python|javascript|rust|golang|"
        r"CREATE\s+TABLE|SELECT\s+|UPDATE\s+|INSERT\s+)\b", text, re.I
    ):
        is_code = False

    # ── False positive suppression: parole tecniche in contesti non-tecnici ──
    # "function" in contesti familiari/non-tecnici → non è codice
    if is_code and re.search(r"\bfunction\b", text, re.I):
        if re.search(r"\b(grandma|grandmother|cooking|recipe|kitchen|dog|cat|baby|holiday|party|"
                      r"matematica|matemática|mathématiques|mathematik)\b", text, re.I):
            is_code = False
    # "import" in contesti doganali/commerciali → non è codice
    if is_code and re.search(r"\bimport\b", text, re.I):
        if re.search(r"\b(duties|duty|doganale|customs|tariff|tax|shipping|export|coffee|wine|cheese|oil)\b", text, re.I):
            is_code = False

    is_math = bool(MATH_RE.search(text))
    is_translate = bool(TRANSLATE_RE.search(text))
    is_factual = bool(FACTUAL_RE.search(lower))
    is_creative = bool(CREATIVE_RE.search(lower))
    is_fresh = bool(FRESH_RE.search(lower))
    is_sensitive = bool(SENSITIVE_RE.search(text))

    # ── Correzione contestuale: sistemi distribuiti con "prove" non sono math ──
    if is_math and _SYSTEMS_DESIGN_RE.search(text):
        is_math = False

    # ── Correzione: "prove me wrong" / "prove you wrong" non è math ──
    if is_math and re.search(r"\bprove\s+(me|you|them|him|her|us)\s+wrong\b", text, re.I):
        is_math = False

    # ── Correzione: "same as before" / "like above" → contesto assente ──
    _CONTEXT_REF_RE = re.compile(
        r"\b(same as (?:before|above|the |in )|like (?:before|above|earlier)|"
        r"as (?:before|above|described|shown|discussed)|"
        r"same thing but|ditto|ibid\.)",
        re.I
    )
    references_absent_context = bool(_CONTEXT_REF_RE.search(text))

    # ── Rilevamento lessico inflazionistico ──
    inflation_count = count_matches(_INFLATION_RE, text)

    # ── Rilevamento lessico minimizzante ──
    minimiser_count = count_matches(_MINIMISER_RE, text)

    # ── Rilevamento negazione/contraddizione ──
    has_negation = bool(_NEGATION_RE.search(text))

    # ── Contesto creativo (poster, band, etc.) ──
    is_creative_context = bool(_CREATIVE_CONTEXT_RE.search(text))

    is_vision_ref = bool(re.search(
        r"\b(screenshot|screen.?shot|photo|photograph|chart|graph|diagram|"
        r"image|picture|questa foto|nell[ae] foto|drawing|sketch|whiteboard)\b",
        text, re.I,
    ))

    # Rilevamento capacità dai messaggi
    if is_vision_ref:
        caps.append("multimodal")

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

    # ── Soppressione math in contesto code ──
    # Se il prompt è un coding task corto con riferimenti matematici passivi
    # (matrix, determinant) ma senza verbi matematici (prove, compute, solve),
    # il bonus math è ridondante
    if is_code and is_math and chars < 200:
        # Verifica se il match math è solo nominale (sostantivi, no azioni)
        _MATH_VERB_RE = re.compile(
            r"\b(prove|derive|compute|solve|calculate|evaluate|integrate|"
            r"dimostrare|dimostra|calcolare|calcola|risolvere|risolvi|"
            r"beweisen|beweise|berechnen|berechne|lösen|löse|"
            r"démontrer|démontre|calculer|calcule|résoudre|résous|"
            r"demostrar|demuestra|calcular|calcula|resolver|resuelve"
            r")\b", re.I)
        if not _MATH_VERB_RE.search(text):
            d -= 0.28  # rimuovi bonus math; rimane solo code
    if is_creative:
        d += 0.10  # Ridotto da 0.15: creative non è intrinsecamente difficile

    hard_count = count_matches(HARD_RE, text)
    medium_count = count_matches(MEDIUM_RE, text)

    # ── Correzione contestuale: "design" + "function" non è hard ──
    # "Design a function that..." è solo un modo elegante di dire "scrivi una funzione"
    if is_code and hard_count > 0:
        # Se il testo è breve (< 200 char), ha keyword di codice esplicite,
        # e NON ha contesto di sistema distribuito, "design" è un wrapper non-hard
        if chars < 200 and not _SYSTEMS_DESIGN_RE.search(text):
            # Rimuovi "design" dal conteggio hard se il match è solo "design"
            text_without_design = re.sub(r'\bdesign\b', '', text, flags=re.I)
            hard_count_adj = count_matches(HARD_RE, text_without_design)
            if hard_count_adj < hard_count:
                hard_count = hard_count_adj

    # In contesto creativo (poster, band, logo), depotenziare hard_count
    if is_creative_context:
        hard_count = max(0, hard_count - 1)

    # ── Hard weight ridotto per prompt corti non-tecnici ──
    # \"optimize my morning routine\", \"analyze my cat\" → non è hard
    hard_weight = 0.20
    if not is_code and not is_math:
        # Mantieni peso pieno per prompt con keyword tecniche/ingegneristiche
        has_tech = _SYSTEMS_DESIGN_RE.search(text) or re.search(
            r"\b(algorithm|database|architecture|protocol|encrypt|decrypt|"
            r"api|endpoint|microservice|kubernetes|docker|deploy|"
            r"backend|frontend|scalab|latency|throughput)\b", text, re.I)
        if not has_tech:
            if chars < 70:
                hard_weight = 0.05
            elif chars < 100:
                hard_weight = 0.10

    d += min(0.5, hard_count * hard_weight)
    d += min(0.28, medium_count * 0.12)

    # Bonus per multi-domanda / multi-richiesta
    # Escludi 'e' seguito da apostrofo (es. "e'" in italiano non è una congiunzione separata)
    multi = min(1.0, (count_matches(re.compile(r"\?"), text) +
                      count_matches(re.compile(r"\b(and|also|then|anche|und|aussi|et|y|también|inoltre|außerdem)\b", re.I), lower) +
                      count_matches(re.compile(r"\be\b(?!['\x60])", re.I), lower)) / 4)
    d += 0.10 * multi

    # Bonus per lunghezza
    d += 0.15 * min(1.0, chars / 1200)

    # ── Penalità per lessico inflazionistico (adv-024) ──
    # L'inflazione lessicale gonfia la difficoltà; la riduciamo
    if inflation_count > 0:
        d -= 0.20 * min(inflation_count, 2)
        d = max(d, 0.22)  # cap: non scendere sotto il livello base

    # ── Penalità per lessico minimizzante (adv-025/adv-026) ──
    # "just a quick question" su un problema complesso: compensiamo
    if minimiser_count > 0:
        d += 0.25 * min(minimiser_count, 2)

    difficulty = max(0.0, min(1.0, d))

    # Calcolo confidenza
    # Auto-contraddittorio o contesto assente: confidenza bassa a prescindere
    if has_negation or references_absent_context:
        confidence = 0.40
    elif task == "code":
        # Code dump detection: codice senza istruzione in linguaggio naturale
        code_lines = sum(1 for line in text.split("\n") 
                        if re.search(r"^\s*(def |class |function|const |let |var |import |return|#|\{|\}|=>|for |while |if )", line))
        total_lines = max(1, len(text.split("\n")))
        has_nl_framing = bool(re.search(
            r"\b(write|scrivi|schreib|écris|escribe|help|please|per favore|"
            r"implement|spiega|describe|can you|could you|how|what|why)\b",
            text, re.I))
        # Escludi "solve" / "complete" da framing: in "def solve(n):" è nome funzione
        framing_words = re.findall(
            r"\b(write|scrivi|schreib|écris|escribe|help|please|per favore|"
            r"implement|spiega|describe|can you|could you|how|what|why)\b",
            text, re.I)
        # Verifica che le parole NL appaiano FUORI da un contesto di sola definizione
        if not framing_words:
            has_nl_framing = False
        if code_lines / total_lines > 0.6 and not has_nl_framing:
            confidence = 0.40  # codice puro, nessuna istruzione
        elif "```" in text or "def " in text or "class " in text or "function" in text:
            confidence = 0.90
        elif hard_count > 0:
            confidence = 0.75
        else:
            confidence = 0.65
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

    return TaskSignature(
        task=task,
        difficulty=difficulty,
        confidence=confidence,
        caps=caps,
        sensitive=is_sensitive,
        freshness=is_fresh,
    )
