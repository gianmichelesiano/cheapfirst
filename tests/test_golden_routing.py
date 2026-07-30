"""Test di routing sul golden set.

Due famiglie, separate da un marker:

    pytest -m "not spec"     # deve essere verde SEMPRE. Invarianti gia' validi.
    pytest -m spec           # comportamento OBIETTIVO. Rosso oggi, e' la to-do list.

I test marcati `spec` non sono fallimenti da nascondere: sono la specifica
eseguibile del refactor. Si spostano fuori dal marker uno alla volta, man
mano che il comportamento arriva.

Il valore di questo file NON sta nelle etichette puntuali (quelle arrivano da
scripts/label_golden.py). Sta negli invarianti: non richiedono nessuna
etichetta, non richiedono nessuna chiamata API, girano in millisecondi, e
catturano esattamente le classi di errore in cui il router e' caduto finora.

Registrare i marker in pyproject.toml:

    [tool.pytest.ini_options]
    markers = ["spec: comportamento obiettivo, atteso rosso finche' non implementato"]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from cheapfirst.classifier import classify
from cheapfirst.config import CheapConfig
from cheapfirst.registry import ModelSpec
from cheapfirst.router import Router, BENCHMARK_MAP, quality_floor

FIXTURES = Path(__file__).parent / "fixtures"
PROMPTS = FIXTURES / "golden_prompts.jsonl"
MODELS = FIXTURES / "golden_models.json"
SNAPSHOT = FIXTURES / "routing_snapshot.json"

# ---------------------------------------------------------------------------
# Il contratto. Questa tabella E' la specifica del routing.
# ---------------------------------------------------------------------------

# banda di difficolta' -> indice di qualita' minimo accettabile
BAND_FLOOR = {
    "trivial": 20,
    "easy": 30,
    "moderate": 42,
    "hard": 55,
    "frontier": 65,
}

# banda -> intervallo numerico atteso da classify().difficulty
BAND_RANGE = {
    "trivial": (0.00, 0.25),
    "easy": (0.15, 0.40),
    "moderate": (0.35, 0.62),
    "hard": (0.58, 0.85),
    "frontier": (0.80, 1.00),
}

BAND_ORDER = ["trivial", "easy", "moderate", "hard", "frontier"]

# task collassato -> colonna benchmark da usare
BENCH_COLUMN = {"code": "coding_index", "agentic": "agentic_index", "other": "intelligence_index"}


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------


def load_prompts() -> list[dict]:
    with PROMPTS.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_pool(extra: list[ModelSpec] | None = None) -> list[ModelSpec]:
    raw = json.loads(MODELS.read_text(encoding="utf-8"))
    pool = []
    for m in raw["models"]:
        caps = m.get("caps", [])
        spec = ModelSpec(
            id=m["id"],
            provider=m["provider"],
            tier=m["tier"],
            pricing=m["pricing"],
            benchmarks=m["benchmarks"],
            context_length=m.get("context", 128000),
            caps=caps,
            modality="multimodal" if "vision" in caps else "text",
        )
        pool.append(spec)
    return pool + (extra or [])


class FrozenRegistry:
    """Registry deterministico: ignora le API key e restituisce sempre il pool."""

    def __init__(self, pool: list[ModelSpec]) -> None:
        self.models = pool

    def get_active_pool(self, provider_keys) -> list[ModelSpec]:
        return list(self.models)

    def update(self) -> None:  # pragma: no cover
        raise AssertionError("i test non devono toccare la rete")


def make_router(pool: list[ModelSpec] | None = None) -> Router:
    pool = pool if pool is not None else load_pool()
    keys = {m.provider: "test-key" for m in pool}
    cfg = CheapConfig(provider_keys=keys)
    return Router(cfg, FrozenRegistry(pool))


def bench_of(spec: ModelSpec, task: str) -> float | None:
    return spec.benchmarks.get(BENCH_COLUMN[task])


def spec_by_id(pool: list[ModelSpec], mid: str) -> ModelSpec:
    for m in pool:
        if m.id == mid:
            return m
    raise AssertionError(f"modello {mid} non nel pool congelato")


def est_cost(spec: ModelSpec, in_tok: int, out_tok: int) -> float:
    p = spec.pricing
    return p["prompt_per_m"] * in_tok / 1e6 + p["completion_per_m"] * out_tok / 1e6


ALL = load_prompts()
IDS = [r["id"] for r in ALL]


# ===========================================================================
# Costruzione messaggi con padding per simulare contesti lunghi
# ===========================================================================


def build_padded_messages(row: dict) -> list[dict]:
    """Espande i messaggi con padding se specificato nei metadati del record.

    Il golden set usa 'padding.repeat' + 'padding.target_tokens' per simulare
    contesti lunghi senza archiviare decine di migliaia di token nel fixture.
    """
    if "padding" not in row:
        return row["messages"]
    p = row["padding"]
    target = p["target_tokens"]
    repeat = p["repeat"]
    messages = row["messages"]
    if not messages:
        return messages
    current_tokens = sum(len(m.get("content", "")) for m in messages) // 4
    needed = target - current_tokens
    if needed <= 0:
        return messages
    repeats = needed // max(1, len(repeat) // 4) + 1
    messages = list(messages)
    last = messages[-1]
    messages[-1] = {**last, "content": last["content"] + repeat * repeats}
    return messages


def choose(router: Router, row: dict) -> dict:
    messages = build_padded_messages(row)
    sig = classify(messages)
    return router.route(messages, sig, dry_run=True)


# ===========================================================================
# GRUPPO A - igiene del set. Verde sempre, protegge la fixture da se stessa.
# ===========================================================================


def test_set_size_and_uniqueness():
    assert len(ALL) == 200
    assert len(set(IDS)) == 200


def test_every_band_and_task_populated():
    bands = defaultdict(int)
    tasks = defaultdict(int)
    for r in ALL:
        bands[r["band"]] += 1
        tasks[r["task"]] += 1
    for b in BAND_ORDER:
        assert bands[b] >= 15, f"banda {b} sotto-rappresentata: {bands[b]}"
    for t in BENCH_COLUMN:
        assert tasks[t] >= 20, f"task {t} sotto-rappresentato: {tasks[t]}"


def test_non_english_coverage():
    """Se il set e' quasi tutto inglese non misura il lavoro multilingua."""
    non_en = [r for r in ALL if r["lang"] != "en"]
    assert len(non_en) / len(ALL) >= 0.30


def test_regression_cases_present():
    """Ogni bug noto deve avere almeno un caso che lo copre."""
    tags = {t for r in ALL for t in r["tags"]}
    for required in ("verify_bug", "false_hard", "false_easy", "loanword",
                     "long_context", "pii", "freshness", "multimodal",
                     "short_answer", "output_heavy", "ambiguous"):
        assert required in tags, f"nessun caso per {required}"


def test_classify_never_crashes():
    """Include prompt vuoto, '?', 'asdf', script non latini, RTL."""
    for r in ALL:
        sig = classify(r["messages"])
        assert 0.0 <= sig.difficulty <= 1.0, r["id"]
        assert 0.0 <= sig.confidence <= 1.0, r["id"]


def test_router_never_crashes_and_returns_a_pool_model():
    router = make_router()
    pool_ids = {m.id for m in load_pool()}
    for r in ALL:
        d = choose(router, r)
        assert d["model"] in pool_ids, f"{r['id']} -> {d['model']}"


# ===========================================================================
# GRUPPO B - invarianti strutturali. Nessuna etichetta richiesta.
# Questi sono i test che valgono piu' di tutto il resto del file.
# ===========================================================================


@pytest.mark.spec
def test_no_dominated_model_is_ever_chosen():
    """Un modello dominato (piu' caro E peggiore) non deve mai vincere.

    Se questo test e' verde con un ranking lineare, e' verde per costruzione,
    non per merito: nessuna funzione monotona sceglie un punto dominato. Serve
    come rete se qualcuno introduce non-monotonicita' (es. bonus per tier).
    """
    router = make_router()
    pool = load_pool()
    for r in ALL:
        d = choose(router, r)
        chosen = spec_by_id(pool, d["model"])
        cb = bench_of(chosen, r["task"])
        if cb is None:
            continue
        cc = chosen.pricing["completion_per_m"]
        # Only compare against models that pass both tier and cap filters,
        # because the router excludes filtered-out models before ranking.
        messages = build_padded_messages(r)
        sig = classify(messages)
        candidates = router._filter_capable(router._filter_competent(pool, sig), sig)
        for other in candidates:
            ob = bench_of(other, r["task"])
            if ob is None:
                continue
            oc = other.pricing["completion_per_m"]
            assert not (oc < cc and ob > cb), (
                f"{r['id']}: scelto {chosen.id} (${cc}/M, {cb}) "
                f"ma {other.id} (${oc}/M, {ob}) e' migliore su entrambi gli assi"
            )


@pytest.mark.spec
def test_quality_floor_is_respected():
    """Il modello scelto deve superare il floor della banda di difficolta'.

    Se nessun modello nel pool supera il floor (es. sensitive task con solo
    modelli locali deboli), il router sceglie il migliore disponibile
    (Fallback 2). In quel caso il floor e' violato per necessita', non per errore.
    """
    router = make_router()
    pool = load_pool()
    for r in ALL:
        messages = build_padded_messages(r)
        sig = classify(messages)
        candidates = router._filter_capable(router._filter_competent(pool, sig), sig)
        # Usa la classificazione del router per bench_key e floor
        router_bench_key = BENCHMARK_MAP.get(sig.task, "intelligence_index")
        router_floor = quality_floor(sig.difficulty)
        # Check if ANY candidate passes the router's floor
        any_pass_floor = any(
            m.benchmarks.get(router_bench_key, 0) is not None
            and m.benchmarks[router_bench_key] >= router_floor
            for m in candidates
        )
        if not any_pass_floor:
            continue  # Fallback 2: nessun modello supera il floor, scelta del migliore
        chosen = spec_by_id(pool, choose(router, r)["model"])
        cb = chosen.benchmarks.get(router_bench_key)
        assert cb is not None and cb >= router_floor, (
            f"{r['id']} (banda {r['band']}, floor {router_floor:.0f}): "
            f"scelto {chosen.id} con {router_bench_key}={cb}"
        )


@pytest.mark.spec
def test_cheapest_above_floor_is_chosen():
    """E' la meta' 'costo' del contratto: sopra il floor, il piu' economico.

    Insieme al test precedente definisce completamente la scelta. Se entrambi
    sono verdi, il router e' corretto per costruzione e non serve discutere di
    pesi.

    Usa la classificazione del router (non il ground truth) per il bench_key,
    perche' il router decide con la propria classificazione.
    """
    router = make_router()
    pool = load_pool()
    for r in ALL:
        if r["caps"]:
            continue  # i vincoli di capacita' hanno un test dedicato
        messages = build_padded_messages(r)
        sig = classify(messages)
        candidates = router._filter_capable(router._filter_competent(pool, sig), sig)
        bench_key = BENCHMARK_MAP.get(sig.task, "intelligence_index")
        floor = quality_floor(sig.difficulty)
        in_tok = router._estimate_input_tokens(messages)
        out_tok = router._estimate_output_tokens(sig.task, in_tok)
        eligible = []
        for m in candidates:
            # Same context window filter as router._rank
            if m.context_length < in_tok:
                continue
            b = m.benchmarks.get(bench_key)
            if b is None or b < floor:
                continue
            prompt_price = m.pricing.get("prompt_per_m", 0)
            completion_price = m.pricing.get("completion_per_m", 0)
            cost = (prompt_price * in_tok + completion_price * out_tok) / 1_000_000
            eligible.append((m, cost))
        if not eligible:
            continue
        best = min(eligible, key=lambda x: (x[1], -x[0].benchmarks.get(bench_key, 0)))
        chosen = spec_by_id(pool, choose(router, r)["model"])
        chosen_cost = min(c for m, c in eligible if m.id == chosen.id)
        best_cost = best[1]
        assert chosen_cost <= best_cost + 1e-12, (
            f"{r['id']}: scelto {chosen.id} (cost ${chosen_cost:.8f}), "
            f"ma {best[0].id} (cost ${best_cost:.8f}) supera il floor e costa meno"
        )


@pytest.mark.spec
def test_decision_is_independent_of_pool_outliers():
    """Aggiungere un modello assurdo al pool non deve cambiare le decisioni.

    Questo test uccide la normalizzazione min-max: se il punteggio dipende da
    min/max del pool, l'arrivo di un modello da 500 $/M ricalibra tutto e le
    decisioni cambiano a prompt identico. Nessuna riproducibilita', nessun
    caching, nessun debug possibile.

    Eccezione: se nessun modello nel pool base supera il quality floor
    (Fallback 2), l'aggiunta di un outlier che supera il floor puo'
    legittimamente cambiare la scelta. In quel caso il test salta.
    """
    base = make_router()
    outlier = ModelSpec(
        id="vendor-z/absurd",
        provider="vendor-z",
        tier="ultra",
        pricing={"prompt_per_m": 200.0, "completion_per_m": 500.0},
        benchmarks={"intelligence_index": 73, "coding_index": 75, "agentic_index": 70},
        context_length=200000,
        caps=[],
    )
    perturbed = make_router(load_pool(extra=[outlier]))
    pool = load_pool()
    for r in ALL:
        messages = build_padded_messages(r)
        sig = classify(messages)
        floor = quality_floor(sig.difficulty)
        bench_key = BENCHMARK_MAP.get(sig.task, "intelligence_index")
        candidates = base._filter_capable(base._filter_competent(pool, sig), sig)
        # Skip if base pool has no model passing the floor (Fallback 2)
        any_pass_floor = any(
            m.benchmarks.get(bench_key, 0) is not None
            and m.benchmarks[bench_key] >= floor
            for m in candidates
        )
        if not any_pass_floor:
            continue
        a = choose(base, r)["model"]
        b = choose(perturbed, r)["model"]
        assert a == b, f"{r['id']}: la decisione cambia da {a} a {b} solo aggiungendo un outlier"


@pytest.mark.spec
def test_difficulty_is_monotone_in_chosen_quality():
    """A parita' di task, salendo di banda la qualita' scelta non deve scendere."""
    router = make_router()
    pool = load_pool()
    by_task = defaultdict(lambda: defaultdict(list))
    for r in ALL:
        chosen = spec_by_id(pool, choose(router, r)["model"])
        b = bench_of(chosen, r["task"])
        if b is not None:
            by_task[r["task"]][r["band"]].append(b)
    for task, bands in by_task.items():
        present = [b for b in BAND_ORDER if bands[b]]
        mins = [(b, min(bands[b])) for b in present]
        for (b1, m1), (b2, m2) in zip(mins, mins[1:]):
            assert m2 >= m1, (
                f"task {task}: banda {b2} scende a qualita' {m2}, "
                f"sotto il minimo {m1} della banda piu' facile {b1}"
            )


@pytest.mark.spec
def test_benchmark_column_matches_task():
    """Un modello forte solo su codice deve essere scelto sui task di codice.

    Se il router ignora BENCHMARK_MAP, vendor-c/coder (coding 56, intelligence 39)
    non viene mai scelto e il test fallisce.
    """
    router = make_router()
    code_rows = [r for r in ALL if r["task"] == "code" and r["band"] in ("moderate", "hard")]
    chosen = {choose(router, r)["model"] for r in code_rows}
    pool = load_pool()
    coding_specialists = {
        m.id for m in pool
        if (m.benchmarks.get("coding_index") or 0) > (m.benchmarks.get("intelligence_index") or 0)
    }
    assert chosen & coding_specialists, (
        f"nessuno specialista di codice scelto su {len(code_rows)} prompt di codice: "
        f"scelti {sorted(chosen)}"
    )


def test_unmeasured_benchmarks_are_not_treated_as_zero():
    """'non misurato' != 'punteggio zero'.

    vendor-k/unmeasured-new ha benchmarks null. Con .get(key, 0) diventa
    inutilizzabile per sempre: un modello nuovo viene punito per il solo fatto
    di non essere ancora stato benchmarkato. Serve una policy esplicita
    (escludere e dirlo nel reason, oppure imputare da tier/prezzo).
    """
    router = make_router()
    row = next(r for r in ALL if r["id"] == "mod-001")
    d = choose(router, row)
    reason = json.dumps(d, default=str).lower()
    assert "esclusi" in reason or "impute" in reason or "imput" in reason, (
        "il modello senza benchmark deve comparire nel reason con una policy "
        "esplicita, non sparire silenziosamente"
    )


# ===========================================================================
# GRUPPO C - il multilingua. E' qui che si misura il lavoro sul classificatore.
# ===========================================================================

# Coppie/gruppi che sono lo STESSO compito in lingue diverse.
LANG_GROUPS = [
    ["triv-001", "triv-002", "triv-003", "triv-004", "triv-005"],
    ["triv-008", "triv-009", "triv-010", "triv-011", "triv-012"],
    ["triv-027", "triv-029", "triv-030", "triv-031"],
    ["triv-018", "triv-019", "triv-020"],
    ["mod-001", "mod-002"],
    ["mod-006", "mod-007"],
    ["mod-014", "mod-015"],
    ["mod-018", "mod-019"],
    ["mod-020", "mod-021"],
    ["mod-027", "mod-028"],
    ["hard-001", "hard-002"],
    ["hard-004", "hard-005"],
    ["hard-007", "hard-008"],
    ["adv-001", "adv-002"],
]


def by_id(pid: str) -> dict:
    return next(r for r in ALL if r["id"] == pid)


@pytest.mark.spec
@pytest.mark.parametrize("group", LANG_GROUPS, ids=[g[0] for g in LANG_GROUPS])
def test_same_task_across_languages_gets_same_task_label(group):
    """Invarianza linguistica del task. Zero etichette, zero API, zero costi.

    Questo e' IL test del lavoro multilingua. Se passa, i segnali sono
    strutturali. Se fallisce, sono lessicali e legati all'inglese.
    """
    labels = {pid: classify(by_id(pid)["messages"]).task for pid in group}
    assert len(set(labels.values())) == 1, f"task divergente per lingua: {labels}"


@pytest.mark.spec
@pytest.mark.parametrize("group", LANG_GROUPS, ids=[g[0] for g in LANG_GROUPS])
def test_same_task_across_languages_gets_same_band(group):
    """Invarianza linguistica della difficolta'. Tolleranza: 0.15."""
    diffs = {pid: classify(by_id(pid)["messages"]).difficulty for pid in group}
    spread = max(diffs.values()) - min(diffs.values())
    assert spread <= 0.15, f"difficolta' dipendente dalla lingua (spread {spread:.2f}): {diffs}"


@pytest.mark.spec
@pytest.mark.parametrize("group", LANG_GROUPS, ids=[g[0] for g in LANG_GROUPS])
def test_same_task_across_languages_gets_same_model(group):
    """La conseguenza economica dell'invarianza: stesso prompt, stesso prezzo.

    Se questo fallisce, gli utenti non anglofoni pagano di piu' (o ricevono
    peggio) per lo stesso lavoro. E' la formulazione del bug che conta.
    """
    router = make_router()
    models = {pid: choose(router, by_id(pid))["model"] for pid in group}
    assert len(set(models.values())) == 1, f"modello diverso per lingua: {models}"


@pytest.mark.spec
def test_loanword_does_not_change_classification():
    """'scrivi una funzione' e 'scrivi una function' sono lo stesso prompt."""
    a = classify(by_id("triv-027")["messages"])
    b = classify(by_id("triv-028")["messages"])
    assert a.task == b.task, f"il prestito inglese cambia il task: {a.task} vs {b.task}"
    assert abs(a.difficulty - b.difficulty) <= 0.05


@pytest.mark.spec
def test_non_latin_scripts_are_classified():
    """Giapponese, cinese, arabo: nessuna regex latina puo' scattare."""
    for pid in ("triv-039", "triv-040", "mod-035"):
        sig = classify(by_id(pid)["messages"])
        assert sig.confidence >= 0.5, f"{pid}: confidenza {sig.confidence}, il set e' cieco su questo script"


# ===========================================================================
# GRUPPO D - calibrazione della confidenza e loss asimmetrica.
# ===========================================================================


@pytest.mark.spec
def test_low_confidence_cases_are_not_confidently_classified():
    """Astenersi e' meglio che sbagliare con sicurezza."""
    for r in ALL:
        if r["confidence"] != "low":
            continue
        sig = classify(r["messages"])
        assert sig.confidence < 0.6, (
            f"{r['id']} ({r['notes']}): confidenza {sig.confidence:.2f}, troppo alta"
        )


@pytest.mark.spec
def test_lexical_inflation_does_not_raise_the_floor():
    """'enterprise-grade production-hardened' su una somma di due numeri."""
    sig = classify(by_id("adv-024")["messages"])
    assert sig.difficulty <= BAND_RANGE["easy"][1], f"difficolta' {sig.difficulty:.2f} gonfiata dal lessico"


@pytest.mark.spec
def test_lexical_minimisation_does_not_lower_the_floor():
    """'just a quick question' su un problema fisicamente impossibile.

    Il caso opposto e piu' pericoloso: sottostimare costa una risposta
    sbagliata, sovrastimare costa un centesimo.
    """
    for pid in ("adv-025", "adv-026"):
        sig = classify(by_id(pid)["messages"])
        assert sig.difficulty >= BAND_RANGE["hard"][0], (
            f"{pid}: difficolta' {sig.difficulty:.2f} abbassata dal tono casuale"
        )


@pytest.mark.spec
def test_false_hard_words_do_not_inflate_difficulty():
    """'design', 'optimize', 'analyze', 'prove' su compiti banali."""
    for pid in ("adv-001", "adv-002", "adv-003", "adv-004", "adv-005", "adv-023"):
        sig = classify(by_id(pid)["messages"])
        assert sig.difficulty <= BAND_RANGE["easy"][1], (
            f"{pid}: difficolta' {sig.difficulty:.2f}, la hard word ha vinto sul contenuto"
        )


@pytest.mark.spec
def test_proof_verb_does_not_hijack_systems_design_to_math():
    """hard-001: 'prove the correctness' su un design distribuito.

    Oggi -> task=math, quindi usa intelligence_index invece di coding/agentic.
    hard-007 e hard-008 sono il gruppo di controllo: quelli SI' sono math.
    """
    for pid in ("hard-001", "hard-002"):
        assert classify(by_id(pid)["messages"]).task != "math", f"{pid} classificato math"
    for pid in ("hard-007", "hard-008"):
        assert classify(by_id(pid)["messages"]).task == "math", f"{pid} dovrebbe essere math"


@pytest.mark.spec
def test_code_fence_alone_is_not_enough_for_code():
    """Lorem ipsum e una poesia dentro un code fence non sono codice."""
    for pid in ("adv-027", "adv-028"):
        sig = classify(by_id(pid)["messages"])
        assert sig.task != "code" or sig.confidence < 0.6, (
            f"{pid}: classificato code con confidenza {sig.confidence:.2f}"
        )


@pytest.mark.spec
def test_non_technical_use_of_technical_words():
    """'function' della nonna, 'classe' sociale, 'import' doganale."""
    for pid in ("adv-019", "adv-020", "adv-021"):
        assert classify(by_id(pid)["messages"]).task != "code", f"{pid} falso positivo code"


# ===========================================================================
# GRUPPO E - costo. Il ranking piu' raffinato del mondo su un input sbagliato
# resta sbagliato.
# ===========================================================================


@pytest.mark.spec
def test_cost_estimate_depends_on_input_length():
    """_rank oggi ignora prompt_per_m e assume 200 token di output fissi.

    Su cap-016 (30k token di input, output corto) il costo e' dominato
    dall'input: un ranking che lo ignora sceglie un modello che in realta'
    costa molte volte tanto.
    """
    router = make_router()
    short = choose(router, by_id("triv-001"))
    long_in = choose(router, by_id("cap-016"))
    assert "cost" in json.dumps(short, default=str).lower()
    c_short = short.get("estimated_cost") or short.get("cost_usd")
    c_long = long_in.get("estimated_cost") or long_in.get("cost_usd")
    assert c_short is not None and c_long is not None, "il dry-run deve esporre il costo stimato"
    assert c_long > c_short * 10, (
        f"costo stimato quasi identico ({c_short} vs {c_long}) per 500 e 30000 token di input"
    )


@pytest.mark.spec
def test_output_heavy_prompts_cost_more_than_short_ones():
    """cap-021/cap-022 hanno input minuscolo e output enorme."""
    router = make_router()
    a = choose(router, by_id("triv-013"))
    b = choose(router, by_id("cap-021"))
    ca = a.get("estimated_cost") or a.get("cost_usd")
    cb = b.get("estimated_cost") or b.get("cost_usd")
    assert cb > ca * 3, f"stima output insensibile al tipo di task ({ca} vs {cb})"


@pytest.mark.spec
def test_context_window_is_a_hard_filter():
    """vendor-l/free-tier ha 8k di contesto: i prompt da 30k non ci stanno.

    Oggi non esiste alcun filtro sul contesto: il router puo' scegliere un
    modello che rifiutera' la richiesta, e l'errore arriva a runtime.
    """
    router = make_router()
    pool = load_pool()
    for r in ALL:
        if "padding" not in r:
            continue
        need = r["padding"]["target_tokens"]
        chosen = spec_by_id(pool, choose(router, r)["model"])
        assert chosen.context_length >= need, (
            f"{r['id']}: servono {need} token, {chosen.id} ne ha {chosen.context_length}"
        )


# ===========================================================================
# GRUPPO F - capacita'. Segnali che il classificatore calcola e il router
# butta via. Il fallimento qui e' silenzioso, che e' il peggior tipo.
# ===========================================================================


@pytest.mark.spec
def test_pii_forces_a_local_model():
    """sensitive e' calcolato da classify() e mai letto dal router.

    Su dati AVS, sanitari o carte di credito questo non e' un dettaglio di
    qualita': e' un problema di conformita' (nLPD).
    """
    router = make_router()
    pool = load_pool()
    for r in ALL:
        if "local_only" not in r["caps"]:
            continue
        chosen = spec_by_id(pool, choose(router, r)["model"])
        assert chosen.provider == "local", (
            f"{r['id']} contiene dati personali ma va su {chosen.id}"
        )


@pytest.mark.spec
def test_freshness_forces_a_search_capable_model():
    router = make_router()
    pool = load_pool()
    for r in ALL:
        if "search" not in r["caps"]:
            continue
        chosen = spec_by_id(pool, choose(router, r)["model"])
        assert "search" in getattr(chosen, "caps", []), (
            f"{r['id']} richiede informazioni correnti ma va su {chosen.id}, senza ricerca"
        )


@pytest.mark.spec
def test_vision_requests_never_go_to_text_only_models():
    router = make_router()
    pool = load_pool()
    for r in ALL:
        if "vision" not in r["caps"]:
            continue
        chosen = spec_by_id(pool, choose(router, r)["model"])
        assert "vision" in getattr(chosen, "caps", []), (
            f"{r['id']} e' multimodale ma va su {chosen.id}, text-only"
        )


# ===========================================================================
# GRUPPO G - snapshot di regressione.
# Non giudica se una decisione e' giusta. Rende VISIBILE ogni cambiamento.
# Si aggiorna con GOLDEN_UPDATE=1 pytest, e il diff si legge nella PR.
# ===========================================================================


def test_routing_snapshot():
    import os

    router = make_router()
    current = {}
    for r in ALL:
        sig = classify(r["messages"])
        d = choose(router, r)
        current[r["id"]] = {
            "task": sig.task,
            "difficulty": round(sig.difficulty, 3),
            "confidence": round(sig.confidence, 3),
            "model": d["model"],
        }

    if os.environ.get("GOLDEN_UPDATE") == "1" or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip("snapshot riscritto: rileggi il diff prima di committare")

    previous = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    changed = {
        pid: (previous.get(pid), current[pid])
        for pid in current
        if previous.get(pid) != current[pid]
    }
    assert not changed, (
        f"{len(changed)} decisioni cambiate. Se e' voluto: GOLDEN_UPDATE=1 pytest "
        f"e committa il diff dello snapshot.\n"
        + "\n".join(f"  {pid}: {old} -> {new}" for pid, (old, new) in list(changed.items())[:15])
    )
