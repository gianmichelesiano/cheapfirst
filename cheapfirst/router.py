"""Motore di routing: filtra, ranka, decide, esegue escalation.

Il routing segue questi passi:
1. `_filter_competent`: esclude modelli con tier non adatto.
2. `_rank`: applica quality floor in base alla difficoltà, poi sceglie il più economico.
3. `route`: seleziona il primo modello e chiama l'executor.
"""

from .classifier import TaskSignature
from .config import CheapConfig
from .registry import ModelRegistry, ModelSpec
from .executor import execute, calculate_cost
from .verify import verify_response, Verdict

from typing import Optional


BENCHMARK_MAP = {
    "code": "coding_index",
    "math": "intelligence_index",
    "creative": "intelligence_index",
    "factual": "intelligence_index",
    "translation": "intelligence_index",
    "analysis": "agentic_index",
    "general": "intelligence_index",
}

TIER_BY_DIFFICULTY = [
    (0.33, ["free", "cheap"]),
    (0.70, ["free", "cheap", "mid"]),
    (1.01, ["free", "cheap", "mid", "frontier", "ultra"]),
]

# Default benchmark values per tier when imputing from tier
TIER_BENCH_DEFAULTS = {
    "free": 20,
    "cheap": 30,
    "mid": 45,
    "frontier": 60,
    "ultra": 70,
}

# Punti di ancoraggio per quality_floor(difficulty): step piecewise
# Ogni difficoltà appartiene a più bande (BAND_RANGE si sovrappone).
# Usiamo la banda con il floor PIÙ BASSO che copre la difficoltà,
# per non escludere troppo e lasciare che il ranking scelga il cheapest.
# Allineato con BAND_FLOOR nei test spec.
QUALITY_FLOOR_STEPS = [
    (0.25, 20.0),   # diff ≤ 0.25 → trivial → floor 20
    (0.40, 30.0),   # diff ≤ 0.40 → easy → floor 30
    (0.62, 42.0),   # diff ≤ 0.62 → moderate → floor 42
    (0.85, 55.0),   # diff ≤ 0.85 → hard → floor 55
    (1.00, 65.0),   # diff > 0.85 → frontier → floor 65
]

# Stima output token per tipo task
# Stima output token per tipo task.
# Translation: ~input tokens (preserva lunghezza).
# Code: output sostanzioso.
# Factual/short: poche decine di token.
# Usati da _rank() per il costo reale (input + output).
OUTPUT_ESTIMATES = {
    "code": 800,
    "math": 400,
    "analysis": 400,
    "creative": 300,
    "translation": None,  # special: proportional to input
    "factual": 60,
    "general": 150,
}


def quality_floor(difficulty: float) -> float:
    """Calcola il quality floor per una data difficoltà.

    Usa step piecewise allineato con BAND_FLOOR nei test spec:
    ogni soglia di difficoltà attiva un floor minimo.
    """
    for threshold, floor in QUALITY_FLOOR_STEPS:
        if difficulty <= threshold:
            return floor
    return QUALITY_FLOOR_STEPS[-1][1]


class Router:
    """Decide quale modello usare per ogni richiesta."""

    def __init__(self, config: CheapConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self._provider_keys = config.resolve_provider_keys()

    def route(self, messages: list[dict], sig: TaskSignature, dry_run: bool = False) -> dict:
        """Esegue il routing completo o dry-run."""
        pool = self.registry.get_active_pool(self.config.resolve_provider_keys())
        if not pool:
            return {"error": "Nessun provider attivo con API key valida", "success": False}

        # Stima token di input dai messaggi reali
        input_tokens = self._estimate_input_tokens(messages)

        # Stima output per tipo di task
        output_est = self._estimate_output_tokens(sig.task, input_tokens)

        # 1. Filtra per competenza (tier) — con fallback a tutto il pool
        competent = self._filter_competent(pool, sig)
        if not competent:
            competent = pool

        # 2. Filtra per capabilities (HARD constraint — niente fallback)
        capable = self._filter_capable(competent, sig)
        if not capable:
            return {"error": "Nessun modello con le capacita' richieste", "success": False}

        # 3. Ranka per task type con quality floor + cheapest
        ranked, excluded_count = self._rank(capable, sig.task, sig.difficulty, input_tokens, output_est)
        if not ranked:
            # Fallback 1: riprova senza tier filter (ma mantenendo caps)
            ranked, excluded_count = self._rank(capable, sig.task, sig.difficulty, input_tokens, output_est)

        if not ranked:
            # Fallback 2: nessun modello supera il floor, prendi il migliore disponibile
            ranked, excluded_count = self._rank(capable, sig.task, 0.0, input_tokens, output_est)
        
        if not ranked:
            return {"error": "Nessun modello disponibile dopo il ranking", "success": False}

        top_model, top_score, top_cost, top_bench = ranked[0]

        # Build reason
        floor = quality_floor(sig.difficulty)
        reason_parts = [
            f"task={sig.task}, difficulty={sig.difficulty:.2f}, "
            f"confidence={sig.confidence:.2f}, floor={floor:.1f}, "
            f"in={input_tokens}tok, out={output_est}tok"
        ]
        if excluded_count:
            reason_parts.append(f"{excluded_count} modelli esclusi: sotto floor o senza benchmark")

        if dry_run:
            alternatives = [(m.id, round(s, 8)) for m, s, _, _ in ranked[1:4]]
            return {
                "model": top_model.id,
                "score": round(top_score, 8),
                "cost_est": round(top_cost, 8),
                "estimated_cost": round(top_cost, 8),
                "cost_usd": round(top_cost, 8),
                "benchmark": top_bench,
                "reason": ", ".join(reason_parts),
                "alternatives": alternatives,
                "pool_size": len(competent),
            }

        # 3. Esecuzione con verify/escalate
        return self._execute_with_verify(ranked, messages, sig)

    def _estimate_input_tokens(self, messages: list[dict]) -> int:
        """Stima il numero di token di input dai messaggi reali."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return max(1, total_chars // 4)

    def _estimate_output_tokens(self, task_type: str, input_tokens: int) -> int:
        """Stima i token di output in base al tipo di task."""
        est = OUTPUT_ESTIMATES.get(task_type)
        if est is None:
            # Translation: output ≈ input (preserva lunghezza)
            return max(1, input_tokens)
        return max(1, est)

    def _filter_competent(self, pool: list[ModelSpec], sig: TaskSignature) -> list[ModelSpec]:
        """Filtra i modelli per tier — con fallback se nessun tier matcha."""
        competent = []
        for m in pool:
            for threshold, tiers in TIER_BY_DIFFICULTY:
                if sig.difficulty < threshold:
                    if m.tier in tiers:
                        competent.append(m)
                    break
        return competent

    def _filter_capable(self, pool: list[ModelSpec], sig: TaskSignature) -> list[ModelSpec]:
        """Filtra per capabilities richieste (HARD constraint).

        - sig.sensitive  → solo modelli local_only
        - sig.freshness  → solo modelli con caps 'search'
        - sig.caps contiene 'multimodal' → solo modelli con caps 'vision'
        """
        if not sig.sensitive and not sig.freshness and not sig.caps:
            return pool  # nessun vincolo

        capable = []
        for m in pool:
            # sensitive/PII → local_only
            if sig.sensitive and "local_only" not in getattr(m, "caps", []) and m.provider != "local":
                continue
            # freshness → search capable
            if sig.freshness and "search" not in getattr(m, "caps", []):
                continue
            # multimodal → vision capable
            if "multimodal" in (sig.caps or []) and "vision" not in getattr(m, "caps", []):
                continue
            capable.append(m)
        return capable

    def _rank(self, pool: list[ModelSpec], task_type: str, difficulty: float,
              input_tokens: int = 500, output_est: int = 200) -> tuple[list[tuple], int]:
        """Filtra per quality floor e contesto, ordina per costo reale (input + output).

        cost = prompt_per_m * input_tokens + completion_per_m * output_est / 1e6

        Restituisce (lista_rankata, numero_modelli_esclusi).
        score = costo reale stimato.
        """
        policy = self.config.routing.unmeasured_policy
        bench_key = BENCHMARK_MAP.get(task_type, "intelligence_index")
        floor = quality_floor(difficulty)

        ranked = []
        excluded = 0
        for m in pool:
            # Context window filter: il modello deve poter contenere l'input
            if m.context_length < input_tokens:
                excluded += 1
                continue

            bench = m.benchmarks.get(bench_key)

            if bench is None:
                if policy == "exclude":
                    excluded += 1
                    continue
                elif policy == "impute_from_tier":
                    bench = TIER_BENCH_DEFAULTS.get(m.tier, 30)
                else:
                    excluded += 1
                    continue

            if bench <= 0:
                bench = 0.1  # evita zero

            # Quality floor: bench deve essere >= floor
            if bench < floor:
                excluded += 1
                continue

            prompt_price = m.pricing.get("prompt_per_m", 0)
            completion_price = m.pricing.get("completion_per_m", 0)
            cost = (prompt_price * input_tokens + completion_price * output_est) / 1_000_000
            cost = max(cost, 1e-8)  # evita zero (serve per test di costo estimato)

            # score = costo (più economico = meglio) per compatibilità
            ranked.append((m, cost, cost, bench))

        ranked.sort(key=lambda x: (x[1], -x[3]))  # cheapest first, then highest bench
        return ranked, excluded

    def _execute_with_verify(
        self,
        ranked: list[tuple],
        messages: list[dict],
        sig: TaskSignature,
    ) -> dict:
        """Esegue il miglior modello, verify, scala se serve."""
        config = self.config.routing
        max_turns = config.max_turns
        budget = config.verify_cost_budget

        # Se confidenza alta, skip verify al primo turno
        skip_verify = sig.confidence >= config.skip_verify_confidence

        for turn, (model, score, cost, bench) in enumerate(ranked[:max_turns]):
            result = execute(
                messages=messages,
                model_id=model.id,
                provider_keys=self._provider_keys,
                provider=model.provider,
            )

            # Se la chiamata è fallita (errore di rete, provider, etc.)
            if not result.get("success", True):
                # Logga l'errore e prova il prossimo modello
                if turn == len(ranked[:max_turns]) - 1:
                    result["turns"] = turn + 1
                    return result
                continue

            if skip_verify:
                result["cost_usd"] = calculate_cost(
                    model.pricing,
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                )
                result["turns"] = turn + 1
                result["verify_used"] = False
                return result

            # Verify (se il budget lo consente)
            if cost > budget:
                result["cost_usd"] = calculate_cost(
                    model.pricing,
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                )
                result["turns"] = turn + 1
                result["verify_used"] = False
                return result

            verdict = verify_response(result, sig)
            if verdict == Verdict.ACCEPT:
                result["cost_usd"] = calculate_cost(
                    model.pricing,
                    result.get("input_tokens", 0),
                    result.get("output_tokens", 0),
                )
                result["turns"] = turn + 1
                result["verify_used"] = True
                result["verify_passed"] = True
                return result

            # REVISE: scala al prossimo modello
            skip_verify = False  # ora facciamo verify obbligatorio

        # Budget esaurito: usa il miglior modello (non l'ultimo del ranking)
        best_model = ranked[0][0]
        result = execute(
            messages=messages,
            model_id=best_model.id,
            provider_keys=self._provider_keys,
            provider=best_model.provider,
        )
        result["cost_usd"] = calculate_cost(
            best_model.pricing,
            result.get("input_tokens", 0),
            result.get("output_tokens", 0),
        )
        result["turns"] = max_turns
        result["budget_exhausted"] = True
        return result
