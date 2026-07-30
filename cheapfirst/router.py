"""Motore di routing: filtra, ranka, decide, esegue escalation."""

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

ESTIMATED_OUTPUT = {
    "translation": lambda input_len: max(50, input_len // 2),
    "factual": 50,
    "code": 1000,
    "math": 500,
    "creative": 800,
    "analysis": 1200,
    "general": 200,
}


def _quality_floor(difficulty: float) -> float:
    """Mappa difficolta (0-1) in soglia di benchmark minima."""
    if difficulty < 0.15:
        return 20.0
    elif difficulty < 0.35:
        return 30.0
    elif difficulty < 0.55:
        return 42.0
    elif difficulty < 0.75:
        return 52.0
    else:
        return 60.0


def _estimate_cost(model: ModelSpec, task_type: str, input_chars: int) -> float:
    """Stima il costo di una richiesta su un modello."""
    prompt_per_m = model.pricing.get("prompt_per_m", 0) or 0
    completion_per_m = model.pricing.get("completion_per_m", 0) or 0
    input_tokens = max(10, input_chars // 4)
    output_est = ESTIMATED_OUTPUT.get(task_type, 200)
    output_tokens = output_est(input_tokens) if callable(output_est) else output_est
    prompt_cost = (input_tokens * prompt_per_m) / 1_000_000
    completion_cost = (output_tokens * completion_per_m) / 1_000_000
    return max(prompt_cost + completion_cost, 0.00001)


def _benchmark_value(model: ModelSpec, task_type: str) -> Optional[float]:
    """Restituisce il benchmark per un task, o None se non misurato."""
    bench_key = BENCHMARK_MAP.get(task_type, "intelligence_index")
    return model.benchmarks.get(bench_key)


class Router:
    """Decide quale modello usare per ogni richiesta."""

    def __init__(self, config: CheapConfig, registry: ModelRegistry):
        self.config = config
        self.registry = registry
        self._provider_keys = config.resolve_provider_keys()

    def route(self, messages: list[dict], sig: TaskSignature, dry_run: bool = False) -> dict:
        """Esegue il routing completo o dry-run."""
        pool = self.registry.get_active_pool(self._provider_keys)
        if not pool:
            return {"error": "Nessun provider attivo con API key valida", "success": False}

        competent = self._filter_competent(pool, sig)
        if not competent:
            competent = pool

        ranked = self._rank(competent, sig)

        if dry_run:
            return self._format_decision(ranked, sig)

        return self._execute_with_verify(ranked, messages, sig)

    def _filter_competent(self, pool: list[ModelSpec], sig: TaskSignature) -> list[ModelSpec]:
        """Filtra i modelli competenti per questo task."""
        min_score = self.config.routing.min_benchmark_score
        bench_key = BENCHMARK_MAP.get(sig.task, "intelligence_index")

        min_tier = "cheap"
        for threshold, tiers in TIER_BY_DIFFICULTY:
            if sig.difficulty < threshold:
                min_tier = tiers[0]
                break

        competent = []
        for m in pool:
            bench = _benchmark_value(m, sig.task)
            if bench is None:
                continue
            if min_score > 0 and bench <= 0:
                continue
            if bench < min_score:
                continue
            tier_ok = False
            for threshold, tiers in TIER_BY_DIFFICULTY:
                if sig.difficulty < threshold:
                    tier_ok = m.tier in tiers
                    break
            if not tier_ok:
                continue
            competent.append(m)

        return competent

    def _rank(self, pool: list[ModelSpec], sig: TaskSignature) -> list[tuple]:
        """Ranking: quality floor + cheapest + lambda tie-break.

        - floor = quality_floor(sig.difficulty): soglia di competenza
        - Filtra: solo modelli con benchmark >= floor
        - Tra quelli, scegli il piu economico (costo stimato)
        - lambda tie-break: se due modelli hanno costo simile (< 5%),
          vince quello con benchmark piu alto.

        Restituisce [(model, cost_est, bench, floor), ...] ordinato per costo.
        """
        floor = _quality_floor(sig.difficulty)
        input_chars = max(50, int(sig.difficulty * 500))

        candidates = []
        for m in pool:
            bench = _benchmark_value(m, sig.task)
            if bench is None or bench < floor:
                continue
            cost = _estimate_cost(m, sig.task, input_chars)
            candidates.append((m, cost, bench))

        if not candidates:
            # Fallback: nessun modello sopra la soglia
            # Prendi il migliore disponibile
            for m in sorted(pool, key=lambda x: _benchmark_value(x, sig.task) or 0, reverse=True):
                bench = _benchmark_value(m, sig.task)
                if bench is not None and bench > 0:
                    cost = _estimate_cost(m, sig.task, input_chars)
                    candidates.append((m, cost, bench))
                    break

        # Ordina per costo
        candidates.sort(key=lambda x: x[1])

        # lambda tie-break: costo simile entro 5% -> vince benchmark piu alto
        result = []
        skip = set()
        for i, (m, cost, bench) in enumerate(candidates):
            if i in skip:
                continue
            tied = [(j, candidates[j]) for j in range(i + 1, len(candidates))
                    if abs(candidates[j][1] - cost) / max(cost, 0.00001) < 0.05]
            if tied:
                group = [(i, candidates[i])] + tied
                best = max(group, key=lambda x: x[1][2])
                result.append(best[1])
                skip.update(j for j, _ in group)
            else:
                result.append((m, cost, bench))

        return [(m, cost, bench, floor) for m, cost, bench in result]

    def _format_decision(self, ranked: list[tuple], sig: TaskSignature) -> dict:
        """Formatta una decisione per dry-run."""
        if not ranked:
            return {"error": "Nessun modello disponibile", "success": False}

        top_model, top_cost, top_bench, top_floor = ranked[0]
        alternatives = [(m.id, round(c, 6)) for m, c, b, f in ranked[1:4]]

        return {
            "model": top_model.id,
            "score": round(top_cost, 8),
            "cost_est": round(top_cost, 8),
            "benchmark": top_bench,
            "quality_floor": top_floor,
            "reason": (
                f"task={sig.task}, difficulty={sig.difficulty:.2f}, "
                f"floor={top_floor:.0f}, cheapest above floor"
            ),
            "alternatives": alternatives,
            "pool_size": len(ranked),
        }

    def _execute_with_verify(
        self, ranked: list[tuple], messages: list[dict], sig: TaskSignature
    ) -> dict:
        """Esegue il miglior modello, verify, scala se serve."""
        config = self.config.routing
        max_turns = config.max_turns
        skip_verify = sig.confidence >= config.skip_verify_confidence

        for turn, (model, cost_est, bench, floor) in enumerate(ranked[:max_turns]):
            result = execute(
                messages=messages,
                model_id=model.id,
                provider_keys=self._provider_keys,
            )

            if not result.get("success", True):
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

            skip_verify = False

        # Budget esaurito: usa il miglior modello
        best = ranked[0]
        result = execute(
            messages=messages,
            model_id=best[0].id,
            provider_keys=self._provider_keys,
        )
        result["cost_usd"] = calculate_cost(
            best[0].pricing,
            result.get("input_tokens", 0),
            result.get("output_tokens", 0),
        )
        result["turns"] = max_turns
        result["budget_exhausted"] = True
        return result
