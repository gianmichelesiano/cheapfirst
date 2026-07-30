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

# Default benchmark values per tier when imputing from tier
TIER_BENCH_DEFAULTS = {
    "free": 20,
    "cheap": 30,
    "mid": 45,
    "frontier": 60,
    "ultra": 70,
}


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

        # 1. Filtra per competenza
        competent = self._filter_competent(pool, sig)
        if not competent:
            competent = pool  # fallback: usa tutto il pool

        # 2. Ranka per task type
        ranked, excluded_count = self._rank(competent, sig.task)
        if not ranked:
            return {"error": "Nessun modello disponibile dopo il ranking", "success": False}

        top_model, top_score, top_cost, top_bench = ranked[0]

        # Build reason
        reason_parts = [
            f"task={sig.task}, difficulty={sig.difficulty:.2f}, "
            f"confidence={sig.confidence:.2f}, best cost/benchmark ratio"
        ]
        if excluded_count:
            reason_parts.append(f"{excluded_count} modelli esclusi: benchmark non disponibile")

        if dry_run:
            alternatives = [(m.id, round(s, 4)) for m, s, _, _ in ranked[1:4]]
            return {
                "model": top_model.id,
                "score": round(top_score, 6),
                "cost_est": round(top_cost, 8),
                "benchmark": top_bench,
                "reason": ", ".join(reason_parts),
                "alternatives": alternatives,
                "pool_size": len(competent),
            }

        # 3. Esecuzione con verify/escalate
        return self._execute_with_verify(ranked, messages, sig)

    def _filter_competent(self, pool: list[ModelSpec], sig: TaskSignature) -> list[ModelSpec]:
        """Filtra i modelli competenti per questo task."""
        routing = self.config.routing
        min_score = routing.min_benchmark_score
        bench_key = BENCHMARK_MAP.get(sig.task, "intelligence_index")

        competent = []
        for m in pool:
            bench = m.benchmarks.get(bench_key)
            if bench is not None and bench < min_score:
                continue

            # Tier check
            tier_ok = False
            for threshold, tiers in TIER_BY_DIFFICULTY:
                if sig.difficulty < threshold:
                    tier_ok = m.tier in tiers
                    break
            if not tier_ok:
                continue

            competent.append(m)

        return competent

    def _rank(self, pool: list[ModelSpec], task_type: str) -> tuple[list[tuple], int]:
        """Ranka i modelli per costo/benchmark. Più basso = meglio.

        Restituisce (lista_rankata, numero_modelli_esclusi).
        """
        policy = self.config.routing.unmeasured_policy
        bench_key = BENCHMARK_MAP.get(task_type, "intelligence_index")
        est_out = 200  # stima output token

        ranked = []
        excluded = 0
        for m in pool:
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
                bench = 0.1  # evita division by zero

            completion_price = m.pricing.get("completion_per_m", 0)
            # Stima costo per questa richiesta
            cost = (completion_price * est_out) / 1_000_000
            cost = max(cost, 0.00001)  # minimo per evitare zero

            score = cost / bench
            ranked.append((m, score, cost, bench))

        ranked.sort(key=lambda x: x[1])
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
