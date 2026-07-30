#!/usr/bin/env python3
"""Riempie le etichette MISURATE del golden set.

`min_quality` e `cheapest_ok_model` non sono opinioni: si ottengono eseguendo
davvero una scala di modelli dal piu' economico al piu' caro e fermandosi al
primo che produce una risposta accettabile. Quella e' l'unica etichetta che
serve al router, ed e' anche l'unico modo per dare un numero all'80% del README.

    # 1. stima quanto costa la campagna, senza spendere niente
    python scripts/label_golden.py --dry-run

    # 2. gira per davvero, con un tetto di spesa
    python scripts/label_golden.py --budget 5.00 --out tests/fixtures/golden_labeled.jsonl

    # 3. riprendi dopo un'interruzione (non rifa' il lavoro gia' fatto)
    python scripts/label_golden.py --budget 5.00 --resume

Il file sorgente non viene MAI sovrascritto: l'output e' un file separato.

Nota sul giudizio: i check deterministici (contains_any, regex, exec_python)
coprono 54 prompt su 200 e sono gratis, ripetibili e non discutibili. Per gli
altri serve un giudice LLM, che e' la parte piu' fragile di tutto il
procedimento: il giudice sbaglia, e sbaglia in modo correlato col modello
giudicato. Per questo:
  - il giudice e' un modello forte e FISSO, dichiarato nell'output
  - restituisce un voto binario piu' una motivazione, che viene salvata
  - `--review` stampa i casi in cui il giudice e' stato incerto, per revisione
    umana. Quei casi vanno guardati a mano. Non c'e' scorciatoia.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tests" / "fixtures" / "golden_prompts.jsonl"
POOL = ROOT / "tests" / "fixtures" / "golden_models.json"

# Scala di escalation: dal piu' economico al piu' caro. Sostituire con gli id
# reali del proprio pool. Sei pioli bastano: piu' pioli = piu' costo per un
# guadagno di risoluzione trascurabile.
LADDER = [
    "vendor-b/flash-lite",
    "vendor-a/nano",
    "vendor-b/flash",
    "vendor-c/chat",
    "vendor-e/luna",
    "vendor-g/sonnet-class",
]

JUDGE = "vendor-i/opus-class"

JUDGE_SYSTEM = """Sei un valutatore. Ricevi un prompt, una risposta e un criterio.
Rispondi SOLO con JSON, nessun markdown, nessun preambolo:
{"ok": true|false, "confident": true|false, "why": "<max 20 parole>"}
ok=true se la risposta soddisfa il criterio.
confident=false se il criterio e' ambiguo o la risposta e' borderline.
Sii severo: una risposta plausibile ma sbagliata e' ok=false."""


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade_deterministic(check: dict, text: str) -> bool | None:
    """Restituisce None se il check non e' deterministico."""
    kind = check["type"]
    if kind == "none":
        return True  # non giudicabile automaticamente: qualsiasi output passa
    if kind == "contains_any":
        return any(v in text for v in check["values"])
    if kind == "regex":
        return re.search(check["pattern"], text) is not None
    if kind == "exec_python":
        return grade_exec(check["asserts"], text)
    return None


def grade_exec(asserts: str, text: str) -> bool:
    """Estrae il primo blocco di codice e lancia gli assert in un subprocess.

    Deliberatamente in subprocess con timeout: il codice viene da un LLM e non
    va eseguito nello stesso processo del harness.
    """
    import subprocess
    import tempfile

    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    code = blocks[0] if blocks else text
    # il set usa `f` come nome canonico della funzione sotto test
    harness = f"{code}\n\nSRC = {code!r}\n"
    for name in re.findall(r"^\s*def\s+(\w+)", code, re.M):
        harness += f"f = {name}\n"
        break
    harness += asserts + "\nprint('PASS')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=10)
        return "PASS" in r.stdout
    except Exception:
        return False
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Chiamate
# ---------------------------------------------------------------------------


@dataclass
class Spend:
    usd: float = 0.0
    calls: int = 0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, usd: float) -> None:
        self.usd += usd
        self.calls += 1
        self.by_model[model] = self.by_model.get(model, 0.0) + usd


def call(model: str, messages: list[dict], pricing: dict, spend: Spend,
         max_tokens: int = 2000) -> tuple[str, float]:
    """Una chiamata via OpenRouter. Un solo endpoint, un solo formato di auth.

    L'harness usa OpenRouter di proposito: qui non ci interessa ottimizzare il
    costo, ci interessa che 6 modelli diversi rispondano senza scrivere 6
    adapter. E' anche la dimostrazione che il percorso OpenRouter e' quello
    giusto come default nel prodotto.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("serve OPENROUTER_API_KEY")
    r = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "max_tokens": max_tokens},
        timeout=180,
    )
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"] or ""
    u = data.get("usage", {})
    cost = (
        pricing["prompt_per_m"] * u.get("prompt_tokens", 0) / 1e6
        + pricing["completion_per_m"] * u.get("completion_tokens", 0) / 1e6
    )
    spend.add(model, cost)
    return text, cost


def judge(prompt_text: str, answer: str, rubric: str, pricing: dict, spend: Spend) -> dict:
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content":
            f"PROMPT:\n{prompt_text[:4000]}\n\nRISPOSTA:\n{answer[:6000]}\n\nCRITERIO:\n{rubric}"},
    ]
    text, _ = call(JUDGE, msgs, pricing, spend, max_tokens=200)
    try:
        return json.loads(re.sub(r"```json|```", "", text).strip())
    except Exception:
        return {"ok": False, "confident": False, "why": "giudice non parsabile"}


# ---------------------------------------------------------------------------
# Padding sintetico
# ---------------------------------------------------------------------------


def inflate(row: dict) -> list[dict]:
    """Espande il campo `padding` in contesto reale.

    Il set non porta 40k token di lorem ipsum nel repository: descrive come
    generarli. Cosi' la fixture resta leggibile e diffabile.
    """
    msgs = [dict(m) for m in row["messages"]]
    pad = row.get("padding")
    if not pad:
        return msgs
    chunk = pad["repeat"]
    n = max(1, pad["target_tokens"] * 4 // max(1, len(chunk)))
    filler = "".join(chunk.replace("{i}", str(i)) for i in range(n))
    msgs[-1]["content"] = f"{msgs[-1]['content']}\n\n---\n{filler}"
    return msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "golden_labeled.jsonl"))
    ap.add_argument("--budget", type=float, default=5.0, help="tetto di spesa in USD")
    ap.add_argument("--dry-run", action="store_true", help="stima il costo e non chiama nulla")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--review", action="store_true", help="stampa i casi da rivedere a mano")
    ap.add_argument("--only", help="filtra per tag, es. --only regression")
    args = ap.parse_args()

    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        rows = [r for r in rows if args.only in r["tags"]]
    pool = {m["id"]: m for m in json.loads(POOL.read_text(encoding="utf-8"))["models"]}
    out_path = Path(args.out)

    done: dict[str, dict] = {}
    if args.resume and out_path.exists():
        done = {json.loads(l)["id"]: json.loads(l)
                for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"resume: {len(done)} prompt gia' etichettati")

    if args.review:
        flagged = [r for r in done.values() if r.get("needs_human_review")]
        print(f"{len(flagged)} casi da rivedere a mano:\n")
        for r in flagged:
            print(f"  {r['id']:12} {r.get('review_why', '')}")
        return

    if args.dry_run:
        # Stima grossolana: ogni prompt sale la scala fino a meta' in media,
        # piu' un giudizio per i check non deterministici.
        est = 0.0
        for r in rows:
            in_tok = 500 + r.get("padding", {}).get("target_tokens", 0)
            out_tok = {"short": 80, "medium": 500, "long": 1800}[r["out_tokens"]]
            for mid in LADDER[: max(1, len(LADDER) // 2)]:
                p = pool[mid]["pricing"]
                est += p["prompt_per_m"] * in_tok / 1e6 + p["completion_per_m"] * out_tok / 1e6
            if r["check"]["type"] == "judge":
                p = pool[JUDGE]["pricing"]
                est += p["prompt_per_m"] * (in_tok + out_tok) / 1e6 + p["completion_per_m"] * 60 / 1e6
        print(f"prompt: {len(rows)}")
        print(f"scala: {len(LADDER)} pioli, giudice: {JUDGE}")
        print(f"costo stimato campagna completa: ${est:.2f}")
        print(f"deterministici (gratis da giudicare): "
              f"{sum(1 for r in rows if r['check']['type'] != 'judge')}/{len(rows)}")
        return

    spend = Spend()
    t0 = time.time()
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            if row["id"] in done:
                continue
            if spend.usd >= args.budget:
                print(f"\ntetto di spesa raggiunto (${spend.usd:.2f}). "
                      f"Riprendi con --resume.")
                break

            msgs = inflate(row)
            prompt_text = msgs[-1]["content"]
            max_tok = {"short": 200, "medium": 800, "long": 3000}[row["out_tokens"]]
            result = dict(row)
            attempts = []
            winner = None

            for mid in LADDER:
                m = pool[mid]
                if m["context_length"] < row.get("padding", {}).get("target_tokens", 0):
                    attempts.append({"model": mid, "skipped": "contesto insufficiente"})
                    continue
                try:
                    text, cost = call(mid, msgs, m["pricing"], spend, max_tok)
                except Exception as e:  # noqa: BLE001
                    attempts.append({"model": mid, "error": str(e)[:120]})
                    continue

                ok = grade_deterministic(row["check"], text)
                why, confident = "check deterministico", True
                if ok is None:
                    v = judge(prompt_text, text, row["check"]["rubric"], pool[JUDGE]["pricing"], spend)
                    ok, confident, why = bool(v.get("ok")), bool(v.get("confident")), v.get("why", "")

                attempts.append({
                    "model": mid, "ok": ok, "confident": confident,
                    "why": why, "cost_usd": round(cost, 6),
                    "answer_head": text[:200],
                })
                if ok:
                    winner = mid
                    result["judge_confident"] = confident
                    break

            result["cheapest_ok_model"] = winner
            result["min_quality"] = (
                pool[winner]["benchmarks"].get(
                    {"code": "coding_index", "agentic": "agentic_index"}.get(
                        row["task"], "intelligence_index")
                ) if winner else None
            )
            result["attempts"] = attempts
            result["ladder_cost_usd"] = round(sum(a.get("cost_usd", 0) for a in attempts), 6)
            result["needs_human_review"] = (
                winner is None or not result.get("judge_confident", True)
            )
            if result["needs_human_review"]:
                result["review_why"] = (
                    "nessun modello della scala e' passato" if winner is None
                    else f"giudice incerto: {attempts[-1].get('why', '')}"
                )
            result["labeled_with"] = {"ladder": LADDER, "judge": JUDGE}

            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            flag = "" if not result["needs_human_review"] else "  <- REVISIONE"
            print(f"[{i:3}/{len(rows)}] {row['id']:12} -> {winner or 'NESSUNO':22} "
                  f"${spend.usd:6.3f}{flag}")

    n = sum(1 for _ in out_path.open(encoding="utf-8"))
    print(f"\n{n} prompt etichettati in {time.time() - t0:.0f}s, spesa ${spend.usd:.3f}")
    print("spesa per modello:")
    for mid, usd in sorted(spend.by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {mid:24} ${usd:.4f}")
    print("\nProssimo passo: `--review` e guardare a mano i casi segnalati. "
          "Il giudice LLM non e' la verita', e' un primo passaggio.")


if __name__ == "__main__":
    main()
