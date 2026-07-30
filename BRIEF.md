# BRIEF — cheapfirst implementation tasks

Documento unico e autosufficiente. Contiene: come prendere il branch di test,
lo stato accertato del progetto, e 11 task ordinati con criteri di accettazione
eseguibili. Non serve altro contesto.

## Regole di lavoro

- Un task per commit. Un commit per PR quando tocca il comportamento.
- `pytest -m "not spec"` deve restare verde a ogni commit.
- I test marcati `spec` sono la to-do list: si spostano fuori dal marker quando
  diventano verdi. Non si cancellano e non si modificano per farli passare.
- Se un criterio di accettazione non passa, il task non è finito.

## Stato accertato

Verificato con curl: OpenRouter API (`/api/v1/models`) espone
`benchmarks.artificial_analysis` con `intelligence_index`, `coding_index`,
`agentic_index` su **194 modelli**. `_parse_openrouter` legge già i campi giusti.

Conseguenza: `_filter_competent` e il filtro per tier funzionano davvero, il pool
non collassa, il ranking non degenera per mancanza di dati.

Questo non salva la formula. `score = costo / benchmark` continua a scegliere
quasi sempre il più economico, perché il prezzo varia su tre ordini di grandezza
e l'indice di qualità su circa due volte. Con benchmark reali (nano=25, gpt-5=70,
opus=72) e un prompt di difficoltà 1.00 la scelta resta gpt-5-nano. Il problema
è la forma della funzione, non il dato in ingresso.

Vocabolario dei tier del router: `free | cheap | mid | frontier | ultra`.
Non esistono `medium` né `premium`.

## Branch golden-set

Il branch `origin/golden-set` contiene 200 prompt etichettati a mano, un pool di
modelli congelato, 64 test (35 passanti, 29 falliti marcati `spec` — la to-do
list). Contiene anche un fix di produzione: `name: str = ""` in ModelExtra.

```
cd cheapfirst
git checkout golden-set
pip install -e ".[dev]"
pytest -m "not spec"      # 34 passed
pytest -m spec            # 29 failed, 35 passed  <- la to-do list
```

Cosa c'è dentro:

| File | Contenuto |
|---|---|
| tests/fixtures/golden_prompts.jsonl | 200 prompt etichettati a mano: 8 lingue (34% non inglese), 53 regression, 44 avversari, 7 con padding fino a 40k token |
| tests/fixtures/golden_models.json | Pool congelato di 20 modelli. Prezzi sintetici (da sostituire in T0) |
| tests/fixtures/routing_snapshot.json | Snapshot delle decisioni correnti |
| tests/test_golden_routing.py | 64 test |
| scripts/label_golden.py | Harness per le etichette misurate (T9) |

---

## Task

### T0 — Congelare il pool reale (30 min)

`tests/fixtures/golden_models.json` ha prezzi sintetici. Vanno sostituiti con
uno snapshot reale, altrimenti i test di costo misurano numeri inventati.

Creare `scripts/freeze_pool.py`:

```python
#!/usr/bin/env python3
"""Congela uno snapshot del pool da OpenRouter per i test di routing.

    python scripts/freeze_pool.py > tests/fixtures/golden_models.json
"""
import json, sys, datetime, httpx

KEEP = []  # ~20 modelli scelti a mano

def main():
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    out = []
    for m in data:
        if KEEP and m["id"] not in KEEP:
            continue
        aa = (m.get("benchmarks") or {}).get("artificial_analysis") or {}
        p = m.get("pricing", {})
        out.append({
            "id": m["id"],
            "provider": m["id"].split("/")[0],
            "tier": tier_for(float(p.get("completion", 0)) * 1e6, m["id"]),
            "pricing": {
                "prompt_per_m": round(float(p.get("prompt", 0)) * 1e6, 4),
                "completion_per_m": round(float(p.get("completion", 0)) * 1e6, 4),
            },
            # NON convertire None in 0: sono cose diverse
            "benchmarks": {
                "intelligence_index": aa.get("intelligence_index"),
                "coding_index": aa.get("coding_index"),
                "agentic_index": aa.get("agentic_index"),
            },
            "context": m.get("context_length", 4096),
            "caps": (["vision"] if "image" in (m.get("architecture", {})
                     .get("input_modalities") or []) else []),
        })
    json.dump({"frozen_at": str(datetime.date.today()), "source": "openrouter",
               "models": out}, sys.stdout, indent=2)
```

`tier_for` deve replicare esattamente la logica di `_parse_openrouter`,
altrimenti la fixture e il registry vivo divergono.

**Accettazione:**
```bash
python scripts/freeze_pool.py > tests/fixtures/golden_models.json
python -c "
import json; r=json.load(open('tests/fixtures/golden_models.json'))
t={m['tier'] for m in r['models']}
assert t <= {'free','cheap','mid','frontier','ultra'}, t
assert any(m['benchmarks']['intelligence_index'] is None for m in r['models'])
assert max(m['pricing']['completion_per_m'] for m in r['models']) > 30
assert min(m['pricing']['completion_per_m'] for m in r['models']) < 0.2
print('pool OK', len(r['models']))"
GOLDEN_UPDATE=1 pytest -m "not spec" && pytest -m "not spec"
```

---

### T1 — `_rank` crasha su benchmark None (10 min)

Bug reale, riproducibile oggi. `_filter_competent` scarta i None, ma quando
scatta il fallback `competent = pool` finiscono dritti in `_rank`:
`TypeError: '<=' not supported between instances of 'NoneType' and 'int'`

Con 194 modelli benchmarkati su un catalogo di centinaia, i None sono la
maggioranza del pool. Ogni volta che il filtro svuota, il router muore.

Serve una policy esplicita, non un `or 0`:

- benchmark assente → il modello è escluso dalla scelta di default
- l'esclusione compare nel reason col conteggio (`"12 modelli esclusi: benchmark
  non disponibile"`), non sparisce in silenzio
- opzione di config `unmeasured_policy`: `exclude | impute_from_tier`, default
  `exclude`

`0` significa "misurato, pessimo". `None` significa "non lo sappiamo". Trattarli
uguale punisce ogni modello nuovo per il solo fatto di essere nuovo.

**Accettazione:** `pytest -m spec -k unmeasured` verde, più il caso del crash
aggiunto come test in `tests/test_router.py`.

---

### T2 — Provider dei modelli locali (20 min)

`get_active_pool` filtra su `ModelSpec.provider` (`"local"`), mentre
`executor.get_provider_info` ricava il provider da `model_id.split("/")[0]`,
cioè "ollama" o "ds4-local". Nessuno dei due sta in `provider_keys`, quindi
nessun modello locale è eseguibile.

La fonte di verità deve essere una sola: `ModelSpec.provider`. O l'executor
riceve lo `spec` invece dell'id nudo, o `ModelExtra` impone che il prefisso
dell'id coincida col provider dichiarato.

Nello stesso commit: `get_provider_info` non deve più costruire URL indovinati
con `f"https://api.{provider}.com/v1"`. Un provider sconosciuto è un errore
esplicito, non un host inventato.

**Accettazione:** con la config di esempio e un server locale attivo,
`cheapfirst run "ciao" --dry-run` sceglie il modello locale ed `execute` lo
raggiunge davvero.

---

### T3 — OpenRouter come provider di esecuzione (mezza giornata)

Oggi il registry scarica gli id da OpenRouter, poi l'executor toglie il prefisso
e chiama l'API nativa del provider. Ma il prefisso OpenRouter è il creatore
del modello, non chi lo serve. Conseguenze:
- Anthropic riceve una POST su `/v1/chat/completions` con `Authorization: Bearer`.
  Serve `/v1/messages`, `x-api-key`, `anthropic-version`. Ogni chiamata fallisce.
- `anthropic/claude-sonnet-4` → `claude-sonnet-4` non è un model string valido.
- Google/Perplexity ricevono URL con doppio /v1.
- Groq non si instrada: i modelli hanno prefisso `meta-llama/`.

Un adapter OpenRouter risolve tutto insieme: un endpoint, un formato di auth, gli
id già corretti perché sono gli stessi del registry. I provider diretti restano
come ottimizzazione opzionale.

**Accettazione:** con la sola `OPENROUTER_API_KEY`, una richiesta reale va a buon
fine su almeno un modello per ciascuno dei creator: `openai`, `anthropic`,
`google`, `deepseek`, `meta-llama`.

---

### T4 — Il ranking (mezza giornata)

Qualità come vincolo, costo come obiettivo. Non pesi.

```python
def quality_floor(difficulty: float) -> float:
    """difficoltà -> indice di qualità minimo accettabile."""
    anchors = [(0.0, 20), (0.25, 30), (0.5, 42), (0.75, 55), (1.0, 65)]
    # interpolazione lineare fra gli ancoraggi

candidati = [m for m in pool if bench(m, task) is not None
             and bench(m, task) >= quality_floor(sig.difficulty)]
scelto = min(candidati, key=lambda m: costo_stimato(m, messages, sig))
```

Se candidati è vuoto, si prende il migliore disponibile e lo si dichiara nel
reason — mai il più economico.

Il `min_quality` deve essere derivato da `sig.difficulty`, non una costante
nel YAML: altrimenti il classificatore diventa decorativo.

#### Da non fare (simulato e scartato)

- **Niente normalizzazione min-max.** Con un outlier a 75$/M tutti i modelli
  sotto 1$/M finiscono con `cost_norm` fra 0.001 e 0.012: indistinguibili.
  Il peso costo muore e la scelta diventa "sempre quasi il migliore".
- **Niente filtro di Pareto davanti al ranking.** Un modello dominato ha
  punteggio peggiore in qualunque funzione monotona: non può vincere comunque.
  Il filtro brucia CPU e non cambia un solo output. Pareto serve per la scala
  di escalation e per popolare `alternatives` nel dry-run.

**Accettazione:**
```
pytest -m spec -k "quality_floor or cheapest_above_floor or monotone or outlier"
```
`test_decision_is_independent_of_pool_outliers` passa già oggi: se diventa
rosso, è entrata una normalizzazione dipendente dal pool.

---

### T5 — Stima del costo (2 ore)

`_rank` ignora `prompt_per_m` e assume 200 token di output fissi.

- token di input contati dai messaggi reali (`len(text) // 4`)
- output stimato per tipo di task: traduzione ≈ lunghezza input, codice 500–1500,
  risposta secca ≈ 50
- `context_length` come filtro rigido

**Accettazione:** `pytest -m spec -k "cost_estimate or output_heavy or context_window"`

---

### T6 — Costo cumulativo ed escalation (2 ore)

Due bug collegati sul KPI centrale:

1. `result["cost_usd"]` conta solo il turno riuscito. I token bruciati nei
   turni falliti non entrano mai nel totale.
2. `_execute_with_verify` itera su `ranked[:max_turns]`, cioè i tre modelli
   **più economici**: un verify fallito porta su un modello quasi identico, non
   su uno più forte. Il fallback finale riesegue `ranked[0]`, cioè quello che
   aveva già fallito.

Escalation corretta: alzare il quality_floor e ripescare.

**Accettazione:** su una richiesta con due escalation, `cost_usd` è la somma dei
tre turni, e ogni turno usa un modello con indice di qualità superiore al
precedente.

---

### T7 — Server (2 ore)

- `execute` è `httpx` sincrono dentro endpoint `async def`: blocca l'event loop.
  Passare ad `AsyncClient`.
- `stream` è accettato e ignorato dal server: o si implementa SSE, o si rifiuta.
- Bind `0.0.0.0` senza autenticazione: default `127.0.0.1` più bearer token.

---

### T8 — Classificatore (penultimo, di proposito)

`task` alimenta una scelta a due valori — code contro tutto il resto — perché
cinque delle sei etichette selezionano `intelligence_index`, e `analysis` non è
nemmeno raggiungibile da `classify()`. Distinguere `creative` da `factual` da
`general` non cambia nessuna decisione.

1. Collassare task a `code | agentic | other`, rendere `agentic` raggiungibile.
2. Segnali strutturali, non lessicali.
3. Difficoltà fail-safe: la loss è asimmetrica. Sbagliare verso l'alto.
4. Lingua rilevata come feature.

**Accettazione:** `pytest -m spec -k "language or loanword or false_hard or
false_easy or proof_verb or non_technical or low_confidence"`.

---

### T9 — Etichette misurate (~3 USD)

```bash
export OPENROUTER_API_KEY=...
python scripts/label_golden.py --dry-run     # stima il costo
python scripts/label_golden.py --budget 5 --resume
python scripts/label_golden.py --review      # da guardare a mano
```

---

### T10 — README e pulizia

- "Save up to 80%" → sostituire col numero misurato in T9.
- Togliere stoccata a Maestro.
- `pip install -e . pytest` → `pip install -e ".[dev]"`.
- `costflow-auto` → `cheapfirst`.
- Messaggi errore in italiano → inglese.
- Togliere dipendenza `openai` da `pyproject.toml`.
- Aggiungere CI (`pytest -m "not spec"`).

---

## Segnali calcolati e mai usati

`sensitive`, `freshness` e `caps` sono calcolati dal classificatore e non usati
da nessuna parte. Anche `cost_weight`, `quality_weight` sono letti e ignorati.
O si collegano o si tolgono.

## Ordine di esecuzione

T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T9 → T8 → T10

T0–T3 sono infrastruttura. T4–T6 sono il prodotto. T8 è ultimo.
