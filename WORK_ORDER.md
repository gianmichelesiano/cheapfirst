# Ordine di lavoro — cheapfirst

Documento operativo. Ogni task ha un criterio di accettazione eseguibile: se il
comando indicato non passa, il task non è finito. Non serve interpretare.

**Regola generale:** un task per commit, un commit per PR quando tocca il
comportamento. `pytest -m "not spec"` deve restare verde a ogni commit. I test
marcati `spec` si spostano fuori dal marker man mano che diventano verdi — non
si cancellano e non si modificano per farli passare.

---

## Fatto — stato accertato

Verificato con `curl`: OpenRouter **espone** `benchmarks.artificial_analysis`
con `intelligence_index`, `coding_index`, `agentic_index` su 194 modelli.
`_parse_openrouter` legge già i campi giusti. L'ipotesi che i benchmark fossero
assenti era sbagliata: `_filter_competent` e il filtro per tier funzionano
davvero, il pool non collassa, e il ranking non degenera per mancanza di dati.

Questo **non** salva la formula. `score = costo / benchmark` continua a scegliere
quasi sempre il più economico, perché il prezzo varia su tre ordini di grandezza
e l'indice di qualità su circa due volte. Con benchmark reali (nano=25, gpt-5=70,
opus=72) e un prompt di difficoltà 1.00 la scelta resta `gpt-5-nano`. Il
problema è la forma della funzione, non il dato in ingresso.

---

## T0 — Congelare il pool reale (30 min)

`tests/fixtures/golden_models.json` contiene prezzi **sintetici**. Vanno
sostituiti con uno snapshot reale, altrimenti i test di costo misurano numeri
inventati.

Creare `scripts/freeze_pool.py`:

```python
#!/usr/bin/env python3
"""Congela uno snapshot del pool da OpenRouter per i test di routing.

    python scripts/freeze_pool.py > tests/fixtures/golden_models.json

Lo snapshot NON si rigenera automaticamente. Se il pool cambia sotto i test,
i test non misurano più niente. Si aggiorna solo di proposito, e il diff si
legge nella PR.
"""
import json, sys, datetime, httpx

# ~20 modelli scelti a mano per coprire: 3 ordini di grandezza di prezzo,
# almeno un dominato, un outlier di prezzo, uno senza benchmark, uno con
# coding_index > intelligence_index, uno multimodale, uno con contesto piccolo.
KEEP = [
    # riempire con gli id reali dopo aver guardato l'output
]

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

`tier_for` deve replicare **esattamente** la logica di `_parse_openrouter`,
altrimenti la fixture e il registry vivo divergono. Vocabolario dei tier:
`free | cheap | mid | frontier | ultra`. Non esistono `medium` né `premium`.

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

## T1 — `_rank` crasha su benchmark `None` (10 min)

Bug reale, raggiungibile oggi. `_filter_competent` scarta i `None`, ma quando il
fallback `competent = pool` scatta, `_rank` li riceve:

```
TypeError: '<=' not supported between instances of 'NoneType' and 'int'
```

Con 194 modelli benchmarkati su un catalogo molto più grande, i `None` sono la
maggioranza del pool. Ogni volta che il filtro svuota, il router muore.

Serve una policy **esplicita**, non un `or 0`:

- benchmark assente → il modello è escluso dalla scelta di default
- l'esclusione compare nel `reason` con il conteggio (`"12 modelli esclusi:
  benchmark non disponibile"`), non sparisce in silenzio
- opzione di config `unmeasured_policy: exclude | impute_from_tier`, default
  `exclude`

`0` significa "misurato, pessimo". `None` significa "non lo sappiamo". Trattarli
uguale punisce ogni modello nuovo per il solo fatto di essere nuovo.

**Accettazione:** `pytest -m spec -k unmeasured` verde, e il caso del crash
aggiunto come test in `tests/test_router.py`.

---

## T2 — Provider dei modelli locali (20 min)

`get_active_pool` filtra su `ModelSpec.provider` (`"local"`), mentre
`executor.get_provider_info` ricava il provider da `model_id.split("/")[0]`,
cioè `"ollama"` o `"ds4-local"`. Nessuno dei due sta in `provider_keys`, quindi
**nessun modello locale è eseguibile**, incluso il setup ds4.

La fonte di verità deve essere una sola: `ModelSpec.provider`. L'executor va
cambiato per ricevere lo spec, non l'id nudo, oppure `ModelExtra` deve imporre
che il prefisso dell'id coincida col provider dichiarato.

Nello stesso commit: `get_provider_info` non deve più costruire URL indovinati
con `f"https://api.{provider}.com/v1"`. Un provider sconosciuto è un errore
esplicito, non un host inventato a cui spedire una API key.

**Accettazione:** con la config di esempio e un server locale attivo,
`cheapfirst run "ciao" --dry-run` sceglie il modello locale ed `execute` lo
raggiunge davvero.

---

## T3 — `openrouter` come provider di esecuzione (mezza giornata)

Il branch `feat/openrouter-executor` va in questa direzione. È il task con il
miglior rapporto valore/sforzo di tutto l'elenco.

Oggi il registry scarica gli id da OpenRouter, poi l'executor toglie il prefisso
e chiama l'API nativa del provider. Ma il prefisso OpenRouter è il **creatore**
del modello, non chi lo serve. Conseguenze già verificate:

- Anthropic riceve una POST su `/v1/chat/completions` con `Authorization: Bearer`.
  Serve `/v1/messages`, `x-api-key`, `anthropic-version`. Ogni chiamata fallisce.
- `anthropic/claude-sonnet-4` → `claude-sonnet-4` non è un model string valido.
- La regola "se non finisce per `/v1`, aggiungi `/v1`" produce
  `.../v1beta/openai/v1/chat/completions` su Google e
  `api.perplexity.ai/v1/chat/completions` su Perplexity. Entrambi rotti.
- Con una chiave Groq non si instrada niente: su OpenRouter i modelli serviti da
  Groq hanno prefisso `meta-llama/`.

Un adapter OpenRouter risolve tutto insieme: un endpoint, un formato di auth, gli
id già corretti perché sono gli stessi del registry. I provider diretti restano
come ottimizzazione opzionale per chi ha le chiavi, non come percorso principale.

**Accettazione:** con la sola `OPENROUTER_API_KEY`, una richiesta reale va a buon
fine su almeno un modello per ciascuno dei creatori `openai`, `anthropic`,
`google`, `deepseek`, `meta-llama`.

---

## T4 — Il ranking (mezza giornata)

**Qualità come vincolo, costo come obiettivo.** Non pesi.

```python
def quality_floor(difficulty: float) -> float:
    """difficoltà -> indice di qualità minimo accettabile."""
    # interpolazione lineare sui punti di ancoraggio
    anchors = [(0.0, 20), (0.25, 30), (0.5, 42), (0.75, 55), (1.0, 65)]
    ...

candidati = [m for m in pool if bench(m, task) is not None
             and bench(m, task) >= quality_floor(sig.difficulty)]
scelto = min(candidati, key=lambda m: costo_stimato(m, messages, sig))
```

Se `candidati` è vuoto, si prende il migliore disponibile e lo si dichiara nel
`reason` — mai il più economico.

**Da non fare, e il perché:**

- **Niente normalizzazione min-max.** Con un outlier a 75 $/M tutti i modelli
  sotto 1 $/M finiscono con `cost_norm` fra 0.001 e 0.012: indistinguibili. Il
  peso costo muore e la scelta diventa "sempre quasi il migliore". Simulato con
  `cost_weight=0.3, quality_weight=0.7`: vince un modello da 15 $/M su ogni
  richiesta, traduzioni di due parole comprese. È il bug attuale rovesciato.
  In più il punteggio dipende da min e max del pool, quindi le decisioni cambiano
  da sole a ogni `registry update`, a prompt identico. Niente riproducibilità,
  niente caching, niente debug.
- **Niente filtro di Pareto davanti al ranking.** Un modello dominato ha
  punteggio peggiore in qualunque funzione monotona: non può vincere comunque.
  Il filtro brucia CPU e non cambia un solo output. Verificato:
  `test_no_dominated_model_is_ever_chosen` passa già oggi.
- Pareto **serve** altrove: per costruire la scala di escalation (risalire la
  frontiera è il percorso giusto) e per popolare `alternatives` nel dry-run.

**Accettazione:**
```bash
pytest -m spec -k "quality_floor or cheapest_above_floor or monotone or outlier"
```
`test_decision_is_independent_of_pool_outliers` passa già: se diventa rosso,
è entrata una normalizzazione dipendente dal pool. Non aggirarlo.

---

## T5 — Stima del costo (2 ore)

`_rank` oggi ignora `prompt_per_m` e assume 200 token di output fissi. Raffinare
la formula tenendo l'input sbagliato non serve: su un prompt lungo l'input domina
il conto, e il ranking sceglie un modello che in realtà costa molte volte tanto.

- token di input contati dai messaggi reali (`len(text) // 4` basta e avanza)
- output stimato per tipo di task: traduzione ≈ lunghezza input, codice 500–1500,
  risposta secca ≈ 50
- `context_length` come filtro rigido: oggi non esiste, e il router può scegliere
  un modello che rifiuterà la richiesta a runtime

**Accettazione:** `pytest -m spec -k "cost_estimate or output_heavy or context_window"`

---

## T6 — Costo cumulativo ed escalation (2 ore)

Due bug collegati, entrambi sul KPI centrale del progetto.

1. `result["cost_usd"]` conta solo il turno riuscito. I token bruciati nei turni
   falliti non entrano mai nel totale, quindi risparmi e metriche sono
   sottostimati per costruzione.
2. `_execute_with_verify` itera su `ranked[:max_turns]`, cioè i tre modelli **più
   economici**: un verify fallito porta su un modello quasi identico, non su uno
   più forte. E il fallback finale riesegue `ranked[0]`, cioè proprio quello che
   aveva già fallito.

Escalation corretta: alzare il `quality_floor` e ripescare. Con T4 in casa sono
tre righe. Aggiungere al retry il motivo del fallimento — oggi si rimandano gli
stessi `messages` senza dire cosa non andava, quindi non è un ciclo di
correzione, è una ripetizione.

**Accettazione:** su una richiesta con due escalation, `cost_usd` è la somma dei
tre turni, e ogni turno usa un modello con indice di qualità superiore al
precedente.

---

## T7 — Server (2 ore)

- `execute` è `httpx` sincrono dentro endpoint `async def`: blocca l'event loop e
  il proxy serializza tutto. Passare ad `AsyncClient`.
- `stream` è accettato e ignorato dal server; nell'executor `body["stream"]=True`
  seguito da `resp.json()` si rompe. O si implementa SSE, o si rifiuta con un
  errore chiaro. Il silenzio è la scelta peggiore.
- Bind `0.0.0.0` senza autenticazione con passthrough a modello arbitrario:
  chiunque sulla rete usa le API key. Default `127.0.0.1` più un bearer token.

---

## T8 — Classificatore (dopo gli altri)

Volutamente ultimo. `task` alimenta una scelta a **due valori** — `code` contro
tutto il resto — perché cinque delle sei etichette prodotte selezionano
`intelligence_index`, e `analysis` non è nemmeno raggiungibile da `classify()`,
quindi `agentic_index` è codice morto. Distinguere `creative` da `factual` da
`general` non cambia nessuna decisione.

1. Collassare `task` a `code | agentic | other`, rendere `agentic` raggiungibile.
2. Segnali **strutturali**, non lessicali: code fence, indentazione,
   `camelCase`/`snake_case`, densità di parentesi e punti e virgola, stack trace,
   LaTeX, densità di cifre, script non latino. Più l'unione delle keyword su 5
   lingue in una regex sola.
3. Difficoltà **fail-safe**: la loss è asimmetrica. Sottostimare costa una
   risposta inutilizzabile e non te ne accorgi; sovrastimare costa frazioni di
   centesimo. Sbagliare verso l'alto.
4. Lingua rilevata come *feature*, non come problema: molti modelli economici
   sono nettamente peggiori fuori dall'inglese.

**Niente modellino addestrato per ora.** I 10k prompt etichettati dovrebbero
uscire da un LLM, quindi il soffitto è quello di una classificazione LLM ma con
in più una pipeline di training, un binario da versionare e zero spiegabilità.
Si valuta **solo se** il golden set mostra le euristiche che perdono soldi.

**Accettazione:** `pytest -m spec -k "language or loanword or false_hard or
false_easy or proof_verb or non_technical or low_confidence"`. Il gruppo di
invarianza linguistica è il criterio vero: stesso compito in cinque lingue →
stesso task, stessa banda entro 0.15, **stesso modello scelto**.

---

## T9 — Etichette misurate (3 USD)

```bash
python scripts/label_golden.py --dry-run
python scripts/label_golden.py --budget 5 --resume
python scripts/label_golden.py --review   # da guardare a mano, non c'è scorciatoia
```

Produce `cheapest_ok_model` e `min_quality`: il modello più economico che dà una
risposta accettabile, misurato risalendo una scala di sei modelli. Da lì esce il
numero da mettere nel README al posto dell'80%, e da lì si calibra
`quality_floor()` invece di sceglierne i punti a mano.

---

## T10 — README

Da correggere quando il resto è in piedi:

- "Save up to 80% on API costs" → sostituire con il numero misurato in T9, o
  toglierlo. È l'affermazione più fragile del documento.
- Togliere la stoccata a Maestro ("5-hour build"): su un progetto di 14 commit si
  ritorce contro.
- `pip install -e . pytest` è un errore di battitura → `pip install -e ".[dev]"`.
- `pip install cheapfirst` non funziona finché non è pubblicato.
- Il modello virtuale si chiama ancora `costflow-auto`: residuo del nome
  precedente.
- Messaggi di errore e report sono in italiano dentro un README inglese
  (`Nessun provider attivo`, `Giorno Richieste Spesa`). Scegliere una lingua.

Aggiungere anche: CI su GitHub Actions che lancia `pytest -m "not spec"`, e la
dipendenza `openai` in `pyproject.toml` va tolta — non è importata da nessuna
parte.

---

## Ordine consigliato

```
T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7 → T9 → T8 → T10
```

T0–T3 sono infrastruttura: senza, tutto il resto misura o esegue cose sbagliate.
T4–T6 sono il prodotto. T8 è l'ultimo perché è la parte divertente ed è quella
che sposta di meno.
