# Golden set

Due cose distinte, spesso confuse.

**Fase 1 — invarianti.** Non richiedono etichette, non chiamano nessuna API,
girano in mezzo secondo. Sono già utilizzabili e sono la parte che vale di più.

**Fase 2 — etichette misurate.** `cheapest_ok_model` e `min_quality` si
ottengono eseguendo davvero i modelli. Costano soldi e tempo, e vanno riviste a
mano nei casi incerti. Sono `null` finché non gira l'harness.

L'errore da non fare è invertire l'ordine: aspettare le etichette per iniziare
a testare. Gli invarianti catturano già la maggior parte delle regressioni.

## File

| file | cos'è |
|---|---|
| `golden_prompts.jsonl` | 200 prompt etichettati a mano. **Fonte di verità**, si edita a mano. |
| `golden_models.json` | pool congelato, 20 modelli. Prezzi e indici **sintetici**: da sostituire con uno snapshot reale. |
| `routing_snapshot.json` | decisione corrente per ogni prompt. Generato. Il diff si legge in PR. |
| `golden_labeled.jsonl` | output della fase 2. Generato. |

## Comandi

```bash
pytest -m "not spec"      # verde sempre. Se è rosso, è una regressione.
pytest -m spec            # 29 rossi oggi: è la to-do list del refactor.
GOLDEN_UPDATE=1 pytest    # riscrive lo snapshot. Guardare il diff PRIMA di committare.

python scripts/label_golden.py --dry-run              # ~$3 per la campagna completa
python scripts/label_golden.py --budget 5 --resume    # fase 2
python scripts/label_golden.py --review               # casi da guardare a mano
```

Registrare il marker in `pyproject.toml`, altrimenti pytest riempie l'output di
warning:

```toml
[tool.pytest.ini_options]
markers = ["spec: comportamento obiettivo, atteso rosso finché non implementato"]
```

## Composizione dei 200 prompt

| | n |
|---|---|
| banda: trivial / easy / moderate / hard / frontier | 27 / 53 / 64 / 36 / 20 |
| task collassato: other / code / agentic | 123 / 51 / 26 |
| lingua: en / it / de / fr / es / ja / zh / ar | 133 / 43 / 10 / 6 / 5 / 1 / 1 / 1 |
| non inglese | 67 (34%) |
| confidenza attesa bassa (il classificatore **deve** astenersi) | 22 |
| casi di regressione su bug noti | 53 |
| avversari (falsi hard, falsi code, output corti, contesto assente) | 44 |
| output atteso: short / medium / long | 76 / 55 / 69 |
| con padding sintetico (fino a 40k token di input) | 7 |

Giudizio: 54 prompt con check **deterministico** (`contains_any`, `regex`,
`exec_python`) — gratis, ripetibile, non discutibile. 85 richiedono un giudice
LLM. 61 non sono giudicabili automaticamente (creativi, `check: none`) e servono
solo a verificare che il routing non spenda troppo.

Il rapporto conta: più check deterministici significa meno dipendenza dal
giudice, che è il punto debole di tutto il procedimento. Quando si aggiungono
prompt, preferire sempre un check verificabile.

## Perché `band` e non un numero

Un valore puntuale di difficoltà (`0.63`) non è verificabile da nessuno: non
esiste un modo di stabilire che sia 0.63 e non 0.58. Una banda sì, ed è quello
che serve, perché il consumo a valle è comunque una soglia discreta:

```python
BAND_FLOOR = {"trivial": 20, "easy": 30, "moderate": 42, "hard": 55, "frontier": 65}
```

Questa tabella **è** la specifica del routing. Qualità come vincolo, costo come
obiettivo: sopra il floor, il più economico. I due test
`test_quality_floor_is_respected` e `test_cheapest_above_floor_is_chosen`
definiscono insieme la scelta in modo completo. Se sono entrambi verdi, non
serve discutere di pesi.

## Il test canarino

`test_decision_is_independent_of_pool_outliers` **passa oggi** e deve restare
verde. Aggiunge al pool un modello da 500 $/M e verifica che nessuna decisione
cambi. Diventa rosso il giorno in cui qualcuno introduce una normalizzazione
min-max: quel giorno il punteggio di ogni modello inizia a dipendere da min e max
del pool, e le decisioni cambiano da sole a ogni `registry update`.

Vale anche l'inverso: `test_no_dominated_model_is_ever_chosen` passa oggi *per
costruzione*, non per merito. Nessuna funzione monotona nelle due dimensioni può
scegliere un punto dominato. Serve solo come rete se qualcuno introduce
non-monotonicità (per esempio un bonus per tier).

## Attenzione ai verdi falsi

`test_pii_forces_a_local_model` passa oggi, ma per il motivo sbagliato: i modelli
locali costano `0.00`, quindi il router li sceglie perché sono i più economici,
non perché il prompt contiene un numero AVS. Diventerà rosso appena il floor di
qualità entra in funzione, perché su un prompt moderate il modello locale non
supererà la soglia. Quel rosso è il vero stato delle cose: `sensitive` è
calcolato da `classify()` e mai letto dal router.

## Manutenzione

- Gli `id` sono stabili. Un id ritirato non si riusa: rompe lo storico dello snapshot.
- Aggiungendo prompt, aggiungere prima il caso avversario e poi il fix.
- `golden_models.json` non si aggiorna automaticamente. Se il pool cambia sotto i
  test, i test non misurano più niente.
- Ogni bug trovato in produzione entra qui come riga nuova, con `tags:
  ["regression"]` e una `notes` che spiega cosa si è rotto.
