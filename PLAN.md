# cheapfirst — Piano di Implementazione Passo Passo

## Stato attuale
Repo: https://github.com/gianmichelesiano/cheapfirst

## Fasi

### Fase 1: Registry + Classifier (completato ✅)
- [x] Classificatore euristico (regex): code, math, creative, factual, translation, general
- [x] Registry da OpenRouter API (fetch + parse + cache)
- [x] Modelli custom da config YAML
- [x] Filtro per API key attive

### Fase 2: Router + Executor (in corso)
- [ ] Router: filtro per competenza + ranking costo/benchmark
- [ ] Executor: chiamata API reale OpenAI-compatibile
- [ ] Risoluzione base_url + api_key per provider
- [ ] Streaming support
- [ ] Timeout e retry

### Fase 3: Verify + Escalate
- [ ] Verify strutturato per tipo di task
- [ ] Escalation ladder con max_turns
- [ ] Budget controllo costi per verify

### Fase 4: Metriche + Report
- [ ] Logging richieste su SQLite
- [ ] Report testuale (costi, risparmio, modelli usati)
- [ ] Report comparativo (costo vs frontier-only)

### Fase 5: CLI Completa
- [ ] `cheapfirst route` — routing + esecuzione
- [ ] `cheapfirst decide` — dry-run
- [ ] `cheapfirst registry update/check/show`
- [ ] `cheapfirst report --days N`

### Fase 6: Server HTTP
- [ ] FastAPI endpoint
- [ ] POST /v1/chat/completions
- [ ] GET /v1/models
- [ ] GET /healthz

### Fase 7: Testing
- [ ] Test classificatore
- [ ] Test router
- [ ] Test registry
- [ ] Test verify
- [ ] Test metriche
- [ ] Test integrazione

### Fase 8: Produzione
- [ ] pip publish
- [ ] README completo con esempi
- [ ] Documentazione API
- [ ] **Articolo tecnico**

## Come monitorare

```bash
# Vedi i file nel repo
gh repo view gianmichelesiano/cheapfirst

# Vedi i commit recenti
cd /tmp/cheapfirst && git log --oneline -20

# Test
cd /tmp/cheapfirst && python -m pytest tests/ -v
```
