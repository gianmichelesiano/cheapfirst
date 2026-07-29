"""Server HTTP FastAPI per cheapfirst.

Espone endpoint OpenAI-compatibile, Anthropic-compatibile e health check.
"""

from cheapfirst import CheapFirst
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    FastAPI = None
    BaseModel = None
    uvicorn = None


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatRequest(BaseModel):
    model: str = "costflow-auto"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: int = 4096


class ModelList(BaseModel):
    object: str = "list"
    data: list[dict] = []


def create_app(config_path: str | None = None) -> FastAPI | None:
    """Crea l'app FastAPI. Restituisce None se FastAPI non è installato."""
    if FastAPI is None:
        return None

    app = FastAPI(title="cheapfirst", version="0.1.0")
    router_engine = CheapFirst(config=config_path)

    @app.get("/v1/models")
    async def list_models():
        """Restituisce la lista dei modelli disponibili."""
        models = router_engine.registry.models
        return ModelList(
            data=[
                {
                    "id": m.id,
                    "object": "model",
                    "created": 0,
                    "owned_by": m.provider,
                }
                for m in models[:50]
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        """Endpoint OpenAI-compatibile per chat completions."""
        messages_dict = [m.model_dump() for m in req.messages]

        if req.model == "costflow-auto":
            result = router_engine.chat(messages_dict)
        else:
            # Passthrough: usa il modello specificato
            from cheapfirst.executor import execute
            provider_keys = router_engine.config.resolve_provider_keys()
            result = execute(
                messages=messages_dict,
                model_id=req.model,
                provider_keys=provider_keys,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            )

        if not result.get("success", True):
            raise HTTPException(
                status_code=502,
                detail=result.get("error", "Provider error"),
            )

        return {
            "id": f"chatcmpl-{hash(str(messages_dict)) & 0xFFFFFFFF:08x}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": result.get("model_used", req.model),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("text", ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.get("input_tokens", 0),
                "completion_tokens": result.get("output_tokens", 0),
                "total_tokens": result.get("input_tokens", 0) + result.get("output_tokens", 0),
            },
            "cost_usd": result.get("cost_usd", 0),
        }

    @app.get("/v1/route")
    async def route_dry_run(prompt: str):
        """Dry-run: mostra quale modello verrebbe usato."""
        decision = router_engine.decide(prompt)
        return decision

    @app.post("/v1/route")
    async def route_dry_run_post(data: dict):
        """Dry-run via POST (per UI)."""
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(400, "prompt required")
        decision = router_engine.decide(prompt)
        return decision

    @app.get("/", response_class=HTMLResponse)
    @app.get("/ui", response_class=HTMLResponse)
    async def ui():
        """Interfaccia web."""
        ui_path = Path(__file__).parent / "ui" / "index.html"
        if ui_path.exists():
            return HTMLResponse(ui_path.read_text())
        return HTMLResponse("<h1>cheapfirst</h1><p>UI not found</p>")

    @app.get("/healthz")
    async def healthz():
        """Health check."""
        return {
            "status": "ok",
            "version": "0.1.0",
            "models_count": len(router_engine.registry.models),
            "providers": list(
                router_engine.config.resolve_provider_keys().keys()
            ),
        }

    return app


def run_server(port: int = 8080, config: str | None = None):
    """Avvia il server HTTP."""
    if uvicorn is None:
        print("Errore: installa cheapfirst con: pip install cheapfirst[server]")
        return

    app = create_app(config)
    if app is None:
        print("Errore: FastAPI non installato. Usa: pip install cheapfirst[server]")
        return

    print(f"cheapfirst server in ascolto su http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


# Esporta solo se FastAPI è installato
__all__ = ["create_app", "run_server"]
