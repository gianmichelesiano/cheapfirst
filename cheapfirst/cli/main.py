"""CLI principale per cheapfirst."""

import click
from .. import CheapFirst
from ..__version__ import __version__


@click.group()
@click.version_option(__version__)
@click.option("--config", "-c", default=None, help="Percorso file di configurazione")
@click.pass_context
def cli(ctx, config):
    """cheapfirst — LLM router: prova il cheap, verifica, scala.

    Risparmia fino all'80% sui costi API LLM senza sacrificare qualità.
    """
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.argument("prompt")
@click.pass_context
def route(ctx, prompt):
    """Esegue una richiesta con routing automatico."""
    cf = CheapFirst(config=ctx.obj.get("config_path"))
    result = cf.chat([{"role": "user", "content": prompt}])

    if result.get("success", True):
        click.echo()
        click.echo(f"  Risposta: {result.get('text', '')[:200]}")
        click.echo(f"  Modello:  {result.get('model_used', '?')}")
        click.echo(f"  Costo:    ${result.get('cost_usd', 0):.8f}")
        click.echo(f"  Route:    {result.get('turns', 1)} turno(i)")
        click.echo(f"  Latenza:  {result.get('latency_ms', 0)}ms")
    else:
        click.echo(f"  Errore: {result.get('error', 'Sconosciuto')}")


@cli.command()
@click.argument("prompt")
@click.pass_context
def decide(ctx, prompt):
    """Mostra quale modello userebbe (dry-run, non esegue)."""
    cf = CheapFirst(config=ctx.obj.get("config_path"))
    result = cf.decide(prompt)

    if "error" in result:
        click.echo(f"  Errore: {result['error']}")
        return

    click.echo()
    click.echo(f"  Raccomandato: {result['model']}")
    click.echo(f"  Score:        {result['score']:.6f} (costo/benchmark)")
    click.echo(f"  Costo stimato: ${result['cost_est']:.8f}")
    click.echo(f"  Task:         {result['task_type']} (difficoltà {result['difficulty']:.2f})")
    click.echo(f"  Confidenza:   {result['confidence']:.2f}")
    click.echo()
    if result.get("alternatives"):
        click.echo("  Alternative:")
        for model, score in result["alternatives"]:
            click.echo(f"    {model}: {score:.4f}")


@cli.group()
def registry():
    """Gestione del registry modelli."""
    pass


@registry.command()
@click.pass_context
def update(ctx):
    """Aggiorna il registry da OpenRouter."""
    cf = CheapFirst(config=ctx.obj.get("config_path"))
    cf.registry.update()
    status = cf.registry.status()
    click.echo(f"Registry aggiornato: {status['models_count']} modelli")


@registry.command()
@click.pass_context
def check(ctx):
    """Mostra stato del registry."""
    cf = CheapFirst(config=ctx.obj.get("config_path"))
    status = cf.registry.status()
    click.echo(f"Modelli: {status['models_count']}")
    click.echo(f"Ultimo aggiornamento: {status.get('last_update', 'sconosciuto')}")


@cli.command()
@click.option("--days", "-d", default=7, help="Numero di giorni per il report")
@click.pass_context
def report(ctx, days):
    """Genera report metriche."""
    cf = CheapFirst(config=ctx.obj.get("config_path"))
    click.echo(cf.report(days))


@cli.command()
@click.option("--port", "-p", default=8080, help="Porta del server")
@click.pass_context
def serve(ctx, port):
    """Avvia server HTTP (richiede pip install cheapfirst[server])."""
    try:
        from ..server import run_server
        run_server(port=port, config=ctx.obj.get("config_path"))
    except ImportError:
        click.echo(
            "Server non disponibile. Installa con: pip install cheapfirst[server]"
        )


def main():
    cli()
