"""FlexTrain CLI."""

import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(name="flextrain", help="FlexTrain - Distributed Training Framework")


@app.command()
def train(
    config: Path = typer.Argument(..., help="Path to config file"),
    resume: Optional[Path] = typer.Option(None, help="Path to checkpoint to resume from"),
):
    """Start a training run."""
    typer.echo(f"Starting training with config: {config}")
    if resume:
        typer.echo(f"Resuming from: {resume}")

    from flextrain.config import load_config
    cfg = load_config(config)
    typer.echo(f"Loaded config for experiment: {cfg.main.experiment_name}")


@app.command()
def status():
    """Show status of running training jobs."""
    typer.echo("No running jobs found.")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port for dashboard"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
):
    """Start the web dashboard."""
    typer.echo(f"Starting dashboard at http://{host}:{port}")
    try:
        import uvicorn
        from flextrain.api.server import app as api_app
        uvicorn.run(api_app, host=host, port=port)
    except ImportError:
        typer.echo("Error: uvicorn not installed. Run: pip install uvicorn")


@app.command()
def validate(config: Path = typer.Argument(..., help="Path to config file")):
    """Validate a configuration file."""
    from flextrain.config import load_config
    try:
        cfg = load_config(config)
        cfg.validate()
        typer.echo(f"Config is valid: {config}")
    except Exception as e:
        typer.echo(f"Config validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
