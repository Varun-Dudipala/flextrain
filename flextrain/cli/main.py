"""FlexTrain command-line interface."""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(name="flextrain", help="FlexTrain - fault-tolerant distributed training", no_args_is_help=True,
                  add_completion=False)


def _load(config: Optional[Path], overrides: Optional[List[str]]):
    from flextrain.config import Config, ConfigError, apply_overrides, load_config

    try:
        if config is None:
            return Config.from_dict(apply_overrides({}, overrides or []))
        return load_config(config, overrides=overrides)
    except (ConfigError, FileNotFoundError) as e:
        typer.secho(f"Invalid config: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def launch(
    ctx: typer.Context,
    script: Path = typer.Argument(..., help="Training script to run on every worker"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config (its 'elastic' section)"),
    overrides: Optional[List[str]] = typer.Option(None, "--set", help="Override: section.key=value"),
    nproc_per_node: Optional[int] = typer.Option(None, help="Workers per node (default: elastic.nproc_per_node)"),
    nnodes: Optional[str] = typer.Option(None, help="N or MIN:MAX nodes (default: from elastic config)"),
    max_restarts: Optional[int] = typer.Option(None, help="Worker-group restarts before giving up"),
    rdzv_endpoint: Optional[str] = typer.Option(None, help="host:port of the rendezvous (multi-node)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the torchrun command and exit"),
):
    """Launch SCRIPT under torchrun's elastic agent (restarts + resize), exporting the config path.

    Extra arguments after the script are forwarded to it.
    """
    from flextrain.elastic import build_torchrun_command
    from flextrain.elastic import launch as run_cmd

    cfg = _load(config, overrides)
    script_args = list(ctx.args)
    try:
        cmd = build_torchrun_command(cfg, str(script), script_args, nproc_per_node=nproc_per_node, nnodes=nnodes,
                                     max_restarts=max_restarts, rdzv_endpoint=rdzv_endpoint)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    env_prefix = ""
    if config:
        env_prefix += f"FLEXTRAIN_CONFIG={shlex.quote(str(config.resolve()))} "
    if overrides:
        env_prefix += f"FLEXTRAIN_OVERRIDES={shlex.quote(chr(10).join(overrides))} "
    typer.echo(env_prefix + shlex.join(cmd))
    if dry_run:
        return
    raise typer.Exit(run_cmd(cmd, config_path=str(config) if config else None, overrides=overrides))


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to config file"),
    overrides: Optional[List[str]] = typer.Option(None, "--set", help="Override: section.key=value"),
    world_size: int = typer.Option(1, help="World size used to show the derived batch geometry"),
    show: bool = typer.Option(False, "--show", help="Print the fully resolved config"),
):
    """Validate a configuration file (strict: unknown keys and bad types are errors)."""
    cfg = _load(config, overrides)
    tc = cfg.training
    typer.secho(f"Config is valid: {config}", fg=typer.colors.GREEN)
    typer.echo(f"  experiment       : {cfg.main.experiment_name}")
    typer.echo(f"  checkpoint dir   : {cfg.resolved_checkpoint_dir()}")
    typer.echo(f"  global batch     : {tc.global_batch_for(world_size)} = {tc.batch_size} (micro) x "
               f"{tc.accumulation_steps_for(world_size)} (accum) x {world_size} (ranks)")
    duration = [f"{tc.max_steps} steps" if tc.max_steps else None, f"{tc.max_epochs} epochs" if tc.max_epochs else None]
    typer.echo(f"  duration         : {' / '.join(d for d in duration if d) or 'UNSET (set max_steps or max_epochs)'}")
    if show:
        typer.echo("\n" + cfg.to_yaml())


@app.command("init-config")
def init_config(path: Path = typer.Argument(Path("flextrain.yaml"), help="Where to write the config"),
                force: bool = typer.Option(False, "--force", help="Overwrite an existing file")):
    """Write a config file with every option at its default value."""
    from flextrain.config import Config

    if path.exists() and not force:
        typer.secho(f"{path} exists (use --force to overwrite)", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    cfg = Config()
    cfg.training.max_steps = 1000
    cfg.to_yaml(path)
    typer.echo(f"Wrote {path}")


@app.command()
def status(
    all_runs: bool = typer.Option(False, "--all", "-a", help="Include finished runs"),
    home: Optional[Path] = typer.Option(None, envvar="FLEXTRAIN_HOME", help="Registry directory"),
):
    """Show training runs from the run registry."""
    from rich.console import Console
    from rich.table import Table

    from flextrain.api.server import _with_liveness
    from flextrain.tracking import RUNNING, RunRegistry

    runs = RunRegistry(home).list()
    host = __import__("socket").gethostname()
    runs = [_with_liveness(r, host) for r in runs]
    if not all_runs:
        runs = [r for r in runs if r.get("status") == RUNNING]
    if not runs:
        typer.echo("No running jobs found." if not all_runs else "No runs found.")
        return
    table = Table(title="FlexTrain runs")
    for col in ("run", "status", "step", "loss", "world", "restarts", "last ckpt", "updated"):
        table.add_column(col)
    for r in runs:
        loss = r.get("loss")
        state = "stale" if r.get("stale") else r.get("status", "?")
        table.add_row(
            r.get("run_id", "?"), state,
            f"{r.get('step', 0)}" + (f"/{r['max_steps']}" if r.get("max_steps") else ""),
            f"{loss:.4f}" if isinstance(loss, (int, float)) else "-",
            str(r.get("world_size", "-")), str(r.get("restart_count", 0)),
            str(r.get("last_checkpoint_step", "-")),
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.get("updated_at", 0))),
        )
    Console().print(table)


@app.command()
def inspect(path: Path = typer.Argument(..., help="Checkpoint file")):
    """Print a checkpoint's metadata (step, topology, sizes) without restoring it."""
    import torch

    from flextrain.checkpoint import validate_checkpoint

    state = torch.load(path, map_location="cpu", weights_only=False)
    problems = validate_checkpoint(state)
    if problems:
        typer.secho(f"Not a valid FlexTrain checkpoint: {'; '.join(problems)}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    model = state["model"]
    n_params = sum(t.numel() for t in model.values() if isinstance(t, torch.Tensor))
    trainer = state.get("trainer", {})
    typer.echo(f"checkpoint      : {path} ({path.stat().st_size / 1e6:.1f} MB)")
    typer.echo(f"format / version: {state['format_version']} / flextrain {state.get('flextrain_version', '?')}")
    typer.echo(f"step / epoch    : {state['step']} / {state.get('epoch')}")
    typer.echo(f"world size      : {state.get('world_size')} (global batch {state.get('global_batch')})")
    typer.echo(f"data position   : {trainer.get('sampler')}")
    typer.echo(f"model           : {len(model)} tensors, {n_params / 1e6:.2f}M elements")
    typer.echo(f"optimizer state : {'yes' if state.get('optimizer') else 'no'}")
    typer.echo(f"metrics         : {state.get('metrics')}")


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port for the dashboard"),
    host: str = typer.Option("127.0.0.1", help="Interface to bind (use 0.0.0.0 to expose on the network)"),
):
    """Serve the REST API and dashboard."""
    import uvicorn

    from flextrain.api.server import create_app

    typer.echo(f"Dashboard at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


@app.command()
def version():
    """Print the FlexTrain version."""
    from flextrain import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
