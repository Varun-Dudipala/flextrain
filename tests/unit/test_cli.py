"""CLI tests via Typer's CliRunner."""

import pytest
import torch
import yaml
from typer.testing import CliRunner

from flextrain import Trainer
from flextrain.cli.main import app
from flextrain.tracking import RunRegistry
from tests.conftest import RegressionDataset, TinyModel

runner = CliRunner()


def write(tmp_path, data, name="config.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_validate_ok_shows_batch_geometry(tmp_path):
    path = write(tmp_path, {"training": {"batch_size": 4, "global_batch_size": 32, "max_steps": 100,
                                         "learning_rate": "1e-4"}})
    result = runner.invoke(app, ["validate", str(path), "--world-size", "2"])
    assert result.exit_code == 0, result.output
    assert "Config is valid" in result.output and "32 = 4 (micro) x 4 (accum) x 2 (ranks)" in result.output


def test_validate_rejects_typo(tmp_path):
    path = write(tmp_path, {"training": {"learning_rat": 0.1}})
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 2 and "did you mean 'learning_rate'" in result.output


def test_validate_with_override_and_show(tmp_path):
    path = write(tmp_path, {})
    result = runner.invoke(app, ["validate", str(path), "--set", "training.max_steps=7", "--show"])
    assert result.exit_code == 0 and "max_steps: 7" in result.output


def test_init_config_round_trips(tmp_path):
    path = tmp_path / "new.yaml"
    assert runner.invoke(app, ["init-config", str(path)]).exit_code == 0
    assert runner.invoke(app, ["validate", str(path)]).exit_code == 0
    assert runner.invoke(app, ["init-config", str(path)]).exit_code == 1  # refuses to overwrite
    assert runner.invoke(app, ["init-config", str(path), "--force"]).exit_code == 0


def test_launch_dry_run_builds_torchrun_command(tmp_path):
    path = write(tmp_path, {"elastic": {"nproc_per_node": 2, "max_restarts": 4}})
    result = runner.invoke(app, ["launch", "-c", str(path), "--dry-run", "--set", "training.max_steps=5",
                                 "train.py", "--epochs", "3"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "torch.distributed.run" in out and "--nproc-per-node=2" in out and "--max-restarts=4" in out
    assert "train.py --epochs 3" in out and "FLEXTRAIN_CONFIG=" in out and "FLEXTRAIN_OVERRIDES=" in out


def test_launch_multi_node_without_endpoint_errors(tmp_path):
    path = write(tmp_path, {"elastic": {"max_nodes": 2}})
    result = runner.invoke(app, ["launch", "-c", str(path), "--dry-run", "train.py"])
    assert result.exit_code == 2 and "rdzv_endpoint" in result.output


def test_status_lists_runs(tmp_path, monkeypatch):
    reg = RunRegistry()
    reg.update("exp-run", status="completed", step=10, max_steps=10, loss=0.25, world_size=2)
    assert "No running jobs" in runner.invoke(app, ["status"]).output
    result = runner.invoke(app, ["status", "--all"], env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "exp-run" in result.output and "0.2500" in result.output


def test_inspect_checkpoint(make_config):
    torch.manual_seed(0)
    trainer = Trainer(TinyModel(), make_config(5), RegressionDataset())
    trainer.train()
    result = runner.invoke(app, ["inspect", trainer.checkpoint_manager.latest_checkpoint()])
    assert result.exit_code == 0, result.output
    assert "step / epoch    : 5" in result.output and "world size      : 1" in result.output


def test_inspect_rejects_foreign_file(tmp_path):
    path = tmp_path / "x.pt"
    torch.save({"weights": 1}, path)
    assert runner.invoke(app, ["inspect", str(path)]).exit_code == 1


def test_version():
    from flextrain import __version__

    assert runner.invoke(app, ["version"]).output.strip() == __version__


@pytest.mark.parametrize("cmd", [["--help"], ["launch", "--help"], ["serve", "--help"]])
def test_help(cmd):
    assert runner.invoke(app, cmd).exit_code == 0
