from __future__ import annotations

from argparse import Namespace
from io import StringIO
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace

from rich.console import Console

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_campaign.py"
_MODULE = run_path(str(_SCRIPT))
_parser = _MODULE["_parser"]
_verify_campaign = _MODULE["_verify_campaign"]


def test_parser_accepts_verify_and_domain_options(tmp_path: Path) -> None:
    domain = tmp_path / "domain.json"
    args = _parser().parse_args(["--verify", "--domain", str(domain)])

    assert args.verify is True
    assert args.domain == domain


def test_verification_skip_message_when_campaign_has_no_failures() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    outcome = SimpleNamespace(result=SimpleNamespace(episodes=[]), diagnoses={})
    args = Namespace()

    assert _verify_campaign(args, outcome, console) is None
    assert "verification skipped: campaign failures are absent" in output.getvalue()


def test_verification_skip_message_when_diagnoses_are_absent() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False)
    episode = SimpleNamespace(failures=[object()])
    outcome = SimpleNamespace(result=SimpleNamespace(episodes=[episode]), diagnoses={})
    args = Namespace()

    assert _verify_campaign(args, outcome, console) is None
    assert "verification skipped: campaign diagnoses are absent" in output.getvalue()
