"""Regression tests for health_snapshot_ticket.py — TICKET-20260609-006.

Covers the structural fixes that ended the recurring false-positive
burst ("[openclaw] the cli command failed." from a 12d-stale
gateway.err.log, 7 instances in 11h, all closed at 10:46Z):

  Fix A (parser): extract_gateway_err_signatures now requires a
    parseable AND in-window timestamp before a line contributes a
    signature. Lines without timestamps are dropped as noise from a
    stale log file.

  Fix B (live-gate): _is_live_healthy() calls `openclaw status` and
    suppresses ticket creation when the gateway reports `state active`,
    even if the parser produced signatures from a stale log.

These tests verify the *contract*, not the implementation, by running
`main()` end-to-end with monkeypatched inputs. They protect against
future regressions of either fix.

Acceptance criteria from the ticket:
  1. 0 false-positive tickets when gateway is healthy + log is stale
  3. Real failures (gateway genuinely down) are still detected
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

# Prevent pytest's argv from leaking into argparse inside hst.main()
sys.argv = ["health_snapshot_ticket.py"]

# Make the script importable as a module
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import health_snapshot_ticket as hst  # noqa: E402


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    """Redirect all log paths to tmp_path so we don't touch real logs."""
    logs = tmp_path / "logs"
    ops = tmp_path / "ops"
    logs.mkdir()
    (ops / "ci").mkdir(parents=True)
    monkeypatch.setattr(hst, "LOGS_DIR", logs)
    monkeypatch.setattr(hst, "OPS_DIR", ops)
    monkeypatch.setattr(hst, "GATEWAY_ERR", logs / "gateway.err.log")
    monkeypatch.setattr(hst, "ERRORS_JSONL", logs / "errors.jsonl")
    monkeypatch.setattr(hst, "HEALTH_JSONL", logs / "health.jsonl")
    monkeypatch.setattr(hst, "CI_LOG", ops / "ci" / "ci-log.jsonl")
    monkeypatch.setattr(hst, "TICKETS_MD", ops / "TICKET-TRACKER.md")
    # Minimal tracker so append_ticket can find the file
    (ops / "TICKET-TRACKER.md").write_text(
        "# TICKET TRACKER\n\n## Active Tickets\n", encoding="utf-8"
    )
    return tmp_path


def _now_utc():
    return datetime.now(timezone.utc)


def _recent_iso(offset_seconds: int = 0) -> str:
    return (_now_utc() + timedelta(seconds=offset_seconds)).isoformat(timespec="seconds")


# ─── Fix A: parser-level timestamp gate ────────────────────────────────


class TestParserLevelTimestampGate:
    """Fix A: a stale log line without a parseable ISO timestamp must
    NOT contribute a signature, even if it contains 'error'."""

    def test_stale_line_without_timestamp_is_dropped(self, isolated_logs):
        """The historical false-positive: 'head -1' of a 12d-stale log
        with no leading ISO timestamp must produce zero signatures."""
        hst.GATEWAY_ERR.write_text(
            "[openclaw] the cli command failed.\n", encoding="utf-8"
        )
        sigs = hst.extract_gateway_err_signatures(
            hst.GATEWAY_ERR, _now_utc() - timedelta(hours=24)
        )
        assert sigs == [], (
            f"Stale timestamp-less line should be dropped, got: {sigs!r}"
        )

    def test_recent_line_with_timestamp_is_kept(self, isolated_logs):
        ts = _recent_iso(-60)  # 1 min ago
        hst.GATEWAY_ERR.write_text(
            f"{ts} gateway error: connection refused\n", encoding="utf-8"
        )
        sigs = hst.extract_gateway_err_signatures(
            hst.GATEWAY_ERR, _now_utc() - timedelta(hours=24)
        )
        assert len(sigs) == 1
        assert "connection refused" in sigs[0]

    def test_old_line_with_parseable_but_out_of_window_timestamp_dropped(
        self, isolated_logs
    ):
        old = (_now_utc() - timedelta(days=12)).isoformat(timespec="seconds")
        hst.GATEWAY_ERR.write_text(
            f"{old} gateway error: ancient failure\n", encoding="utf-8"
        )
        sigs = hst.extract_gateway_err_signatures(
            hst.GATEWAY_ERR, _now_utc() - timedelta(hours=24)
        )
        assert sigs == [], (
            "Old-but-parseable timestamped line should be filtered by window"
        )


# ─── Fix B: live-status gate ───────────────────────────────────────────


class TestLiveStatusGate:
    """Fix B: when `openclaw status` reports 'state active', ticket
    creation must be suppressed — even if the parser produced signatures.
    This is the criterion-1 / criterion-4 contract."""

    def test_healthy_status_suppresses_even_when_sigs_exist(self, isolated_logs, monkeypatch, capsys):
        # Produce a stale timestamp-less line (the historical false-positive
        # signature) — without the live gate, this would write a ticket.
        hst.GATEWAY_ERR.write_text(
            "[openclaw] the cli command failed.\n", encoding="utf-8"
        )

        # Force _is_live_healthy=True to simulate a healthy gateway.
        monkeypatch.setattr(hst, "_is_live_healthy", lambda: True)

        rc = hst.main()
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out == "NO_REPLY", f"Expected NO_REPLY, got: {out!r}"
        # Crucially: no ticket was written.
        assert "OPEN" not in hst.TICKETS_MD.read_text() or hst.TICKETS_MD.read_text().count("**Status:** OPEN") == 0

    def test_unhealthy_status_still_pages(self, isolated_logs, monkeypatch, capsys):
        """Criterion 3: if the gateway is genuinely down, we must still
        open a ticket. Simulate by feeding errors.jsonl with in-window
        signatures AND forcing _is_live_healthy=False."""
        recent = _recent_iso(-30)
        err_entry = {
            "timestamp": recent,
            "error": {"message": "gateway down: connection refused on port 18789"},
        }
        with hst.ERRORS_JSONL.open("w", encoding="utf-8") as f:
            # Threshold is 3; emit 4 to ensure it qualifies.
            for _ in range(4):
                f.write(json.dumps(err_entry) + "\n")

        monkeypatch.setattr(hst, "_is_live_healthy", lambda: False)

        rc = hst.main()
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out.startswith("TICKETS_OPENED:"), (
            f"Expected TICKETS_OPENED, got: {out!r}"
        )
        # Verify the ticket was actually appended.
        tracker = hst.TICKETS_MD.read_text(encoding="utf-8")
        assert "gateway down: connection refused" in tracker
        assert "**Status:** OPEN" in tracker

    def test_healthy_status_with_no_sigs_noop(self, isolated_logs, monkeypatch, capsys):
        monkeypatch.setattr(hst, "_is_live_healthy", lambda: True)
        rc = hst.main()
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out == "NO_REPLY"


# ─── _is_live_healthy unit tests ──────────────────────────────────────


class TestIsLiveHealthy:
    """Direct unit tests for the live-status helper."""

    def test_recognises_state_active_in_stdout(self, monkeypatch):
        class _R:
            stdout = "Gateway: state active\n"
            stderr = ""
        monkeypatch.setattr(hst.subprocess, "run", lambda *a, **k: _R())
        assert hst._is_live_healthy() is True

    def test_recognises_state_active_in_stderr(self, monkeypatch):
        class _R:
            stdout = ""
            stderr = "warning\nstate active\n"
        monkeypatch.setattr(hst.subprocess, "run", lambda *a, **k: _R())
        assert hst._is_live_healthy() is True

    def test_missing_marker_returns_false(self, monkeypatch):
        class _R:
            stdout = "Gateway: state stopped\n"
            stderr = ""
        monkeypatch.setattr(hst.subprocess, "run", lambda *a, **k: _R())
        assert hst._is_live_healthy() is False

    def test_timeout_returns_false_does_not_raise(self, monkeypatch):
        def _boom(*a, **k):
            raise hst.subprocess.TimeoutExpired(cmd="openclaw", timeout=3)
        monkeypatch.setattr(hst.subprocess, "run", _boom)
        assert hst._is_live_healthy() is False

    def test_filenotfound_returns_false_does_not_raise(self, monkeypatch):
        def _boom(*a, **k):
            raise FileNotFoundError("openclaw")
        monkeypatch.setattr(hst.subprocess, "run", _boom)
        assert hst._is_live_healthy() is False


# ─── Build verification: 7 historical false-positive instances ─────────


class TestHistoricalFalsePositives:
    """Build verification per the ticket: replay the 7 historical
    false-positive signatures against the *current* detector and assert
    all 7 are now suppressed (criterion 1, 4)."""

    HISTORICAL_SIGS = [
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
        "[openclaw] the cli command failed.",
    ]

    def test_all_seven_historical_false_positives_suppressed(self, isolated_logs, monkeypatch, capsys):
        # Write the stale log line that the historical detector mishandled.
        hst.GATEWAY_ERR.write_text(
            "\n".join(self.HISTORICAL_SIGS) + "\n", encoding="utf-8"
        )
        # Healthy gateway → must suppress.
        monkeypatch.setattr(hst, "_is_live_healthy", lambda: True)

        rc = hst.main()
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert out == "NO_REPLY", (
            f"All 7 historical false-positive instances must be suppressed. Got: {out!r}"
        )
        tracker = hst.TICKETS_MD.read_text(encoding="utf-8")
        assert "**Status:** OPEN" not in tracker
