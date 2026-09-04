from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


class OutputWriteError(PermissionError):
    """Raised when result files cannot be written (e.g. open in Excel)."""


def _permission_hint(path: Path) -> str:
    return (
        f"Konnte '{path}' nicht schreiben.\n"
        "Haeufige Ursache unter Windows: Die Datei ist in Excel, im Editor "
        "oder in einer Vorschau geoeffnet.\n"
        "Loesung: Tab/Programm schliessen und erneut starten, oder ein "
        "anderes Ausgabeverzeichnis nutzen:\n"
        "  python run_client_day_log_analysis.py --run-id 0003 "
        "--output-dir data/client_day_log_evaluations_neu"
    )


def _write_with_retry(
    path: Path,
    write_fn,
    *,
    max_retries: int = 5,
    retry_delay_s: float = 0.4,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")

    last_error: PermissionError | None = None
    for attempt in range(max_retries):
        try:
            write_fn(tmp_path)
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            if attempt < max_retries - 1:
                time.sleep(retry_delay_s)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 32 or isinstance(exc, PermissionError):
                last_error = PermissionError(exc.errno, str(exc))
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_s)
                continue
            raise

    raise OutputWriteError(_permission_hint(path)) from last_error


def safe_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    def _write(target: Path) -> None:
        df.to_csv(target, **kwargs)

    _write_with_retry(path, _write)


def safe_write_json(path: Path, payload: Any, **kwargs) -> None:
    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, **kwargs)

    _write_with_retry(path, _write)
