"""
Result export: JSON (machine-readable), CSV (flat parameters),
and self-contained HTML reports.

All exports contain only measured/derived values. Missing values are
rendered as ``null``/empty, never fabricated.
"""

from __future__ import annotations

import base64
import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from prototype.core.exceptions import ExportError


def jsonify(value: Any) -> Any:
    """Recursively convert numpy/complex values into JSON-safe types."""
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonify(value.tolist())
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def export_json(data: dict[str, Any], path: str | Path | None = None) -> str:
    """Serialize a result dict to JSON; returns the JSON string and
    writes ``path`` when given."""
    payload = json.dumps(jsonify(data), indent=2, sort_keys=False, default=str)
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    return payload


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested dicts/lists into (dotted.key, scalar) rows."""
    rows: list[tuple[str, Any]] = []
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            if all(not isinstance(v, (dict, list, tuple)) for v in value):
                rows.append((name, ";".join(str(jsonify(v)) for v in value)))
            else:
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        rows.extend(_flatten(item, f"{name}[{index}]"))
                    else:
                        rows.append((f"{name}[{index}]", jsonify(item)))
        else:
            rows.append((name, jsonify(value)))
    return rows


def export_csv(data: dict[str, Any], path: str | Path | None = None) -> str:
    """Flatten a result dict into two-column CSV (parameter, value)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["parameter", "value"])
    for name, value in _flatten(data):
        writer.writerow([name, value])
    text = buffer.getvalue()
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


def _figure_to_base64(figure) -> str | None:
    """Render a matplotlib figure to a base64 PNG (for HTML embedding)."""
    try:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception:
        return None


def _html_table(rows: list[tuple[str, Any]]) -> str:
    if not rows:
        return "<p><em>No data recorded for this section.</em></p>"
    cells = "".join(
        f"<tr><td>{_escape(name)}</td><td>{_escape(str(value))}</td></tr>"
        for name, value in rows
    )
    return f"<table>{cells}</table>"


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


_SECTION_KEYS = [
    ("input", "Input"),
    ("detections", "Detections"),
    ("selected_candidate", "Selected Candidate"),
    ("isolation", "Isolation / DDC"),
    ("parameters", "Parameters"),
    ("symbol_rate", "Symbol Rate"),
    ("synchronization", "Synchronization"),
    ("classification", "Classification"),
    ("demodulation", "Demodulation"),
    ("protocol", "Protocol / Frame"),
    ("ber", "BER"),
    ("warnings", "Warnings"),
    ("provenance", "Provenance"),
]


def export_html(
    data: dict[str, Any],
    path: str | Path | None = None,
    figures: dict[str, Any] | None = None,
    title: str = "Spectra Analysis Report",
) -> str:
    """
    Render a self-contained HTML report.

    ``figures`` optionally maps section names ("spectrum",
    "constellation", ...) to matplotlib figures which are embedded as
    base64 PNGs.
    """
    data = jsonify(data) or {}
    sections_html: list[str] = []

    for key, heading in _SECTION_KEYS:
        value = data.get(key)
        if value is None:
            continue
        rows: list[tuple[str, Any]]
        if isinstance(value, dict):
            rows = _flatten(value)
        elif isinstance(value, list):
            if key == "warnings":
                rows = [(f"warning[{i}]", w) for i, w in enumerate(value)]
            elif all(isinstance(v, dict) for v in value):
                rows = []
                for index, item in enumerate(value):
                    rows.extend(_flatten(item, f"item[{index}]"))
            else:
                rows = [(f"{key}[{i}]", v) for i, v in enumerate(value)]
        else:
            rows = [(key, value)]
        sections_html.append(f"<h2>{_escape(heading)}</h2>")
        sections_html.append(_html_table(rows))

        if figures and key in figures:
            encoded = _figure_to_base64(figures[key])
            if encoded:
                sections_html.append(
                    f'<img class="fig" src="data:image/png;base64,{encoded}" '
                    f'alt="{_escape(key)}"/>'
                )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_escape(title)}</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         margin: 2rem auto; max-width: 960px; color: #1a1a2e; line-height: 1.45; }}
  h1 {{ border-bottom: 3px solid #16213e; padding-bottom: .3rem; }}
  h2 {{ color: #16213e; margin-top: 2rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
  td {{ border: 1px solid #d7d9e0; padding: .35rem .6rem; vertical-align: top; }}
  tr:nth-child(odd) {{ background: #f4f5fa; }}
  td:first-child {{ font-weight: 600; width: 38%; }}
  img.fig {{ max-width: 100%; margin: 1rem 0; border: 1px solid #d7d9e0; }}
  footer {{ margin-top: 3rem; font-size: .8rem; color: #667;
            border-top: 1px solid #d7d9e0; padding-top: .6rem; }}
</style>
</head>
<body>
<h1>{_escape(title)}</h1>
<p>Generated {generated} &#8212; all values are measured or derived from the
analyzed capture; missing values are omitted rather than estimated.</p>
{''.join(sections_html)}
<footer>Spectra RF signal analysis platform &#8212; engineering prototype,
synthetic-data validated. Not certified for operational/mission use.</footer>
</body>
</html>
"""
    if path is not None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
    return html
