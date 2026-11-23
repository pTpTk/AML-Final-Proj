"""Small report helpers shared by the numbered analysis steps."""

import json

from .models import REPORT_DIR, markdown_table


def write_report(filename, title, introduction, tables):
    REPORT_DIR.mkdir(exist_ok=True)
    parts = [f"# {title}", "", introduction, ""]
    for heading, table in tables:
        parts.extend([f"## {heading}", "", markdown_table(table), ""])
    (REPORT_DIR / filename).write_text("\n".join(parts), encoding="utf-8")


def all_periods(metrics, scope=None, window="Common"):
    mask = metrics["Period"].eq("All") & metrics["Window"].eq(window)
    if scope is not None:
        mask &= metrics["Scope"].eq(scope)
    return metrics.loc[mask]


def preparation_note():
    path = REPORT_DIR / "data_preparation.json"
    if not path.exists():
        return "Run step 01 to record preparation provenance."
    metadata = json.loads(path.read_text())
    return (f"Preparation: {metadata['Mode']}. Odds inputs: {metadata['OddsInput']}. "
            f"Dates changed: {metadata['DatesChanged']:,}. Missing/invalid targets: "
            f"{metadata['MissingOrInvalidTargets']:,}.")
