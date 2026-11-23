"""Prepare supplied matches/odds, optionally verifying dates and results with public sources."""

import argparse
import hashlib
import json
import pandas as pd

from src.data_sources import (
    RAW_PATH, CORRECTED_PATH, RAW_KEYS, RAW_COLUMNS, MARKET_COLUMNS,
    STANDARD_DIVISIONS, historical_source_urls,
    load_history_sources, attach_market_odds, prepare_matches, DOWNLOAD_MANIFEST,
)
from src.models import REPORT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-only", action="store_true", help="Use Matches.csv alone; skip external date/result verification.")
    arguments = parser.parse_args()
    raw = pd.read_csv(RAW_PATH, parse_dates=["MatchDate"], dtype={"MatchTime": "str"})
    REPORT_DIR.mkdir(exist_ok=True)
    metadata = {"RawInput": "data/Matches.csv", "RawSHA256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
                "Rows": len(raw), "OddsInput": "Matches.csv: OddHome/OddDraw/OddAway",
                "ExternalOddsUsed": False}
    if arguments.raw_only:
        prepared = prepare_matches(raw)
        prepared.to_csv(CORRECTED_PATH, index=False)
        metadata.update(Mode="Raw only", DatesChanged=0, MissingOrInvalidTargets=int(prepared["FTResult"].isna().sum()))
        (REPORT_DIR / "data_preparation.json").write_text(json.dumps(metadata, indent=2))
        print(f"Saved {len(prepared):,} rows using Matches.csv alone; dates and supplied odds preserved.")
        return
    if not DOWNLOAD_MANIFEST.exists():
        raise FileNotFoundError("Run 00_download_sources.py first, or use --raw-only to skip external verification.")
    urls = historical_source_urls(raw)
    print(f"Preparing {len(raw):,} matches using {len(urls)} source files.", flush=True)
    sources, source_audit = load_history_sources(urls)
    supplemental = sorted(set(raw["Division"]) - set(STANDARD_DIVISIONS))
    matching = raw.copy()
    # Two Portuguese fixtures have amended dates in the original source files.
    # The explicit identities make these small source corrections reviewable.
    for old_date, new_date, away in [
        ("2020-02-01", "2020-02-02", "Pacos Ferreira"),
        ("2021-01-03", "2021-01-04", "Benfica"),
    ]:
        mask = matching["Division"].eq("P1") & matching["HomeTeam"].eq("Santa Clara")
        mask &= matching["AwayTeam"].eq(away) & matching["MatchDate"].eq(old_date)
        matching.loc[mask, "MatchDate"] = pd.Timestamp(new_date)
    matched = attach_market_odds(
        matching[RAW_KEYS + ["FTResult"]], matching[RAW_KEYS + ["FTResult", "FTHome", "FTAway", *RAW_COLUMNS]],
        sources, supplemental=supplemental, odds_tolerance=0.0050001,
        verify_missing_odds_with_goals=True,
    )
    amended = matching["MatchDate"].ne(raw["MatchDate"])
    matched.loc[amended, "DateMatch"] = "Source date amended"
    matched["MatchDate"] = raw["MatchDate"]
    REPORT_DIR.mkdir(exist_ok=True)
    source_audit.to_csv(REPORT_DIR / "data_cleaning_sources.csv", index=False)
    matched.to_csv(REPORT_DIR / "data_cleaning_source_matches.csv", index=False)
    unmatched = matched.loc[matched["SourceDate"].isna()]
    if not unmatched.empty:
        print(unmatched[RAW_KEYS].head(30).to_string(index=False), flush=True)
        raise ValueError(f"{len(unmatched)} matches could not be verified against their source; see source audit.")
    prepared = prepare_matches(raw, matched)
    prepared.to_csv(CORRECTED_PATH, index=False)
    changed = int(matched["SourceDate"].ne(raw["MatchDate"]).sum())
    excluded = int(prepared["FTResult"].isna().sum())
    metadata.update(Mode="Source-verified dates/results; supplied odds", DatesChanged=changed,
                    MissingOrInvalidTargets=excluded, SourceManifest=str(DOWNLOAD_MANIFEST.relative_to(RAW_PATH.parents[1])))
    (REPORT_DIR / "data_preparation.json").write_text(json.dumps(metadata, indent=2))
    print(f"Corrected {changed:,} dates; excluded {excluded:,} missing/invalid or conflicting targets.")
    print(f"Kept original odds; no external quote filled or replaced a supplied value. Saved {CORRECTED_PATH}.")



if __name__ == "__main__":
    main()
