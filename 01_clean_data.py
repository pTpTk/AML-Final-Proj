"""Prepare supplied matches/odds, optionally verifying dates and results with public sources."""

from importlib import import_module
import argparse
import hashlib
import json
import unicodedata

import numpy as np
import pandas as pd


_step_00 = import_module('00_download_sources')
DOWNLOAD_MANIFEST = _step_00.DOWNLOAD_MANIFEST
RAW_PATH = _step_00.RAW_PATH
ROOT = _step_00.ROOT
SOURCE_DIR = _step_00.SOURCE_DIR
STANDARD_DIVISIONS = _step_00.STANDARD_DIVISIONS
historical_source_urls = _step_00.historical_source_urls
REPORT_DIR = ROOT / "reports"
LABELS = ["A", "D", "H"]  # Probability columns and confusion matrices use this order.
CORRECTED_PATH = ROOT / "data" / "MatchesChronological.csv"
MARKET_COLUMNS = ["MarketHome", "MarketDraw", "MarketAway"]
RAW_COLUMNS = ["OddHome", "OddDraw", "OddAway"]
RAW_KEYS = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]
MATCH_KEYS = ["Division", "MatchDate", "HomeKey", "AwayKey"]


def team_key(name):
    """Ignore accents, punctuation and case; never guess unrelated team names."""
    name = unicodedata.normalize("NFKD", name).casefold()
    key = "".join(character for character in name if character.isalnum())
    # Football-Data renamed the same Mexican club in its historical source.
    return {"uanltigres": "tigresuanl"}.get(key, key)


def add_match_keys(table):
    """Preserve original names while adding consistent keys for exact matching."""
    table = table.copy()
    table["HomeKey"] = table["HomeTeam"].map(team_key)
    table["AwayKey"] = table["AwayTeam"].map(team_key)
    return table


def choose_average_odds(table):
    """Prefer closing averages; use documented earlier averages when closing is absent."""
    chosen = pd.DataFrame(np.nan, index=table.index, columns=MARKET_COLUMNS)
    timing = pd.Series("Unavailable", index=table.index)
    # Older source files do not publish closing prices. Never substitute an
    # individual bookmaker or the maximum price for a genuine market average.
    for columns, name in [
        (["AvgCH", "AvgCD", "AvgCA"], "Closing"),
        (["AvgH", "AvgD", "AvgA"], "Pre-closing"),
        (["BbAvH", "BbAvD", "BbAvA"], "Legacy pre-closing"),
    ]:
        if not set(columns).issubset(table.columns):
            continue
        odds = table[columns].apply(pd.to_numeric, errors="coerce")
        valid = odds.gt(1).all(axis=1) & np.isfinite(odds).all(axis=1) & timing.eq("Unavailable")
        chosen.loc[valid] = odds.loc[valid].to_numpy()
        timing.loc[valid] = name
    chosen["MarketTiming"] = timing
    return chosen


def load_history_sources(urls):
    """Read exact source identities, true dates, and market averages for all seasons."""
    frames, audit = [], []
    average_columns = [f"{prefix}{result}" for prefix in ["AvgC", "Avg", "BbAv"] for result in ["H", "D", "A"]]
    identity = ["Date", "HomeTeam", "AwayTeam", "FTR", "Home", "Away", "Res", "FTHG", "FTAG", "HG", "AG"]
    for filename, url in urls.items():
        path = SOURCE_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run 00_download_sources.py first.")
        try:
            table = pd.read_csv(path, usecols=lambda field: field in identity + average_columns)
        except UnicodeDecodeError:
            table = pd.read_csv(path, usecols=lambda field: field in identity + average_columns, encoding="latin1")
        table = table.rename(columns={
            "Home": "HomeTeam", "Away": "AwayTeam", "FTR": "SourceResult", "Res": "SourceResult",
            "FTHG": "SourceHomeGoals", "FTAG": "SourceAwayGoals", "HG": "SourceHomeGoals", "AG": "SourceAwayGoals",
        })
        table["SourceDate"] = pd.to_datetime(table["Date"], format="mixed", dayfirst=True, errors="coerce")
        table = table.loc[table["SourceDate"].notna() & table["HomeTeam"].notna() & table["AwayTeam"].notna()].copy()
        averages = choose_average_odds(table)
        table = pd.concat([table, averages], axis=1)
        table["Division"] = path.stem.split("_")[-1]
        table["SourceFile"], table["SourceURL"] = filename, url
        audit.append({
            "File": filename, "URL": url, "SHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "Rows": len(table), "WithMarketOdds": int(table["MarketTiming"].ne("Unavailable").sum()),
        })
        columns = ["Division", "SourceDate", "HomeKey", "AwayKey", "SourceResult", "SourceHomeGoals", "SourceAwayGoals", *MARKET_COLUMNS, "MarketTiming", "SourceFile", "SourceURL"]
        frames.append(add_match_keys(table)[columns])
    sources = pd.concat(frames, ignore_index=True)
    keys = ["Division", "SourceDate", "HomeKey", "AwayKey"]
    if sources.duplicated(keys).any():
        raise ValueError("Duplicate source match identities; inspect before preparing training data.")
    return sources, pd.DataFrame(audit)


def match_source_records(raw, sources):
    """Match raw fixtures to verification records, without replacing supplied odds.

    Supplemental divisions require all three prices to agree within rounding,
    or an agreeing score/result when every supplied price is missing. Return
    source evidence even when a result disagrees; prepare_matches excludes that
    target. Unmatched records have a missing SourceDate; ambiguity is an error.
    """
    supplemental = set(raw["Division"]) - set(STANDARD_DIVISIONS)
    joined = add_match_keys(raw)
    joined["RawRow"] = np.arange(len(joined))
    candidates = sources.copy()
    candidates["MatchDate"] = candidates["SourceDate"]
    candidates["DateMatch"] = "Exact"

    # Some supplemental dates were transposed in Matches.csv. Build an audit
    # candidate for that known error, then require both team identities AND the
    # entire supplied odds triplet to identify the actual game.
    # Step 01 uses the returned SourceDate to rebuild chronology before features.
    transposed = candidates.loc[
        candidates["Division"].isin(supplemental)
        & candidates["SourceDate"].dt.day.le(12)
        & candidates["SourceDate"].dt.day.ne(candidates["SourceDate"].dt.month)
    ].copy()
    transposed["MatchDate"] = pd.to_datetime({
        "year": transposed["SourceDate"].dt.year,
        "month": transposed["SourceDate"].dt.day,
        "day": transposed["SourceDate"].dt.month,
    })
    transposed["DateMatch"] = "Transposed; odds verified"
    candidates = pd.concat([candidates, transposed], ignore_index=True)
    matched = joined.merge(candidates, on=MATCH_KEYS, how="inner", validate="many_to_many")
    same_odds = np.isclose(
        matched[RAW_COLUMNS], matched[MARKET_COLUMNS], rtol=0, atol=0.0050001,
    ).all(axis=1)
    # Awarded/cancelled matches can have no prices. Identify these using
    # both teams, the candidate date, the score and the outcome instead.
    missing_quotes = matched[RAW_COLUMNS].isna().all(axis=1)
    same_score = matched["FTHome"].eq(matched["SourceHomeGoals"]) & matched["FTAway"].eq(matched["SourceAwayGoals"])
    same_odds |= missing_quotes & same_score & matched["FTResult"].eq(matched["SourceResult"])
    matched = matched.loc[~matched["Division"].isin(supplemental) | same_odds]
    if matched["RawRow"].duplicated().any():
        raise ValueError("Ambiguous source matches; refuse to choose between candidates.")
    source_columns = ["RawRow", "SourceDate", "SourceResult", *MARKET_COLUMNS, "SourceFile", "SourceURL", "DateMatch"]
    if "MarketTiming" in matched:
        source_columns.append("MarketTiming")
    for field in ["SourceHomeGoals", "SourceAwayGoals"]:
        if field in matched:
            source_columns.append(field)
    return joined.merge(matched[source_columns], on="RawRow", how="left", validate="one_to_one")


def prepare_matches(raw, source_matches=None):
    """Use supplied odds; optionally verify dates/results against explicit source matches."""
    prepared = raw.copy()
    prepared["MatchDate"] = pd.to_datetime(prepared["MatchDate"], errors="coerce")
    if prepared[RAW_KEYS].isna().any().any():
        raise ValueError("Missing/invalid match identity or date; inspect the raw-data audit.")
    known = prepared["FTResult"].isin(LABELS)
    if source_matches is not None:
        if len(source_matches) != len(prepared) or source_matches["SourceDate"].isna().any():
            raise ValueError("Every raw row needs one verified source match for date repair.")
        prepared["MatchDate"] = pd.to_datetime(source_matches["SourceDate"].to_numpy())
        prepared["SourceResult"] = source_matches["SourceResult"].to_numpy()
        known &= prepared["FTResult"].eq(prepared["SourceResult"])
    prepared.loc[~known, "FTResult"] = pd.NA
    # These are exactly the supplied prices: external quotes never fill or replace them.
    odds = prepared[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = odds.gt(1).all(axis=1) & np.isfinite(odds).all(axis=1)
    prepared[MARKET_COLUMNS] = odds.where(valid, np.nan).to_numpy()
    prepared["MarketTiming"] = "Supplied; timing unverified"
    if prepared.duplicated(RAW_KEYS).any():
        raise ValueError("Duplicate match identities; inspect the audit before building histories.")
    return prepared


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
    matched = match_source_records(
        matching[RAW_KEYS + ["FTResult", "FTHome", "FTAway", *RAW_COLUMNS]], sources,
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
