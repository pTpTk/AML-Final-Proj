"""Verify original Football-Data snapshots, identities, dates and market quotes."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime, timezone
import time
import unicodedata
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .models import LABELS, ROOT

# Public Football-Data snapshots are obtained explicitly by step 00.
SOURCE_DIR = ROOT / "data" / "football_data_sources"
DOWNLOAD_MANIFEST = SOURCE_DIR / "download_manifest.csv"
RAW_PATH = ROOT / "data" / "Matches.csv"
CORRECTED_PATH = ROOT / "data" / "MatchesChronological.csv"
MARKET_COLUMNS = ["MarketHome", "MarketDraw", "MarketAway"]
RAW_COLUMNS = ["OddHome", "OddDraw", "OddAway"]
RAW_KEYS = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]
MATCH_KEYS = ["Division", "MatchDate", "HomeKey", "AwayKey"]
SUPPLEMENTAL = ["AUT", "SWE", "POL", "ROM", "IRL", "FIN"]
STANDARD_DIVISIONS = [
    "E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3",
    "F1", "F2", "D1", "D2", "I1", "I2", "SP1", "SP2", "B1", "N1", "P1", "T1", "G1",
]


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

def historical_source_urls(raw):
    """Request only league/season files represented in the original dataset."""
    urls = {}
    for code, group in raw.groupby("Division"):
        if code in STANDARD_DIVISIONS:
            years = group["MatchDate"].dt.year - group["MatchDate"].dt.month.lt(7).astype(int)
            for year in sorted(years.unique()):
                season = f"{year % 100:02d}{(year + 1) % 100:02d}"
                urls[f"{season}_{code}.csv"] = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        else:
            source_code = {"DEN": "DNK", "JAP": "JPN", "SUI": "SWZ"}.get(code, code)
            urls[f"{code}.csv"] = f"https://www.football-data.co.uk/new/{source_code}.csv"
    return urls

def download_history(urls, refresh=False):
    """Fetch required public CSVs and record URLs, hashes, sizes and retrieval status."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    def download(item):
        filename, url = item
        path = SOURCE_DIR / filename
        status = "Cached"
        if refresh or not path.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=40) as response:
                        body = response.read()
                    if not body or b"<html" in body[:500].lower():
                        raise ValueError(f"Expected a CSV from {url}")
                    temporary = path.with_suffix(".csv.part")
                    temporary.write_bytes(body)
                    temporary.replace(path)
                    status = "Downloaded"
                    break
                except (OSError, HTTPError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        body = path.read_bytes()
        return {"File": filename, "URL": url, "SHA256": hashlib.sha256(body).hexdigest(),
                "Bytes": len(body), "Status": status,
                "CheckedAtUTC": datetime.now(timezone.utc).isoformat()}

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Consume every result so failures never leave missing files unnoticed.
        futures = [(item, pool.submit(download, item)) for item in urls.items()]
        for position, ((filename, url), future) in enumerate(futures, start=1):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(filename)
                rows.append({"File": filename, "URL": url, "Status": "Failed", "Error": str(error)})
            if position % 40 == 0 or position == len(urls):
                print(f"Source files {position}/{len(urls)}; failures: {len(failures)}", flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DOWNLOAD_MANIFEST, index=False)
    if failures:
        raise RuntimeError(f"Could not obtain {len(failures)} required sources; see {DOWNLOAD_MANIFEST}")
    return manifest


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

def attach_market_odds(predictions, raw, sources, supplemental=SUPPLEMENTAL,
                       odds_tolerance=1e-9, verify_missing_odds_with_goals=False):
    """Audit source identities and quote provenance; return true dates for date repair."""
    # First attach the repository's odds solely for auditing supplemental matches.
    joined = predictions.merge(
        raw.rename(columns={"FTResult": "RawResult"}), on=RAW_KEYS,
        how="left", validate="one_to_one", sort=False, indicator="_RawMatch",
    )
    if joined["_RawMatch"].ne("both").any() or joined["RawResult"].fillna("").ne(joined["FTResult"].fillna("")).any():
        raise ValueError("The raw file no longer matches the saved prediction targets.")
    joined = joined.drop(columns="_RawMatch")
    joined = add_match_keys(joined)
    joined["EvaluationRow"] = np.arange(len(joined))
    candidates = sources.copy()
    candidates["MatchDate"] = candidates["SourceDate"]
    candidates["DateMatch"] = "Exact"

    # Some supplemental dates were transposed in Matches.csv. Build an audit
    # candidate for that known error, then require both team identities AND the
    # entire original market-average odds triplet to identify the actual game.
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
        matched[RAW_COLUMNS], matched[MARKET_COLUMNS], rtol=0, atol=odds_tolerance,
    ).all(axis=1)
    if verify_missing_odds_with_goals:
        # Awarded/cancelled matches can have no prices. Identify these using
        # both teams, the candidate date, the score and the outcome instead.
        missing_quotes = matched[RAW_COLUMNS].isna().all(axis=1)
        same_score = matched["FTHome"].eq(matched["SourceHomeGoals"]) & matched["FTAway"].eq(matched["SourceAwayGoals"])
        same_odds |= missing_quotes & same_score & matched["FTResult"].eq(matched["SourceResult"])
    matched = matched.loc[~matched["Division"].isin(supplemental) | same_odds]
    if matched["EvaluationRow"].duplicated().any():
        raise ValueError("Ambiguous source matches; refuse to choose between candidates.")
    source_columns = ["EvaluationRow", "SourceDate", "SourceResult", *MARKET_COLUMNS, "SourceFile", "SourceURL", "DateMatch"]
    if "MarketTiming" in matched:
        source_columns.append("MarketTiming")
    for field in ["SourceHomeGoals", "SourceAwayGoals"]:
        if field in matched:
            source_columns.append(field)
    joined = joined.merge(matched[source_columns], on="EvaluationRow", how="left", validate="one_to_one")
    valid = joined[MARKET_COLUMNS].gt(1).all(axis=1) & np.isfinite(joined[MARKET_COLUMNS]).all(axis=1)
    joined["Status"] = "Eligible"
    joined.loc[~valid, "Status"] = "Missing/invalid market odds"
    joined.loc[joined["SourceFile"].isna(), "Status"] = "No verified source match"
    joined.loc[joined["SourceResult"].notna() & joined["SourceResult"].ne(joined["FTResult"]), "Status"] = "Source result differs"
    invalid_result = ~joined["FTResult"].isin(LABELS) | ~joined["SourceResult"].isin(LABELS)
    joined.loc[invalid_result & joined["SourceFile"].notna(), "Status"] = "Missing/invalid result"
    joined["Eligible"] = joined["Status"].eq("Eligible")
    return joined


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
