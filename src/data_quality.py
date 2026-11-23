"""Measure missingness, invalid values and external-cleaning changes without guessing repairs."""

import numpy as np
import pandas as pd

from .data_sources import RAW_KEYS, RAW_COLUMNS, MARKET_COLUMNS
from .features import STAT_COLUMNS
from .models import LABELS, TRAIN_START, TEST_START, TEST_END

DETAIL_COLUMNS = [f"{side}{stat}" for side in ["Home", "Away"]
                  for stat in ["Shots", "Target", "Fouls", "Corners", "Yellow", "Red"]]
COUNT_COLUMNS = [column for pair in STAT_COLUMNS.values() for column in pair]
ODDS_COLUMNS = RAW_COLUMNS + ["MaxHome", "MaxDraw", "MaxAway", "Over25", "Under25",
                              "MaxOver25", "MaxUnder25", "HandiHome", "HandiAway"]


def valid_odds(table, columns=RAW_COLUMNS):
    odds = table[columns].apply(pd.to_numeric, errors="coerce")
    return odds.gt(1).all(axis=1) & np.isfinite(odds).all(axis=1)


def audit_raw(raw):
    """Count missing cells separately from invalid observed values in every column."""
    fields = []
    for column in raw:
        values = raw[column]
        invalid = pd.Series(False, index=raw.index)
        rule = "No additional domain constraint"
        if column in ["MatchDate", "MatchTime"]:
            date_format = "%Y-%m-%d" if column == "MatchDate" else "%H:%M:%S"
            invalid = values.notna() & pd.to_datetime(values, format=date_format, errors="coerce").isna()
            rule = date_format
        elif column in ["FTResult", "HTResult"]:
            invalid = values.notna() & ~values.isin(LABELS)
            rule = "H, D or A"
        elif column in ["Division", "HomeTeam", "AwayTeam"]:
            invalid = values.notna() & values.astype("string").str.strip().eq("")
            rule = "Nonempty identifier"
        elif pd.api.types.is_numeric_dtype(values) or column in COUNT_COLUMNS + ODDS_COLUMNS + ["HomeElo", "AwayElo"]:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & (numeric.isna() | ~np.isfinite(numeric))
            rule = "Finite numeric value"
            if column in COUNT_COLUMNS:
                invalid |= numeric.notna() & (numeric.lt(0) | numeric.mod(1).ne(0))
                rule = "Finite nonnegative integer"
            elif column in ODDS_COLUMNS:
                invalid |= numeric.notna() & numeric.le(1)
                rule = "Finite decimal odds > 1"
            elif column in ["HomeElo", "AwayElo"]:
                invalid |= numeric.notna() & numeric.le(0)
                rule = "Finite Elo > 0"
        fields.append({"Field": column, "Rows": len(raw), "Missing": int(values.isna().sum()),
                       "MissingPercent": 100 * values.isna().mean(), "InvalidObserved": int(invalid.sum()),
                       "Rule": rule})
    dates = pd.to_datetime(raw["MatchDate"], errors="coerce")
    goals_known = raw[["FTHome", "FTAway"]].notna().all(axis=1)
    goal_result = pd.Series(np.select([raw.FTHome.gt(raw.FTAway), raw.FTHome.lt(raw.FTAway)],
                                     ["H", "A"], default="D"), index=raw.index)
    masks = {
        "Missing/invalid full-time result": ~raw.FTResult.isin(LABELS),
        "Missing at least one full-time goal count": ~goals_known,
        "Missing at least one Elo": raw[["HomeElo", "AwayElo"]].isna().any(axis=1),
        "Missing both Elo values": raw[["HomeElo", "AwayElo"]].isna().all(axis=1),
        "Missing at least one half-time goal count": raw[["HTHome", "HTAway"]].isna().any(axis=1),
        "Missing at least one detailed statistic": raw[DETAIL_COLUMNS].isna().any(axis=1),
        "Missing all detailed statistics": raw[DETAIL_COLUMNS].isna().all(axis=1),
        "Missing/incomplete odds triplet": raw[RAW_COLUMNS].isna().any(axis=1),
        "Complete but invalid odds triplet": raw[RAW_COLUMNS].notna().all(axis=1) & ~valid_odds(raw),
        "Missing kickoff time": raw.MatchTime.isna(),
        "Unparseable date": dates.isna(),
        "Duplicate full row (after first)": raw.duplicated(),
        "Duplicate match identity (after first)": raw.duplicated(RAW_KEYS),
        "Same home and away team": raw.HomeTeam.eq(raw.AwayTeam),
        "Known result disagrees with supplied goals": raw.FTResult.isin(LABELS) & goals_known & raw.FTResult.ne(goal_result),
        "Half-time goals exceed full-time goals": raw.HTHome.gt(raw.FTHome) | raw.HTAway.gt(raw.FTAway),
    }
    summary = pd.DataFrame([{"Check": check, "Matches": int(mask.sum()), "Percent": 100 * mask.mean()}
                            for check, mask in masks.items()])
    by_league = pd.DataFrame({"Division": raw.Division, **{name: mask.astype(int) for name, mask in masks.items()}})
    grouped = by_league.groupby("Division").sum()
    grouped.insert(0, "Matches", raw.groupby("Division").size())
    return pd.DataFrame(fields), summary, grouped.reset_index()


def audit_external(raw, audit):
    """Count repairs separately from missing data, aligned by the original row identity."""
    if len(raw) != len(audit):
        raise ValueError("External audit must contain every original row.")
    raw_dates = pd.to_datetime(raw.MatchDate)
    source_dates = pd.to_datetime(audit.SourceDate)
    identities = raw[RAW_KEYS].copy()
    identities["MatchDate"] = raw_dates
    pd.testing.assert_frame_equal(identities.reset_index(drop=True),
                                  audit[RAW_KEYS].assign(MatchDate=pd.to_datetime(audit.MatchDate)).reset_index(drop=True))
    raw_train = raw_dates.ge(TRAIN_START) & raw_dates.lt(TEST_START)
    source_train = source_dates.ge(TRAIN_START) & source_dates.lt(TEST_START)
    raw_test = raw_dates.ge(TEST_START) & raw_dates.lt(TEST_END)
    source_test = source_dates.ge(TEST_START) & source_dates.lt(TEST_END)
    raw_valid = valid_odds(raw)
    source_valid = valid_odds(audit, MARKET_COLUMNS)
    masks = {
        "Date differs from external source": raw_dates.ne(source_dates),
        "Date change moves match into/out of training": raw_train.ne(source_train),
        "In raw training only": raw_train & ~source_train,
        "In source-date training only": source_train & ~raw_train,
        "Date change moves match into/out of test": raw_test.ne(source_test),
        "Known result disagrees with source": raw.FTResult.isin(LABELS) & audit.SourceResult.isin(LABELS)
                                                   & raw.FTResult.ne(audit.SourceResult),
        "Missing raw odds but usable external averages": ~raw_valid & source_valid,
        "Usable raw odds but no usable external averages": raw_valid & ~source_valid,
        "No usable external market averages": ~source_valid,
    }
    summary = pd.DataFrame([{"Check": check, "Matches": int(mask.sum()), "Percent": 100 * mask.mean()}
                            for check, mask in masks.items()])
    grouped = pd.DataFrame({"Division": raw.Division, **{name: mask.astype(int) for name, mask in masks.items()}}).groupby("Division").sum()
    grouped.insert(0, "Matches", raw.groupby("Division").size())
    return summary, grouped.reset_index()


def paired_test_mask(table, end):
    """Compare identical games only when both recorded and verified dates are held out."""
    return (table["RawDate"].ge(TEST_START) & table["RawDate"].le(end)
            & table["SourceDate"].ge(TEST_START) & table["SourceDate"].le(end)
            & table["FTResult"].isin(LABELS) & table["FTResult"].eq(table["SourceResult"]))
