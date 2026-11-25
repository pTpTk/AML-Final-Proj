"""Compare the market favourite and frozen classifiers on exactly the same matches."""

from importlib import import_module

import numpy as np
import pandas as pd


_step_01 = import_module('01_clean_data')
CORRECTED_PATH = _step_01.CORRECTED_PATH
LABELS = _step_01.LABELS
MARKET_COLUMNS = _step_01.MARKET_COLUMNS
RAW_KEYS = _step_01.RAW_KEYS
REPORT_DIR = _step_01.REPORT_DIR

_step_03 = import_module('03_train_models')

_step_04 = import_module('04_evaluate_leagues')
MODEL_NAMES = _step_04.MODEL_NAMES
prediction_scores = _step_04.prediction_scores


def add_odds_predictions(matches, odds_columns, benchmark_name):
    """Use H/D/A odds to choose a favourite and calculate normalized probabilities."""
    matches = matches.copy()
    odds = matches[odds_columns].to_numpy(dtype=float)
    if not np.isfinite(odds).all() or not (odds > 1).all():
        raise ValueError("All three decimal odds must be finite and greater than one.")
    slug = benchmark_name.lower().replace(" ", "_")
    # Columns are explicitly H/D/A. argmin breaks equal-price ties in that order.
    matches[f"{slug}_prediction"] = np.array(["H", "D", "A"])[odds.argmin(axis=1)]
    matches["TiedFavourite"] = (odds == odds.min(axis=1, keepdims=True)).sum(axis=1) > 1
    inverse_odds = 1 / odds
    probabilities = inverse_odds / inverse_odds.sum(axis=1, keepdims=True)
    for label in LABELS:
        matches[f"{slug}_probability_{label}"] = probabilities[:, ["H", "D", "A"].index(label)]
    matches["OddsOverround"] = inverse_odds.sum(axis=1) - 1
    return matches


def score_comparisons(matches, benchmark_name):
    """Evaluate every model on the same rows, with tier, league and year breakdowns."""
    benchmark_slug = benchmark_name.lower().replace(" ", "_")
    rows = []
    for window in ["Common", "Available"]:
        window_matches = matches.loc[matches["CommonWindow"]] if window == "Common" else matches
        for period in ["All", *sorted(window_matches["MatchDate"].dt.year.unique().astype(str))]:
            selected = window_matches if period == "All" else window_matches.loc[
                window_matches["MatchDate"].dt.year.eq(int(period))
            ]
            groups = [("Combined", "All", selected)]
            groups += [("Tier", tier, group) for tier, group in selected.groupby("Tier")]
            groups += [("League", code, group) for code, group in selected.groupby("Division")]
            for scope, group_name, group in groups:
                if group.empty:
                    continue
                benchmark_correct = group[f"{benchmark_slug}_prediction"].eq(group["FTResult"])
                unique = ~group["TiedFavourite"]
                for name in [benchmark_name, *MODEL_NAMES]:
                    slug = name.lower().replace(" ", "_")
                    prediction = group[f"{slug}_prediction"]
                    scores = prediction_scores(group["FTResult"], prediction, group[
                        [f"{slug}_probability_{label}" for label in LABELS]
                    ])
                    correct = prediction.eq(group["FTResult"])
                    rows.append({
                        "Window": window, "Period": period, "Scope": scope, "Group": group_name,
                        "Model": name, "Matches": len(group), **scores,
                        "GapToBenchmark": correct.mean() - benchmark_correct.mean(),
                        "ModelOnlyCorrect": int((correct & ~benchmark_correct).sum()),
                        "BenchmarkOnlyCorrect": int((~correct & benchmark_correct).sum()),
                        "TiedFavourites": int((~unique).sum()),
                        "UniqueFavouriteMatches": int(unique.sum()),
                        "UniqueFavouriteAccuracy": correct.loc[unique].mean(),
                    })
    metrics = pd.DataFrame(rows)
    # Retain both pooled accuracy and the mean of the five league accuracies.
    keys = ["Window", "Period", "Model"]
    league_scores = metrics.loc[metrics["Scope"].eq("League")].copy()
    league_tiers = matches[["Division", "Tier"]].drop_duplicates()
    league_scores = league_scores.merge(league_tiers, left_on="Group", right_on="Division", validate="many_to_one")
    means = league_scores.groupby([*keys, "Tier"])["Accuracy"].mean()
    metrics["MeanLeagueAccuracy"] = np.nan
    for tier in matches["Tier"].unique():
        mask = metrics["Scope"].eq("Tier") & metrics["Group"].eq(tier)
        lookup = pd.MultiIndex.from_frame(metrics.loc[mask, keys])
        metrics.loc[mask, "MeanLeagueAccuracy"] = means.xs(tier, level="Tier").reindex(lookup).to_numpy()
    return metrics


BENCHMARK = "Supplied odds"


def all_periods(metrics, scope=None, window="Available"):
    mask = metrics["Period"].eq("All") & metrics["Window"].eq(window)
    if scope is not None:
        mask &= metrics["Scope"].eq(scope)
    return metrics.loc[mask]


def main():
    predictions = pd.read_csv(REPORT_DIR / "league_tier_predictions.csv", parse_dates=["MatchDate"])
    columns = RAW_KEYS + MARKET_COLUMNS
    sources = pd.read_csv(CORRECTED_PATH, usecols=columns, parse_dates=["MatchDate"])
    joined = predictions.merge(sources, on=RAW_KEYS, how="left", validate="one_to_one")
    valid = joined[MARKET_COLUMNS].gt(1).all(axis=1) & np.isfinite(joined[MARKET_COLUMNS]).all(axis=1)
    joined["Status"] = "Eligible"
    joined.loc[~valid, "Status"] = "Missing/invalid supplied odds"
    joined["Eligible"] = joined["Status"].eq("Eligible")
    if not joined["Eligible"].any():
        raise ValueError("No complete valid supplied odds; run steps 01–04 first.")
    paired = add_odds_predictions(joined.loc[joined["Eligible"]], MARKET_COLUMNS, BENCHMARK)
    metrics = score_comparisons(paired, BENCHMARK)
    coverage = joined.groupby(["Division", "Tier", "Status"]).size().rename("Matches").reset_index()
    joined.to_csv(REPORT_DIR / "supplied_odds_match_audit.csv", index=False)
    paired.to_csv(REPORT_DIR / "supplied_odds_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "supplied_odds_metrics.csv", index=False)
    coverage.to_csv(REPORT_DIR / "supplied_odds_coverage.csv", index=False)
    print(all_periods(metrics, "Tier")[["Group", "Model", "Matches", "Accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
