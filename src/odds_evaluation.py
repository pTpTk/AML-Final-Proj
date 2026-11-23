"""Shared odds-to-prediction rules and paired scoring for market benchmarks."""

import numpy as np
import pandas as pd

from .evaluation import MODEL_NAMES, prediction_scores
from .models import LABELS


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


def score_comparisons(matches, benchmark_name, tied_column="TiedFavourite"):
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
                unique = ~group[tied_column]
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
