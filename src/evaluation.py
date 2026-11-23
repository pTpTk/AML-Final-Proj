"""Compare frozen classifiers across 15 leagues chosen using 2023 UEFA ranks."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score

from .models import (
    CANDIDATES, LABELS, ROOT, TEST_START, TEST_END,
)


LEAGUE_PATH = ROOT / "docs" / "league_strength_2023.csv"
TIERS = ["Top", "Mid", "Lower"]
MODEL_NAMES = list(CANDIDATES)


def select_matches(table, divisions, as_of=None):
    """Keep only requested leagues, known results, and the held-out date window."""
    today = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    return table.loc[
        table["Division"].isin(divisions)
        & table["FTResult"].isin(LABELS)
        & table["MatchDate"].ge(TEST_START)
        & table["MatchDate"].lt(TEST_END)
        & table["MatchDate"].le(today)
    ]


def common_end_date(matches, divisions):
    """Stop the main comparison at the earliest last match among all 15 leagues."""
    last_dates = matches.groupby("Division")["MatchDate"].max().reindex(divisions)
    if last_dates.isna().any():
        missing = last_dates.index[last_dates.isna()].tolist()
        raise ValueError(f"Missing evaluation matches for: {missing}")
    return last_dates.min()


def prediction_scores(actual, predicted, probabilities):
    """Score a stored prediction subset; this function never fits a model."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "BalancedAccuracy": recall_score(
            actual, predicted, labels=np.unique(actual), average="macro", zero_division=0,
        ),
        "MacroF1": f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0),
        "LogLoss": log_loss(actual, probabilities, labels=LABELS),
        "DrawRecall": recall_score(actual, predicted, labels=["D"], average="macro", zero_division=0),
    }


def summarize_predictions(predictions, common_end):
    """Score every model by league and tier, for matching dates and all available dates."""
    rows = []
    for window in ["Common", "Available"]:
        # All available dates are secondary because some leagues end much earlier.
        matches = predictions if window == "Available" else predictions.loc[
            predictions["MatchDate"].le(common_end)
        ]
        for period in ["All", *sorted(matches["MatchDate"].dt.year.unique().astype(str))]:
            # Calendar years work for both summer-to-spring and calendar-year leagues.
            period_matches = matches if period == "All" else matches.loc[
                matches["MatchDate"].dt.year.eq(int(period))
            ]
            for tier in TIERS:
                tier_matches = period_matches.loc[period_matches["Tier"].eq(tier)]
                groups = [("Tier", tier, tier_matches)]
                groups += [("League", code, group) for code, group in tier_matches.groupby("Division")]
                for scope, code, group in groups:
                    if group.empty:
                        continue
                    for name in MODEL_NAMES:
                        slug = name.lower().replace(" ", "_")
                        probabilities = group[[f"{slug}_probability_{label}" for label in LABELS]]
                        scores = prediction_scores(group["FTResult"], group[f"{slug}_prediction"], probabilities)
                        rows.append({
                            "Window": window, "Period": period, "Scope": scope,
                            "Tier": tier, "Division": code, "Model": name,
                            "Matches": len(group), **scores,
                        })
    metrics = pd.DataFrame(rows)
    # A pooled accuracy weights leagues by match count. Also show each league
    # receiving one fifth of the tier weight, regardless of its sample size.
    keys = ["Window", "Period", "Tier", "Model"]
    league_means = metrics.loc[metrics["Scope"].eq("League")].groupby(keys)["Accuracy"].mean()
    league_means = league_means.rename("MeanLeagueAccuracy").reset_index()
    metrics = metrics.merge(league_means, on=keys, how="left", validate="many_to_one")
    metrics.loc[metrics["Scope"].eq("League"), "MeanLeagueAccuracy"] = np.nan
    return metrics


def coverage_table(matches, leagues, features, training_counts, common_end):
    """Measure missing inputs and outcome frequencies without changing the inputs."""
    # These detailed statistics exclude goals, results, venue, elapsed days and Elo.
    stat_names = ("Shots", "ShotsOnTarget", "Fouls", "Corners", "YellowCards", "RedCards")
    stat_columns = [field for field in features if any(
        field.endswith(f"_{stat}{direction}") for stat in stat_names for direction in ["For", "Against"]
    )]
    rows = []
    for window in ["Common", "Available"]:
        selected = matches if window == "Available" else matches.loc[matches["MatchDate"].le(common_end)]
        for code, group in selected.groupby("Division"):
            rows.append({
                "Window": window, "Division": code, "Matches": len(group),
                "FirstMatch": group["MatchDate"].min().date(),
                "LastMatch": group["MatchDate"].max().date(),
                "TrainingMatches": int(training_counts.get(code, 0)),
                "CurrentEloMissing": group[["HomeElo", "AwayElo"]].isna().to_numpy().mean(),
                "DetailedStatsMissing": group[stat_columns].isna().to_numpy().mean(),
                "AllFeaturesMissing": group[features].isna().to_numpy().mean(),
                "CompleteFiveGameHistory": group[["Home_HistoryCount", "Away_HistoryCount"]].eq(5).all(axis=1).mean(),
                "HomeWinRate": group["FTResult"].eq("H").mean(),
                "DrawRate": group["FTResult"].eq("D").mean(),
                "AwayWinRate": group["FTResult"].eq("A").mean(),
            })
    return pd.DataFrame(rows).merge(leagues, on="Division", validate="many_to_one")


def accuracy_comparison(metrics, window="Common"):
    """Return pooled tier accuracies with one column per frozen model."""
    selected = metrics.loc[
        metrics["Window"].eq(window) & metrics["Period"].eq("All") & metrics["Scope"].eq("Tier")
    ]
    comparison = selected.pivot(index="Tier", columns="Model", values="Accuracy").reindex(TIERS)
    comparison = comparison.reindex(columns=MODEL_NAMES)
    comparison.insert(0, "Matches", selected.groupby("Tier")["Matches"].first().reindex(TIERS))
    return comparison.reset_index()


