"""Controlled comparison of the same classifiers with and without market inputs."""

import numpy as np
import pandas as pd
from sklearn.base import clone

from .data_sources import MARKET_COLUMNS
from .evaluation import TIERS
from .models import LABELS, TRAIN_START, TEST_START, TEST_END

BENCHMARK = "Supplied odds"
FEATURE_SETS = ["No odds", "With market odds"]


def fit_same_architecture(original, X, y):
    """Keep selected settings, discard fitted state, and fit fresh preprocessing."""
    return clone(original).fit(X, y)


def select_train_and_test(table, divisions, as_of=None):
    """Use corrected source dates and targets; never randomly split the matches."""
    today = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    known = table["FTResult"].isin(LABELS)
    if "SourceResult" in table:
        known &= table["FTResult"].eq(table["SourceResult"])
    train = table.loc[known & table["MatchDate"].ge(TRAIN_START) & table["MatchDate"].lt(TEST_START)]
    valid_market = table[MARKET_COLUMNS].gt(1).all(axis=1) & np.isfinite(table[MARKET_COLUMNS]).all(axis=1)
    test = table.loc[
        known & table["Division"].isin(divisions) & table["MatchDate"].ge(TEST_START)
        & table["MatchDate"].lt(TEST_END) & table["MatchDate"].le(today) & valid_market
    ].copy()
    if train.empty or test.empty:
        raise ValueError("The chronological training or evaluation window is empty.")
    return train, test

def accuracy_changes(metrics):
    """Pair the corrected control and augmented scores on identical test rows."""
    keys = ["Window", "Period", "Scope", "Group", "Model"]
    columns = [*keys, "Matches", "Accuracy", "BalancedAccuracy", "MacroF1", "LogLoss", "DrawRecall", "MeanLeagueAccuracy", "GapToBenchmark"]
    control = metrics.loc[metrics["FeatureSet"].eq("No odds") & metrics["Model"].ne(BENCHMARK), columns]
    augmented = metrics.loc[metrics["FeatureSet"].eq("With market odds") & metrics["Model"].ne(BENCHMARK), columns]
    changes = control.merge(augmented, on=keys, suffixes=("_NoOdds", "_WithOdds"), validate="one_to_one")
    if not changes["Matches_NoOdds"].eq(changes["Matches_WithOdds"]).all():
        raise ValueError("Control and augmented models must use exactly the same match counts.")
    changes["AccuracyChangePP"] = 100 * (changes["Accuracy_WithOdds"] - changes["Accuracy_NoOdds"])
    changes["LogLossChange"] = changes["LogLoss_WithOdds"] - changes["LogLoss_NoOdds"]
    return changes

def accuracy_table(changes, window="Common", scope="Combined"):
    """Display the before/after accuracy for each architecture, keeping the groups."""
    rows = changes.loc[
        changes["Window"].eq(window) & changes["Period"].eq("All") & changes["Scope"].eq(scope),
        ["Group", "Model", "Matches_NoOdds", "Accuracy_NoOdds", "Accuracy_WithOdds", "AccuracyChangePP"],
    ].copy()
    return rows.rename(columns={"Matches_NoOdds": "Matches", "Accuracy_NoOdds": "NoOdds", "Accuracy_WithOdds": "WithOdds"})

def tier_accuracy_table(changes, metrics, window="Common"):
    """Compare both model versions with the market favourite on the same games."""
    rows = accuracy_table(changes, window, scope="Tier")
    market = metrics.loc[
        metrics["FeatureSet"].eq("With market odds") & metrics["Window"].eq(window)
        & metrics["Period"].eq("All") & metrics["Scope"].eq("Tier")
        & metrics["Model"].eq(BENCHMARK),
        ["Group", "Matches", "Accuracy"],
    ].rename(columns={"Accuracy": "Market favourite"})
    # Repeat the tier's market score beside each architecture for easy comparison.
    rows = rows.merge(market, on=["Group", "Matches"], validate="many_to_one")
    rows = pd.concat([rows.loc[rows["Group"].eq(tier)] for tier in TIERS], ignore_index=True)
    rows = rows.rename(columns={
        "Group": "Tier", "NoOdds": "Model without market odds",
        "WithOdds": "Model with market odds", "AccuracyChangePP": "Change (pp)",
    })
    for column in ["Model without market odds", "Market favourite", "Model with market odds"]:
        rows[column] = rows[column].map(lambda accuracy: f"{accuracy:.2%}")
    rows["Change (pp)"] = rows["Change (pp)"].map(lambda change: f"{change:+.2f}")
    return rows[[
        "Tier", "Model", "Matches", "Model without market odds", "Market favourite",
        "Model with market odds", "Change (pp)",
    ]]
