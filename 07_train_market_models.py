"""Fit the same selected classifiers with and without the 33 market inputs."""

from importlib import import_module
import argparse
import hashlib
import json
import time

from sklearn.base import clone
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd


_step_01 = import_module('01_clean_data')
CORRECTED_PATH = _step_01.CORRECTED_PATH
LABELS = _step_01.LABELS
MARKET_COLUMNS = _step_01.MARKET_COLUMNS
RAW_KEYS = _step_01.RAW_KEYS
REPORT_DIR = _step_01.REPORT_DIR

_step_03 = import_module('03_train_models')
FIELD_PATH = _step_03.FIELD_PATH
MODEL_DIR = _step_03.MODEL_DIR
evaluation_dates = _step_03.evaluation_dates
TEST_START = _step_03.TEST_START
TRAIN_START = _step_03.TRAIN_START
evaluate = _step_03.evaluate

_step_04 = import_module('04_evaluate_leagues')
LEAGUE_PATH = _step_04.LEAGUE_PATH
MODEL_NAMES = _step_04.MODEL_NAMES
TIERS = _step_04.TIERS

_step_05 = import_module('05_compare_market')
BENCHMARK = _step_05.BENCHMARK
add_odds_predictions = _step_05.add_odds_predictions
score_comparisons = _step_05.score_comparisons

_step_06 = import_module('06_add_market_features')
MARKET_FIELD_PATH = _step_06.MARKET_FIELD_PATH
TRAINING_PATH = _step_06.MARKET_TRAINING_PATH
FEATURE_SETS = ["No odds", "With market odds"]


def fit_same_architecture(original, X, y):
    """Keep selected settings, discard fitted state, and fit fresh preprocessing."""
    return clone(original).fit(X, y)


def select_train_and_test(table, divisions, as_of=None):
    """Use corrected source dates and targets; never randomly split the matches."""
    known = table["FTResult"].isin(LABELS)
    train = table.loc[known & table["MatchDate"].ge(TRAIN_START) & table["MatchDate"].lt(TEST_START)]
    valid_market = table[MARKET_COLUMNS].gt(1).all(axis=1) & np.isfinite(table[MARKET_COLUMNS]).all(axis=1)
    test = table.loc[
        known & table["Division"].isin(divisions)
        & evaluation_dates(table["MatchDate"], as_of) & valid_market
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


def accuracy_table(changes, window="Available", scope="Combined"):
    """Display the before/after accuracy for each architecture, keeping the groups."""
    rows = changes.loc[
        changes["Window"].eq(window) & changes["Period"].eq("All") & changes["Scope"].eq(scope),
        ["Group", "Model", "Matches_NoOdds", "Accuracy_NoOdds", "Accuracy_WithOdds", "AccuracyChangePP"],
    ].copy()
    return rows.rename(columns={"Matches_NoOdds": "Matches", "Accuracy_NoOdds": "NoOdds", "Accuracy_WithOdds": "WithOdds"})


OUTPUT_MODEL_DIR = MODEL_DIR / "market_augmented"
PREDICTION_PATH = REPORT_DIR / "league_tier_predictions.csv"


def main(evaluate_only=False):
    fields = pd.read_csv(MARKET_FIELD_PATH)
    all_features = fields.loc[fields["role"].eq("feature"), "field"].tolist()
    base_fields = pd.read_csv(FIELD_PATH)
    features = {
        "No odds": base_fields.loc[base_fields["role"].eq("feature"), "field"].tolist(),
        "With market odds": all_features,
    }
    table = pd.read_csv(TRAINING_PATH, parse_dates=["MatchDate"], dtype={field: "float32" for field in all_features})
    odds = pd.read_csv(CORRECTED_PATH, usecols=RAW_KEYS + MARKET_COLUMNS, parse_dates=["MatchDate"])
    table = table.merge(odds, on=RAW_KEYS, validate="one_to_one", how="left")
    leagues = pd.read_csv(LEAGUE_PATH)
    leagues = leagues.loc[leagues["Tier"].isin(TIERS)]
    train, test = select_train_and_test(table, leagues["Division"])
    test = test.merge(leagues[["Division", "Tier", "Country", "League"]], on="Division", validate="many_to_one")
    previous = pd.read_csv(PREDICTION_PATH, usecols=["MatchDate", "CommonWindow"], parse_dates=["MatchDate"])
    common_end = previous.loc[previous["CommonWindow"], "MatchDate"].max()
    test["CommonWindow"] = test["MatchDate"].le(common_end)
    predictions = test[[
        "Division", "MatchDate", "HomeTeam", "AwayTeam", "FTResult", "Tier", "Country", "League", "CommonWindow", *MARKET_COLUMNS,
    ]].copy()
    predictions = add_odds_predictions(predictions, MARKET_COLUMNS, BENCHMARK)
    print(f"Prepared training rows: {len(train):,}; test: {len(test):,}; common: {test['CommonWindow'].sum():,}.", flush=True)
    REPORT_DIR.mkdir(exist_ok=True)
    OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    originals, settings = {}, {}
    for name in MODEL_NAMES:
        slug = name.lower().replace(" ", "_")
        path = MODEL_DIR / f"{slug}.joblib"
        originals[name] = joblib.load(path)
        settings[name] = {"OriginalSHA256": hashlib.sha256(path.read_bytes()).hexdigest(), "Estimator": str(originals[name])}
    if not evaluate_only:
        (REPORT_DIR / "market_augmented_settings.json").write_text(json.dumps(settings, indent=2))
    fit_records, prediction_sets, metric_sets = [], [], []
    with threadpool_limits(limits=2):
        for feature_set in FEATURE_SETS:
            selected = features[feature_set]
            variant = predictions.copy()
            variant["FeatureSet"] = feature_set
            folder = OUTPUT_MODEL_DIR / feature_set.lower().replace(" ", "_")
            folder.mkdir(exist_ok=True)
            for name in MODEL_NAMES:
                started = time.perf_counter()
                slug = name.lower().replace(" ", "_")
                print(f"{'Evaluating' if evaluate_only else 'Fitting'} {name}: {feature_set}, {len(selected)} features...", flush=True)
                if evaluate_only:
                    model = joblib.load(folder / f"{slug}.joblib")
                else:
                    model = fit_same_architecture(originals[name], train[selected], train["FTResult"])
                    joblib.dump(model, folder / f"{slug}.joblib", compress=3)
                predicted, probabilities, _ = evaluate(model, test[selected], test["FTResult"])
                variant[f"{slug}_prediction"] = predicted
                for position, label in enumerate(LABELS):
                    variant[f"{slug}_probability_{label}"] = probabilities[:, position]
                seconds = time.perf_counter() - started
                fit_records.append({"FeatureSet": feature_set, "Model": name, "TrainingMatches": len(train), "Features": len(selected), "Seconds": seconds})
                if not evaluate_only:
                    pd.DataFrame(fit_records).to_csv(REPORT_DIR / "market_augmented_training.csv", index=False)
                print(f"Completed {name}: {feature_set} in {seconds:.1f}s.", flush=True)
            metrics = score_comparisons(variant, BENCHMARK)
            metrics["FeatureSet"] = feature_set
            metric_sets.append(metrics)
            prediction_sets.append(variant)
    metrics = pd.concat(metric_sets, ignore_index=True)
    changes = accuracy_changes(metrics)
    metrics.to_csv(REPORT_DIR / "market_augmented_metrics.csv", index=False)
    changes.to_csv(REPORT_DIR / "market_augmented_accuracy_changes.csv", index=False)
    pd.concat(prediction_sets, ignore_index=True).to_csv(REPORT_DIR / "market_augmented_predictions.csv", index=False)
    print(accuracy_table(changes).to_string(index=False))
    print(f"Comparison exports saved to {REPORT_DIR}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse fitted models and refresh test results.")
    main(evaluate_only=parser.parse_args().evaluate_only)
