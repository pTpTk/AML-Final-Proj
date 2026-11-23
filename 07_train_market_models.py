"""Fit the same selected classifiers with and without the 33 market inputs."""

import hashlib
import json
import time
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data_sources import MARKET_COLUMNS, RAW_KEYS, CORRECTED_PATH
from src.evaluation import LEAGUE_PATH, MODEL_NAMES, TIERS
from src.features import MARKET_FIELD_PATH, MARKET_TRAINING_PATH as TRAINING_PATH
from src.market_models import (
    BENCHMARK, FEATURE_SETS, fit_same_architecture, select_train_and_test,
    accuracy_changes, accuracy_table, tier_accuracy_table,
)
from src.models import FIELD_PATH, MODEL_DIR, REPORT_DIR, LABELS, evaluate
from src.odds_evaluation import add_odds_predictions, score_comparisons
from src.reporting import write_report, preparation_note

OUTPUT_MODEL_DIR = MODEL_DIR / "market_augmented"
PREDICTION_PATH = REPORT_DIR / "league_tier_predictions.csv"


def main():
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
                print(f"Fitting {name}: {feature_set}, {len(selected)} features...", flush=True)
                model = fit_same_architecture(originals[name], train[selected], train["FTResult"])
                slug = name.lower().replace(" ", "_")
                joblib.dump(model, folder / f"{slug}.joblib", compress=3)
                predicted, probabilities, _ = evaluate(model, test[selected], test["FTResult"])
                variant[f"{slug}_prediction"] = predicted
                for position, label in enumerate(LABELS):
                    variant[f"{slug}_probability_{label}"] = probabilities[:, position]
                seconds = time.perf_counter() - started
                fit_records.append({"FeatureSet": feature_set, "Model": name, "TrainingMatches": len(train), "Features": len(selected), "Seconds": seconds})
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
    tables = [("Tier accuracy", tier_accuracy_table(changes, metrics)),
              ("Pooled accuracy changes", accuracy_table(changes)),
              ("Training runs", pd.DataFrame(fit_records))]
    write_report("market_augmented_evaluation.md", "Adding market probabilities",
                 f"Both feature sets use the same {len(train):,} labeled training rows before July 2023. "
                 "The selected settings from step 03 are cloned and fitted afresh, including preprocessing. "
                 f"No test-set tuning. Primary test window ends {common_end.date()}; "
                 "every comparison uses identical eligible rows. The control uses 214 numeric features; "
                 "the augmented models use 247. Market features use normalized inverse decimal odds. "
                 "Historical quotes follow the team's venue and are shifted before joining. "
                 "All quotes are supplied OddHome/OddDraw/OddAway values from Matches.csv; missing quotes stay missing. "
                 "Results are descriptive and can differ from the draft outline after date preparation and "
                 "the June 2026 cutoff. Detailed league/year metrics are in market_augmented_metrics.csv. " + preparation_note(), tables)
    print(accuracy_table(changes).to_string(index=False))
    print(f"Report saved to {REPORT_DIR / 'market_augmented_evaluation.md'}")


if __name__ == "__main__":
    main()
