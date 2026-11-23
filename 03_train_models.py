"""Select the three classifiers on earlier validation and test the Premier League."""

import time
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from src.models import (
    CANDIDATES, JUSTIFICATIONS, DATA_PATH, FIELD_PATH, MODEL_DIR, REPORT_DIR, LABELS,
    VALIDATION_START, split_data, make_model, evaluate, season_labels,
)
from src.reporting import write_report, preparation_note


def main():
    # Read the feature dictionary so only the documented numeric inputs are used.
    fields = pd.read_csv(FIELD_PATH)
    features = fields.loc[fields["role"].eq("feature"), "field"].tolist()
    # float32 reduces memory use without changing the stored training CSV.
    table = pd.read_csv(DATA_PATH, parse_dates=["MatchDate"], dtype={field: "float32" for field in features})
    train, test = split_data(table)
    tuning_train = train.loc[train["MatchDate"].lt(VALIDATION_START)]
    validation = train.loc[train["Division"].eq("E0") & train["MatchDate"].ge(VALIDATION_START)]
    if tuning_train.empty or validation.empty:
        raise ValueError("The earlier model-selection training/validation window is empty.")
    print(f"Training: {len(train):,}; earlier validation: {len(validation):,}; Premier League test: {len(test):,}.", flush=True)

    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    predictions_table = test[["MatchDate", "HomeTeam", "AwayTeam", "FTResult"]].copy()
    predictions_table.insert(1, "Season", season_labels(test["MatchDate"]))
    validation_rows, metric_rows, details = [], [], []

    # Bound CPU usage for reproducibility and avoid excessive BLAS/OpenMP threads.
    with threadpool_limits(limits=2):
        for name, candidates in CANDIDATES.items():
            started = time.perf_counter()
            print(f"Selecting and fitting {name}...", flush=True)
            best_loss, best_parameters = float("inf"), None
            for parameters in candidates:
                candidate = make_model(name, parameters)
                candidate.fit(tuning_train[features], tuning_train["FTResult"])
                _, _, scores = evaluate(candidate, validation[features], validation["FTResult"])
                validation_rows.append({"Model": name, "Parameters": str(parameters), **scores})
                if scores["LogLoss"] < best_loss:
                    best_loss, best_parameters = scores["LogLoss"], parameters

            # Refit from scratch so the final estimator sees every allowed season.
            model = make_model(name, best_parameters)
            model.fit(train[features], train["FTResult"])
            predicted, probabilities, scores = evaluate(model, test[features], test["FTResult"])
            metric_rows.append({"Model": name, "Season": "All", "Matches": len(test), **scores})
            slug = name.lower().replace(" ", "_")
            predictions_table[f"{slug}_prediction"] = predicted
            for position, label in enumerate(LABELS):
                predictions_table[f"{slug}_probability_{label}"] = probabilities[:, position]
            joblib.dump(model, MODEL_DIR / f"{slug}.joblib", compress=3)

            # Score the same frozen estimator separately for each test season.
            seasons = season_labels(test["MatchDate"])
            for season in sorted(seasons.unique()):
                mask = seasons.eq(season)
                _, _, season_scores = evaluate(model, test.loc[mask, features], test.loc[mask, "FTResult"])
                metric_rows.append({"Model": name, "Season": season, "Matches": int(mask.sum()), **season_scores})
            class_report = classification_report(test["FTResult"], predicted, labels=LABELS, output_dict=True, zero_division=0)
            class_scores = pd.DataFrame([
                {"Result": label, **class_report[label]} for label in LABELS
            ])
            matrix = pd.DataFrame(confusion_matrix(test["FTResult"], predicted, labels=LABELS), columns=LABELS)
            matrix.insert(0, "Actual", LABELS)
            details.append((name, class_scores, matrix))
            print(f"{name}: accuracy={scores['Accuracy']:.4f}, log loss={scores['LogLoss']:.4f}, "
                  f"parameters={best_parameters}, elapsed={time.perf_counter() - started:.1f}s.", flush=True)

    # Save compact machine-readable outputs and a single explanatory report.
    metrics = pd.DataFrame(metric_rows)
    validation_scores = pd.DataFrame(validation_rows)
    validation_scores["Selected"] = validation_scores.groupby("Model")["LogLoss"].rank(method="first").eq(1)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    validation_scores.to_csv(REPORT_DIR / "validation_metrics.csv", index=False)
    predictions_table.to_csv(REPORT_DIR / "model_predictions.csv", index=False)
    tables = [("Model choices", pd.DataFrame(JUSTIFICATIONS.items(), columns=["Model", "Reason"])),
              ("Earlier validation", validation_scores), ("Premier League scores", metrics)]
    for name, class_scores, matrix in details:
        tables.extend([(f"{name}: class scores", class_scores), (f"{name}: confusion matrix (actual rows)", matrix)])
    write_report("model_evaluation.md", "Match outcome classification",
                 f"Training: {len(train):,} matches across all available leagues, July 2000–June 2023. "
                 f"Premier League test: {len(test):,} matches, July 2023–June 2026. "
                 "Parameters minimize log loss on earlier Premier League validation (July 2021–June 2023); "
                 "candidate fitting uses only dates before July 2021. Preprocessing is fitted on training rows. "
                 + preparation_note() + " Scores are descriptive; see README for limitations.", tables)
    print(f"Report saved to {REPORT_DIR / 'model_evaluation.md'}", flush=True)


if __name__ == "__main__":
    main()
