"""Select the three classifiers on earlier validation and test the Premier League."""

from importlib import import_module
import argparse
import time

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd


_step_00 = import_module('00_download_sources')
ROOT = _step_00.ROOT

_step_01 = import_module('01_clean_data')
LABELS = _step_01.LABELS
REPORT_DIR = _step_01.REPORT_DIR
DATA_PATH = ROOT / "data" / "MatchTraining.csv"
FIELD_PATH = ROOT / "docs" / "match_prediction_fields.csv"
MODEL_DIR = ROOT / "models"
TRAIN_START = pd.Timestamp("2000-07-01")
VALIDATION_START = pd.Timestamp("2021-07-01")
TEST_START = pd.Timestamp("2023-07-01")
TEST_AS_OF = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
TEST_END = TEST_AS_OF + pd.Timedelta(days=1)  # Exclusive: include today's matches.


def evaluation_dates(dates, as_of=None):
    """Apply the same July 2023–as-of-date window to every evaluation."""
    end = pd.Timestamp(as_of).normalize() if as_of is not None else TEST_AS_OF
    return dates.ge(TEST_START) & dates.lt(end + pd.Timedelta(days=1))

# Select settings on the earlier chronological validation window, before refitting.
CANDIDATES = {
    "Prior baseline": [{}],
    "Gaussian Naive Bayes": [
        {"var_smoothing": 0.01}, {"var_smoothing": 1.0}, {"var_smoothing": 10.0},
    ],
    "Logistic regression": [{"C": 0.1}, {"C": 1.0}],
    "Histogram gradient boosting": [
        {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"max_leaf_nodes": 31, "l2_regularization": 10.0},
    ],
}


def split_data(table, as_of=None):
    """Use date windows, never a random split, and keep only known H/D/A targets."""
    known_result = table["FTResult"].isin(LABELS)
    dates = table["MatchDate"]
    # Every league contributes to training from July 2000 through June 2023.
    train = table.loc[known_result & dates.ge(TRAIN_START) & dates.lt(TEST_START)]
    # E0 is the English Premier League; E1 is the English Championship.
    test = table.loc[
        known_result & table["Division"].eq("E0")
        & evaluation_dates(dates, as_of)
    ]
    if train.empty or test.empty:
        raise ValueError("The requested training or Premier League evaluation window is empty.")
    return train, test


def make_model(name, parameters):
    """Return a fresh estimator; preprocessing is learned only when fit is called."""
    if name == "Prior baseline":
        return DummyClassifier(strategy="prior")
    if name == "Histogram gradient boosting":
        # Native missing-value handling avoids imputing this tree model.
        # Disable its random internal early-stopping split; use our earlier
        # chronological validation window to select model complexity instead.
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, min_samples_leaf=50,
            early_stopping=False, random_state=42, **parameters,
        )
    if name == "Gaussian Naive Bayes":
        classifier = GaussianNB(**parameters)
    elif name == "Logistic regression":
        classifier = LogisticRegression(max_iter=1000, solver="lbfgs", **parameters)
    else:
        raise ValueError(f"Unknown model: {name}")
    # The median fills missing values. Indicators preserve information about
    # which fields were missing. Scaling is important for the linear model
    # and makes variance smoothing comparable across Naive Bayes features.
    preprocessing = [
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
    ]
    if name == "Gaussian Naive Bayes":
        # Summing hundreds of log likelihoods in float32 causes cancellation
        # when GaussianNB converts them to probabilities. Use float64 here.
        preprocessing.append(FunctionTransformer(np.asarray, kw_args={"dtype": np.float64}))
    return make_pipeline(*preprocessing, classifier)


def evaluate(model, X, y):
    """Return class predictions, ordered probabilities, and four summary scores."""
    predictions = model.predict(X)
    # Explicitly align probability columns with A/D/H, including saved exports.
    probability_order = [list(model.classes_).index(label) for label in LABELS]
    probabilities = model.predict_proba(X)[:, probability_order].astype("float64")
    # Remove floating-point sum errors before scoring or exporting as CSV.
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    scores = {
        "Accuracy": accuracy_score(y, predictions),
        # Average recall over the actual classes present in this group. Some
        # partial seasons have no examples of one outcome; this also avoids
        # sklearn's warning when that absent outcome is nevertheless predicted.
        "BalancedAccuracy": recall_score(
            y, predictions, labels=np.unique(y), average="macro", zero_division=0,
        ),
        "MacroF1": f1_score(y, predictions, labels=LABELS, average="macro", zero_division=0),
        "LogLoss": log_loss(y, probabilities, labels=LABELS),
    }
    return predictions, probabilities, scores


def season_labels(dates):
    """Label European seasons using their July-to-June windows."""
    start = dates.dt.year - dates.dt.month.lt(7).astype(int)
    return start.astype(str) + "/" + ((start + 1) % 100).astype(str).str.zfill(2)


def main(evaluate_only=False):
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
    validation_rows, metric_rows = [], []

    # Bound CPU usage for reproducibility and avoid excessive BLAS/OpenMP threads.
    with threadpool_limits(limits=2):
        for name, candidates in CANDIDATES.items():
            started = time.perf_counter()
            print(f"{'Evaluating' if evaluate_only else 'Selecting and fitting'} {name}...", flush=True)
            slug = name.lower().replace(" ", "_")
            best_loss, best_parameters = float("inf"), None
            for parameters in ([] if evaluate_only else candidates):
                candidate = make_model(name, parameters)
                candidate.fit(tuning_train[features], tuning_train["FTResult"])
                _, _, scores = evaluate(candidate, validation[features], validation["FTResult"])
                validation_rows.append({"Model": name, "Parameters": str(parameters), **scores})
                if scores["LogLoss"] < best_loss:
                    best_loss, best_parameters = scores["LogLoss"], parameters

            # Refit from scratch so the final estimator sees every allowed season.
            if evaluate_only:
                model = joblib.load(MODEL_DIR / f"{slug}.joblib")
            else:
                model = make_model(name, best_parameters)
                model.fit(train[features], train["FTResult"])
            predicted, probabilities, scores = evaluate(model, test[features], test["FTResult"])
            metric_rows.append({"Model": name, "Season": "All", "Matches": len(test), **scores})
            predictions_table[f"{slug}_prediction"] = predicted
            for position, label in enumerate(LABELS):
                predictions_table[f"{slug}_probability_{label}"] = probabilities[:, position]
            if not evaluate_only:
                joblib.dump(model, MODEL_DIR / f"{slug}.joblib", compress=3)

            # Score the same frozen estimator separately for each test season.
            seasons = season_labels(test["MatchDate"])
            for season in sorted(seasons.unique()):
                mask = seasons.eq(season)
                _, _, season_scores = evaluate(model, test.loc[mask, features], test.loc[mask, "FTResult"])
                metric_rows.append({"Model": name, "Season": season, "Matches": int(mask.sum()), **season_scores})
            print(f"{name}: accuracy={scores['Accuracy']:.4f}, log loss={scores['LogLoss']:.4f}, "
                  f"parameters={best_parameters}, elapsed={time.perf_counter() - started:.1f}s.", flush=True)

    # Save numerical exports; the unified study document is built after the audit.
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(REPORT_DIR / "model_metrics.csv", index=False)
    if not evaluate_only:
        validation_scores = pd.DataFrame(validation_rows)
        validation_scores["Selected"] = validation_scores.groupby("Model")["LogLoss"].rank(method="first").eq(1)
        validation_scores.to_csv(REPORT_DIR / "validation_metrics.csv", index=False)
    predictions_table.to_csv(REPORT_DIR / "model_predictions.csv", index=False)
    print(f"Metrics and predictions saved to {REPORT_DIR}.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true", help="Reuse fitted models and refresh test results.")
    main(evaluate_only=parser.parse_args().evaluate_only)
