"""Classifier settings, chronological splits and probability scoring."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    log_loss, recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "MatchTraining.csv"
FIELD_PATH = ROOT / "docs" / "match_prediction_fields.csv"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
LABELS = ["A", "D", "H"]  # Probability columns and confusion matrices use this order.
TRAIN_START = pd.Timestamp("2000-07-01")
VALIDATION_START = pd.Timestamp("2021-07-01")
TEST_START = pd.Timestamp("2023-07-01")
TEST_END = pd.Timestamp("2026-07-01")  # Exclusive: stop after 2025/26.

# These small grids are chosen before inspecting any 2023/24-or-later scores.
# var_smoothing adds variance to stabilize Naive Bayes probabilities.
# Smaller C means stronger logistic regression regularization.
# Fewer leaves and larger l2_regularization restrict the boosted trees.
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

JUSTIFICATIONS = {
    "Prior baseline": "Always predicts the most frequent training result; probabilities are training class frequencies. Establishes a minimum benchmark.",
    "Gaussian Naive Bayes": "A fast Bayesian classification baseline for numeric inputs. Its conditional-independence and Gaussian assumptions are imperfect for correlated match statistics; variance smoothing limits overconfidence.",
    "Logistic regression": "A regularized, interpretable linear multiclass model. Suitable for many correlated numerical inputs; scaling and regularization control the influence of their different units.",
    "Histogram gradient boosting": "Boosted trees capture nonlinear relationships and interactions between Elo, venue, and individual previous games. Histogram binning scales to this dataset and the model handles missing statistics directly.",
}


def split_data(table, as_of=None):
    """Use date windows, never a random split, and keep only known H/D/A targets."""
    today = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    known_result = table["FTResult"].isin(LABELS)
    dates = table["MatchDate"]
    # Every league contributes to training from July 2000 through June 2023.
    train = table.loc[known_result & dates.ge(TRAIN_START) & dates.lt(TEST_START)]
    # E0 is the English Premier League; E1 is the English Championship.
    test = table.loc[
        known_result & table["Division"].eq("E0")
        & dates.ge(TEST_START) & dates.lt(TEST_END) & dates.le(today)
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


def markdown_table(frame):
    """Format a report table without needing a separate Markdown dependency."""
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


