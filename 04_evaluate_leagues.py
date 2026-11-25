"""Evaluate frozen classifiers on the top five leagues and all three league tiers."""

from importlib import import_module
import json

from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score
from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd


_step_00 = import_module('00_download_sources')
ROOT = _step_00.ROOT

_step_01 = import_module('01_clean_data')
LABELS = _step_01.LABELS
REPORT_DIR = _step_01.REPORT_DIR

_step_03 = import_module('03_train_models')
CANDIDATES = _step_03.CANDIDATES
DATA_PATH = _step_03.DATA_PATH
FIELD_PATH = _step_03.FIELD_PATH
MODEL_DIR = _step_03.MODEL_DIR
TEST_AS_OF = _step_03.TEST_AS_OF
evaluation_dates = _step_03.evaluation_dates
TEST_START = _step_03.TEST_START
TRAIN_START = _step_03.TRAIN_START
evaluate = _step_03.evaluate
season_labels = _step_03.season_labels
LEAGUE_PATH = ROOT / "docs" / "league_strength_2023.csv"
TIERS = ["Top", "Mid", "Lower"]
MODEL_NAMES = list(CANDIDATES)


def select_matches(table, divisions, as_of=None):
    """Keep only requested leagues, known results, and the held-out date window."""
    return table.loc[
        table["Division"].isin(divisions)
        & table["FTResult"].isin(LABELS)
        & evaluation_dates(table["MatchDate"], as_of)
    ]


def common_end_date(matches, divisions):
    """Find the earlier overlap cutoff for secondary sensitivity scores only."""
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
        # Available is primary; Common is a secondary overlap-window sensitivity.
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


def accuracy_comparison(metrics, window="Available"):
    """Return pooled tier accuracies with one column per frozen model."""
    selected = metrics.loc[
        metrics["Window"].eq(window) & metrics["Period"].eq("All") & metrics["Scope"].eq("Tier")
    ]
    comparison = selected.pivot(index="Tier", columns="Model", values="Accuracy").reindex(TIERS)
    comparison = comparison.reindex(columns=MODEL_NAMES)
    comparison.insert(0, "Matches", selected.groupby("Tier")["Matches"].first().reindex(TIERS))
    return comparison.reset_index()


def main():
    leagues = pd.read_csv(LEAGUE_PATH)
    leagues = leagues.loc[leagues["Tier"].isin(TIERS)]
    features = pd.read_csv(FIELD_PATH).query("role == 'feature'")["field"].tolist()
    selected, counts = [], []
    for chunk in pd.read_csv(DATA_PATH, parse_dates=["MatchDate"], chunksize=20000,
                             dtype={field: "float32" for field in features}):
        training = chunk.loc[chunk["FTResult"].isin(LABELS)
                             & chunk["MatchDate"].ge(TRAIN_START) & chunk["MatchDate"].lt(TEST_START)]
        counts.append(training["Division"].value_counts())
        selected.append(select_matches(chunk, leagues["Division"]))
    matches = pd.concat(selected, ignore_index=True)
    if matches.empty:
        raise ValueError("No evaluation matches found; run steps 01–03 first.")
    training_counts = pd.concat(counts, axis=1).fillna(0).sum(axis=1)
    common_end = common_end_date(matches, leagues["Division"])
    metadata = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTResult"]
    predictions = matches[metadata].merge(leagues, on="Division", validate="many_to_one", sort=False)
    predictions["CommonWindow"] = predictions["MatchDate"].le(common_end)
    with threadpool_limits(limits=2):
        for name in MODEL_NAMES:
            slug = name.lower().replace(" ", "_")
            model = joblib.load(MODEL_DIR / f"{slug}.joblib")
            predicted, probabilities, _ = evaluate(model, matches[features], matches["FTResult"])
            predictions[f"{slug}_prediction"] = predicted
            for position, label in enumerate(LABELS):
                predictions[f"{slug}_probability_{label}"] = probabilities[:, position]
    metrics = summarize_predictions(predictions, common_end)
    coverage = coverage_table(matches, leagues, features, training_counts, common_end)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "evaluation_window.json").write_text(json.dumps({
        "StartDate": str(TEST_START.date()), "AsOf": str(TEST_AS_OF.date()),
        "PrimaryWindow": "Available", "CommonEnd": str(common_end.date()),
    }, indent=2) + "\n")
    predictions.to_csv(REPORT_DIR / "league_tier_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "league_tier_metrics.csv", index=False)
    coverage.to_csv(REPORT_DIR / "league_tier_coverage.csv", index=False)
    # Big Five and tier summaries use the same July 2023–today prediction cohort.
    top = predictions.loc[predictions["Tier"].eq("Top")].copy()
    top["Season"] = season_labels(top["MatchDate"])
    top_rows = []
    for division, group in top.groupby("Division"):
        for season in ["All", *sorted(group["Season"].unique())]:
            subset = group if season == "All" else group.loc[group["Season"].eq(season)]
            for name in MODEL_NAMES:
                slug = name.lower().replace(" ", "_")
                scores = prediction_scores(subset["FTResult"], subset[f"{slug}_prediction"],
                                           subset[[f"{slug}_probability_{label}" for label in LABELS]])
                top_rows.append({"Division": division, "Season": season, "Model": name,
                                 "Matches": len(subset), **scores})
    top_metrics = pd.DataFrame(top_rows)
    top_metrics.to_csv(REPORT_DIR / "top_five_model_metrics.csv", index=False)
    top.to_csv(REPORT_DIR / "top_five_model_predictions.csv", index=False)
    print(accuracy_comparison(metrics).to_string())
    print(f"Evaluation: {TEST_START.date()}–{TEST_AS_OF.date()}; secondary overlap ends {common_end.date()}.")


if __name__ == "__main__":
    main()
