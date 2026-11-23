"""Evaluate frozen classifiers on the top five leagues and all three league tiers."""

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from src.models import (
    DATA_PATH, FIELD_PATH, LABELS, MODEL_DIR, REPORT_DIR, TRAIN_START, TEST_START,
    evaluate, season_labels,
)
from src.evaluation import (
    LEAGUE_PATH, TIERS, MODEL_NAMES, select_matches, common_end_date,
    summarize_predictions, coverage_table, accuracy_comparison, prediction_scores,
)
from src.reporting import write_report, preparation_note


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
    predictions.to_csv(REPORT_DIR / "league_tier_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "league_tier_metrics.csv", index=False)
    coverage.to_csv(REPORT_DIR / "league_tier_coverage.csv", index=False)
    # The outline first compares each top league over the full three-season window.
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
    write_report("top_five_evaluation.md", "Predictability across the top five leagues",
                 "Frozen models; test dates July 2023–June 2026. No refitting or parameter selection. "
                 "Earlier held-out games can become history for later games. " + preparation_note(),
                 [("All seasons", top_metrics.loc[top_metrics["Season"].eq("All")]),
                  ("By season", top_metrics.loc[top_metrics["Season"].ne("All")])])
    write_report("league_tier_evaluation.md", "Predictability across 15 leagues",
                 f"Primary comparison: July 2023 through {common_end.date()}, the earliest last match "
                 "among all 15 leagues. Secondary Available results stop by June 2026. "
                 "Tiers use published 2023 UEFA ranks. Coverage differences and outcome frequencies "
                 "can affect accuracy; observed gaps are not significance or popularity claims. "
                 "Full results, including calendar-year checks, are in league_tier_metrics.csv. " + preparation_note(),
                 [("Tier accuracy", accuracy_comparison(metrics)),
                  ("Coverage", coverage),
                  ("League scores in the common window", metrics.loc[metrics["Window"].eq("Common")
                   & metrics["Period"].eq("All") & metrics["Scope"].eq("League")])])
    print(accuracy_comparison(metrics).to_string())
    print(f"Common end date: {common_end.date()}; saved top-five and tier reports.")


if __name__ == "__main__":
    main()
