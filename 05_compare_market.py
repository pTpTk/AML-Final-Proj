"""Compare the market favourite and frozen classifiers on exactly the same matches."""

import numpy as np
import pandas as pd

from src.data_sources import CORRECTED_PATH, RAW_KEYS, MARKET_COLUMNS
from src.market_models import BENCHMARK
from src.models import REPORT_DIR
from src.odds_evaluation import add_odds_predictions, score_comparisons
from src.reporting import write_report, all_periods


def main():
    predictions = pd.read_csv(REPORT_DIR / "league_tier_predictions.csv", parse_dates=["MatchDate"])
    columns = RAW_KEYS + MARKET_COLUMNS
    sources = pd.read_csv(CORRECTED_PATH, usecols=columns, parse_dates=["MatchDate"])
    joined = predictions.merge(sources, on=RAW_KEYS, how="left", validate="one_to_one")
    valid = joined[MARKET_COLUMNS].gt(1).all(axis=1) & np.isfinite(joined[MARKET_COLUMNS]).all(axis=1)
    joined["Status"] = "Eligible"
    joined.loc[~valid, "Status"] = "Missing/invalid supplied odds"
    joined["Eligible"] = joined["Status"].eq("Eligible")
    if not joined["Eligible"].any():
        raise ValueError("No complete valid supplied odds; run steps 01–04 first.")
    paired = add_odds_predictions(joined.loc[joined["Eligible"]], MARKET_COLUMNS, BENCHMARK)
    metrics = score_comparisons(paired, BENCHMARK)
    coverage = joined.groupby(["Division", "Tier", "Status"]).size().rename("Matches").reset_index()
    joined.to_csv(REPORT_DIR / "market_average_match_audit.csv", index=False)
    paired.to_csv(REPORT_DIR / "market_average_predictions.csv", index=False)
    metrics.to_csv(REPORT_DIR / "market_average_metrics.csv", index=False)
    coverage.to_csv(REPORT_DIR / "market_average_coverage.csv", index=False)
    write_report("market_average_evaluation.md", "Supplied odds versus classifiers",
                 "The benchmark uses OddHome/OddDraw/OddAway supplied in Matches.csv. "
                 "Bookmaker identity and closing/pre-closing timing are not established for every row. "
                 "The lowest quote selects the favourite; ties use home/draw/away priority. "
                 "Probabilities are (1/odds) divided by the sum of all three inverse odds. "
                 "Every model is scored on the same eligible games, using prepared dates and "
                 "the common window from step 04. Accuracy does not measure betting returns. "
                 "All windows, calendar-year checks and tie sensitivity are in market_average_metrics.csv.",
                 [("Tier comparison", all_periods(metrics, "Tier")),
                  ("League comparison", all_periods(metrics, "League")),
                  ("Coverage", coverage)])
    print(all_periods(metrics, "Tier")[["Group", "Model", "Matches", "Accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
