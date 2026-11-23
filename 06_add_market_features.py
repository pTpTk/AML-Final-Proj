"""Add current-match and five-game historical market probabilities for each team."""

import pandas as pd

from src.data_sources import CORRECTED_PATH
from src.features import market_features, MARKET_TRAINING_PATH, MARKET_FIELD_PATH, TRAINING_PATH
from src.models import FIELD_PATH


def main():
    raw = pd.read_csv(CORRECTED_PATH, parse_dates=["MatchDate"], dtype={"MatchTime": "str"})
    base = pd.read_csv(TRAINING_PATH, parse_dates=["MatchDate"])
    ordered = raw.sort_values(["MatchDate", "MatchTime"], kind="stable", na_position="first").reset_index(drop=True)
    keys = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTResult"]
    if not base[keys].equals(ordered[keys]):
        raise ValueError("Base features do not match the corrected data; rerun step 02.")
    additions = market_features(raw)
    pd.concat([base, additions], axis=1).to_csv(MARKET_TRAINING_PATH, index=False)
    fields = pd.read_csv(FIELD_PATH)
    extra = pd.DataFrame([
        {"field": field, "type": "float", "role": "feature", "description":
         "Normalized inverse odds supplied in Matches.csv: current H/D/A or historical team W/D/L. "
         "Previous1 is most recent; missing odds remain missing."}
        for field in additions
    ])
    pd.concat([fields, extra], ignore_index=True).to_csv(MARKET_FIELD_PATH, index=False)
    print(f"Added {len(additions.columns)} market inputs; saved {MARKET_TRAINING_PATH}.")


if __name__ == "__main__":
    main()
