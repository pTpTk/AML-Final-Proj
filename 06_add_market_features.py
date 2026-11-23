"""Add current-match and five-game historical market probabilities for each team."""

from importlib import import_module

import numpy as np
import pandas as pd


_step_01 = import_module('01_clean_data')
CORRECTED_PATH = _step_01.CORRECTED_PATH
MARKET_COLUMNS = _step_01.MARKET_COLUMNS

_step_02 = import_module('02_build_features')
DATA_PATH = _step_02.DATA_PATH
TRAINING_PATH = _step_02.TRAINING_PATH

_step_03 = import_module('03_train_models')
FIELD_PATH = _step_03.FIELD_PATH
MARKET_TRAINING_PATH = DATA_PATH.with_name("MarketTraining.csv")
MARKET_FIELD_PATH = DATA_PATH.parents[1] / "docs" / "market_training_fields.csv"
PROBABILITY_COLUMNS = ["MarketHomeProbability", "MarketDrawProbability", "MarketAwayProbability"]
HISTORICAL_PROBABILITIES = ["MarketWinProbability", "MarketDrawProbability", "MarketLossProbability"]


def market_features(matches):
    """Return 33 probabilities: current H/D/A and each team's five prior W/D/L quotes."""
    # Keep exactly the same ordering and row index as load_match_history.
    matches = matches.sort_values(["MatchDate", "MatchTime"], kind="stable", na_position="first").reset_index(drop=True)
    odds = matches[MARKET_COLUMNS].to_numpy(dtype=float)
    valid = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    probabilities = np.full(odds.shape, np.nan)
    inverse = 1 / odds[valid]
    probabilities[valid] = inverse / inverse.sum(axis=1, keepdims=True)
    current = pd.DataFrame(probabilities, columns=PROBABILITY_COLUMNS)
    team_games = []
    for side, is_home in [("Home", 1), ("Away", 0)]:
        games = matches[["MatchDate", "MatchTime"]].copy()
        games["MatchId"] = matches.index
        games["Team"] = matches[f"{side}Team"]
        games["IsHome"] = is_home
        # Historical prices follow the team's perspective, regardless of venue.
        games["MarketWinProbability"] = probabilities[:, 0 if is_home else 2]
        games["MarketDrawProbability"] = probabilities[:, 1]
        games["MarketLossProbability"] = probabilities[:, 2 if is_home else 0]
        team_games.append(games)
    games = pd.concat(team_games, ignore_index=True).sort_values(
        ["MatchDate", "MatchTime"], kind="stable", na_position="first",
    )
    grouped = games.groupby("Team")[HISTORICAL_PROBABILITIES]
    history = [games[["MatchId", "IsHome"]]]
    for previous in range(1, 6):
        # Shift excludes the current game's quote from each historical block.
        history.append(grouped.shift(previous).add_prefix(f"Previous{previous}_"))
    history = pd.concat(history, axis=1)
    result = [current]
    for side, is_home in [("Home", 1), ("Away", 0)]:
        rows = history.loc[history["IsHome"].eq(is_home)].set_index("MatchId").drop(columns="IsHome")
        result.append(rows.add_prefix(f"{side}_"))
    return pd.concat(result, axis=1)


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
