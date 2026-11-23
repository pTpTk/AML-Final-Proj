"""Build Elo and five-game histories from the corrected chronological matches."""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "data" / "MatchesChronological.csv"
TRAINING_PATH = DATA_PATH.with_name("MatchTraining.csv")

STAT_COLUMNS = {
    "Goals": ("FTHome", "FTAway"),
    "HalfTimeGoals": ("HTHome", "HTAway"),
    "Shots": ("HomeShots", "AwayShots"),
    "ShotsOnTarget": ("HomeTarget", "AwayTarget"),
    "Fouls": ("HomeFouls", "AwayFouls"),
    "Corners": ("HomeCorners", "AwayCorners"),
    "YellowCards": ("HomeYellow", "AwayYellow"),
    "RedCards": ("HomeRed", "AwayRed"),
}


def load_match_history(csv_path=DATA_PATH):
    """Return one row per match with separate home and away history columns.

    Previous1 is the most recent game; Previous5 is the fifth most recent.
    History includes games in either venue and across seasons and divisions.
    Points encodes each historical result: win=3, draw=1, loss=0.
    IsHome describes the team's venue in that historical game (1=home, 0=away).
    DaysBeforeMatch is the gap from the historical game to the current game.
    Missing games and unavailable statistics remain NaN.
    Division, MatchDate, HomeTeam, and AwayTeam are metadata. FTResult is the
    current match's target; exclude these five columns from numeric model inputs.
    """
    match_columns = [
        "Division", "MatchDate", "HomeTeam", "AwayTeam",
        "HomeElo", "AwayElo", "FTResult",
    ]
    # Read only identifiers, pre-match Elo, the target, and real match statistics.
    # Raw statistics are needed to build history; they are never current inputs.
    stat_columns = [column for pair in STAT_COLUMNS.values() for column in pair]
    matches = pd.read_csv(
        csv_path,
        usecols=match_columns + ["MatchTime"] + stat_columns,
        parse_dates=["MatchDate"],
        dtype={"MatchTime": "str"},
    )
    # Parse dates, keep kickoff times as strings, then put the matches in order.
    # Unknown times sort first on their date. MatchId is the sorted row number.
    matches = matches.sort_values(
        ["MatchDate", "MatchTime"], kind="stable", na_position="first"
    ).reset_index(drop=True)
    matches.index.name = "MatchId"

    # Put every game into each team's history, regardless of where they played.
    team_games = []
    for side, opponent_side in [("Home", "Away"), ("Away", "Home")]:
        # Make two records per match: one from each participating team's view.
        games = matches[["MatchDate", "MatchTime"]].copy()
        games["Team"] = matches[f"{side}Team"]
        games["IsHome"] = int(side == "Home")
        games["Elo"] = matches[f"{side}Elo"]
        games["OpponentElo"] = matches[f"{opponent_side}Elo"]
        # Translate the match's H/D/A target into points for this specific team.
        win_result = "H" if side == "Home" else "A"
        loss_result = "A" if side == "Home" else "H"
        games["Points"] = matches["FTResult"].map(
            {win_result: 3, "D": 1, loss_result: 0}
        )
        for stat, (home_column, away_column) in STAT_COLUMNS.items():
            # For/Against always follow this team, even when it played away.
            own_column = home_column if side == "Home" else away_column
            opponent_column = away_column if side == "Home" else home_column
            games[f"{stat}For"] = matches[own_column]
            games[f"{stat}Against"] = matches[opponent_column]
        # Preserve MatchId as a column so the history can be joined back later.
        team_games.append(games.reset_index())

    # Mix home and away appearances into one chronological record per team.
    team_games = pd.concat(team_games, ignore_index=True).sort_values(
        ["MatchDate", "MatchTime"], kind="stable", na_position="first"
    )
    history_columns = [
        "MatchDate", "IsHome", "Elo", "OpponentElo", "Points",
    ] + [
        f"{stat}{direction}"
        for stat in STAT_COLUMNS
        for direction in ["For", "Against"]
    ]
    # Group by team only: promotion and a new season do not reset its history.
    grouped_games = team_games.groupby("Team")[history_columns]
    history = [team_games[["MatchId", "IsHome"]]]
    for previous in range(1, 6):
        # Shift before joining so the current game's statistics are excluded.
        past_games = grouped_games.shift(previous)
        # Replace the old date with elapsed calendar days, a numeric input.
        past_games["MatchDate"] = (
            team_games["MatchDate"] - past_games["MatchDate"]
        ).dt.days
        past_games = past_games.rename(columns={"MatchDate": "DaysBeforeMatch"})
        history.append(past_games.add_prefix(f"Previous{previous}_"))
    # Keep each lag in its own columns rather than averaging the five games.
    history = pd.concat(history, axis=1)

    # Start the output with current-match metadata, Elo, and the target only.
    result = [matches[match_columns]]
    history_counts = {}
    for side, is_home in [("Home", 1), ("Away", 0)]:
        # Route each team's history to its role in the current match.
        team_history = history[history["IsHome"] == is_home]
        team_history = team_history.set_index("MatchId").drop(columns="IsHome")
        result.append(team_history.add_prefix(f"{side}_"))
        # Count actual historical games; missing statistics still remain NaN.
        previous_days = [f"Previous{n}_DaysBeforeMatch" for n in range(1, 6)]
        history_counts[f"{side}_HistoryCount"] = team_history[previous_days].notna().sum(axis=1)
    result.append(pd.DataFrame(history_counts, index=matches.index))
    # Assemble the complete flattened table without inserting columns repeatedly.
    return pd.concat(result, axis=1)


def main():
    table = load_match_history()
    table.to_csv(TRAINING_PATH, index=False)
    print(f"Saved {len(table):,} matches and {table.shape[1]} columns to {TRAINING_PATH}.")


if __name__ == "__main__":
    main()
