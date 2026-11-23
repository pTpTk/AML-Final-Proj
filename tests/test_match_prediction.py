from importlib import import_module
from pathlib import Path
import csv
import tempfile
import unittest

import pandas as pd


_step_02 = import_module('02_build_features')
STAT_COLUMNS = _step_02.STAT_COLUMNS
load_match_history = _step_02.load_match_history


class MatchHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.csv_path = Path(directory.name) / "matches.csv"
        rows = []
        for day in range(1, 8):
            row = {
                "Division": "F1" if day < 4 else "F2",
                "MatchDate": f"2001-01-{day:02d}",
                "MatchTime": "15:00:00",
                "HomeTeam": "A" if day % 2 else "B",
                "AwayTeam": "B" if day % 2 else "A",
                "HomeElo": 1500 + day,
                "AwayElo": 1400 + day,
                "Form3Home": 3,
                "Form5Home": 6,
                "Form3Away": 1,
                "Form5Away": 4,
                "FTResult": "H",
            }
            for offset, (home_column, away_column) in enumerate(STAT_COLUMNS.values()):
                row[home_column] = day + offset
                row[away_column] = offset
            rows.append(row)
        # Input order must not determine which games are considered previous.
        self.matches = pd.DataFrame(rows[::-1])

    def load(self):
        self.matches.to_csv(self.csv_path, index=False)
        return load_match_history(self.csv_path)

    def test_five_games_follow_team_across_venues_and_divisions(self):
        result = self.load()
        latest = result.iloc[-1]
        self.assertEqual(len(result), 7)
        self.assertEqual(latest["Home_Previous1_DaysBeforeMatch"], 1)
        self.assertEqual(latest["Home_Previous5_DaysBeforeMatch"], 5)
        self.assertEqual(latest["Home_Previous1_IsHome"], 0)
        self.assertEqual(latest["Home_Previous2_IsHome"], 1)
        self.assertEqual(latest["Home_Previous1_GoalsFor"], 0)
        self.assertEqual(latest["Home_Previous1_GoalsAgainst"], 6)
        self.assertEqual(latest["Home_Previous1_Points"], 0)
        self.assertEqual(latest["Away_Previous1_Points"], 3)
        self.assertEqual(latest["Home_Previous1_Elo"], 1406)
        self.assertEqual(latest["Home_Previous1_OpponentElo"], 1506)
        self.assertEqual(latest["HomeElo"], 1507)
        self.assertEqual(latest["AwayElo"], 1407)
        self.assertEqual(latest["Home_HistoryCount"], 5)
        self.assertEqual(result.loc[1, "Home_HistoryCount"], 1)
        self.assertEqual(result.loc[1, "Home_Previous1_GoalsFor"], 0)

    def test_current_and_future_outcomes_do_not_change_pre_match_features(self):
        original = self.load()
        changed_games = self.matches["MatchDate"] >= "2001-01-04"
        self.matches.loc[changed_games, "FTResult"] = "A"
        for home_column, away_column in STAT_COLUMNS.values():
            self.matches.loc[changed_games, [home_column, away_column]] = 99
        changed = self.load()
        features = original.columns.drop("FTResult")
        pd.testing.assert_frame_equal(original.loc[:3, features], changed.loc[:3, features])
        self.assertEqual(changed.loc[3, "FTResult"], "A")
        # The changed result is available to subsequent games.
        self.assertEqual(changed.loc[4, "Home_Previous1_Points"], 3)

    def test_missing_history_and_unavailable_statistics_stay_empty(self):
        self.matches["HomeShots"] = float("nan")
        result = self.load()
        first = result.iloc[0]
        for side in ["Home", "Away"]:
            self.assertEqual(first[f"{side}_HistoryCount"], 0)
            for column in ["Previous1_Points", "Previous5_Elo", "Previous1_IsHome", "Previous1_DaysBeforeMatch"]:
                self.assertTrue(pd.isna(first[f"{side}_{column}"]))
        self.assertTrue(pd.isna(result.iloc[-1]["Away_Previous1_ShotsFor"]))

    def test_draws_are_one_point_for_both_teams_and_odds_are_excluded(self):
        self.matches.loc[self.matches["MatchDate"] == "2001-01-06", "FTResult"] = "D"
        self.matches["OddHome"] = 1.5
        self.matches["C_HTB"] = 0.8
        result = self.load()
        for side in ["Home", "Away"]:
            self.assertEqual(result.iloc[-1][f"{side}_Previous1_Points"], 1)
        self.assertNotIn("OddHome", result.columns)
        self.assertNotIn("C_HTB", result.columns)
        self.assertNotIn("FTHome", result.columns)

    def test_final_schema_has_only_documented_fields_and_numeric_features(self):
        result = self.load()
        schema_path = Path(__file__).resolve().parents[1] / "docs" / "match_prediction_fields.csv"
        with schema_path.open(newline="") as file:
            fields = list(csv.DictReader(file))
        self.assertEqual(result.columns.tolist(), [field["field"] for field in fields])
        self.assertEqual(len(result.columns), 219)
        metadata = ["Division", "MatchDate", "HomeTeam", "AwayTeam", "FTResult"]
        features = result.drop(columns=metadata)
        self.assertEqual(features.shape[1], 214)
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes))
        self.assertFalse(any("Form" in column or "Last5Mean" in column for column in result.columns))

        # Precomputed form and kickoff time may be absent or missing in practice.
        self.matches = self.matches.drop(columns=["Form3Home", "Form5Home", "Form3Away", "Form5Away"])
        self.matches.loc[self.matches.index[::2], "MatchTime"] = float("nan")
        pd.testing.assert_frame_equal(result, self.load())


if __name__ == "__main__":
    unittest.main()
