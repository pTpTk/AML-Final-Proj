from importlib import import_module
import unittest

import numpy as np
import pandas as pd


_step_01 = import_module('01_clean_data')
MARKET_COLUMNS = _step_01.MARKET_COLUMNS
choose_average_odds = _step_01.choose_average_odds

_step_06 = import_module('06_add_market_features')
PROBABILITY_COLUMNS = _step_06.PROBABILITY_COLUMNS
market_features = _step_06.market_features


class MarketTrainingFeatureTests(unittest.TestCase):
    def test_average_priority_uses_complete_triplets_and_never_individual_bookmakers(self):
        source = pd.DataFrame({
            "AvgCH": [2, np.nan, 2, np.nan], "AvgCD": [3, np.nan, np.nan, np.nan],
            "AvgCA": [4, np.nan, 4, np.nan],
            "AvgH": [5, 5, 5, np.nan], "AvgD": [6, 6, 6, np.nan], "AvgA": [7, 7, 7, np.nan],
            "BbAvH": [8, 8, 8, np.nan], "BbAvD": [9, 9, 9, np.nan], "BbAvA": [10, 10, 10, np.nan],
            "B365H": [1.5] * 4, "B365D": [3.5] * 4, "B365A": [6] * 4,
        })
        result = choose_average_odds(source)
        self.assertEqual(result.MarketTiming.tolist(), ["Closing", "Pre-closing", "Pre-closing", "Unavailable"])
        np.testing.assert_array_equal(result.loc[0, MARKET_COLUMNS], [2, 3, 4])
        np.testing.assert_array_equal(result.loc[2, MARKET_COLUMNS], [5, 6, 7])
        self.assertTrue(result.loc[3, MARKET_COLUMNS].isna().all())
        legacy = choose_average_odds(source.drop(columns=["AvgCH", "AvgCD", "AvgCA", "AvgH", "AvgD", "AvgA"]))
        self.assertEqual(legacy.loc[0, "MarketTiming"], "Legacy pre-closing")

    def make_games(self):
        return pd.DataFrame({
            "MatchDate": pd.to_datetime(["2022-01-03", "2022-01-01", "2022-01-02"]),
            "MatchTime": ["15:00:00"] * 3, "HomeTeam": ["A", "A", "B"], "AwayTeam": ["B", "B", "A"],
            "MarketHome": [1.5, 2.0, 3.0], "MarketDraw": [4.0, 3.0, 4.0], "MarketAway": [8.0, 4.0, 2.0],
        })

    def test_history_tracks_team_perspective_across_venues_and_excludes_current_quotes(self):
        games = self.make_games()
        features = market_features(games)
        self.assertEqual(features.shape, (3, 33))
        np.testing.assert_allclose(features[PROBABILITY_COLUMNS].sum(axis=1), 1)
        self.assertTrue(features.loc[0, features.columns.str.startswith("Home_Previous")].isna().all())
        # Current home team A played away in the previous game, and at home two games ago.
        self.assertAlmostEqual(features.loc[2, "Home_Previous1_MarketWinProbability"], (1 / 2) / (1 / 3 + 1 / 4 + 1 / 2))
        self.assertAlmostEqual(features.loc[2, "Home_Previous2_MarketWinProbability"], (1 / 2) / (1 / 2 + 1 / 3 + 1 / 4))
        self.assertEqual(features.loc[2, "Away_Previous1_MarketWinProbability"], features.loc[2, "Home_Previous1_MarketLossProbability"])
        changed = games.copy()
        changed.loc[changed.MatchDate.eq("2022-01-03"), MARKET_COLUMNS] = [5, 3, 1.5]
        updated = market_features(changed)
        historical = features.columns.str.contains("Previous")
        pd.testing.assert_frame_equal(features.loc[:, historical], updated.loc[:, historical])
        pd.testing.assert_frame_equal(features.iloc[:2], updated.iloc[:2])
        self.assertNotEqual(features.loc[2, "MarketHomeProbability"], updated.loc[2, "MarketHomeProbability"])

    def test_missing_prices_remain_missing_and_do_not_create_fake_history(self):
        games = self.make_games()
        games.loc[games.MatchDate.eq("2022-01-01"), "MarketDraw"] = np.nan
        features = market_features(games)
        self.assertTrue(features.loc[0, PROBABILITY_COLUMNS].isna().all())
        self.assertTrue(pd.isna(features.loc[1, "Home_Previous1_MarketDrawProbability"]))
        self.assertTrue(pd.isna(features.loc[2, "Home_Previous5_MarketWinProbability"]))


if __name__ == "__main__":
    unittest.main()
