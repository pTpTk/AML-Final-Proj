import unittest

import numpy as np
import pandas as pd

from src.data_sources import add_match_keys, team_key
from src.data_sources import MARKET_COLUMNS, attach_market_odds
from src.evaluation import MODEL_NAMES
from src.odds_evaluation import add_odds_predictions, score_comparisons
from src.models import LABELS


class MarketEvaluationTests(unittest.TestCase):
    def test_market_favourite_ties_and_probability_class_order(self):
        games = pd.DataFrame([[1.5, 4, 7], [4, 3.5, 1.8], [2.5, 3, 2.5], [4, 2, 5]],
                             columns=MARKET_COLUMNS)
        paired = add_odds_predictions(games, MARKET_COLUMNS, "Market average")
        self.assertEqual(paired.market_average_prediction.tolist(), ["H", "A", "H", "D"])
        self.assertEqual(paired.TiedFavourite.tolist(), [False, False, True, False])
        probabilities = paired[[f"market_average_probability_{label}" for label in LABELS]]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertAlmostEqual(probabilities.iloc[0, 0], (1 / 7) / (1 / 1.5 + 1 / 4 + 1 / 7))
        self.assertEqual(games.columns.tolist(), MARKET_COLUMNS)

    def test_invalid_odds_never_enter_scoring(self):
        for invalid in [np.nan, np.inf, 0, -2, 1]:
            with self.subTest(odds=invalid), self.assertRaisesRegex(ValueError, "finite and greater than one"):
                add_odds_predictions(pd.DataFrame([[invalid, 3, 4]], columns=MARKET_COLUMNS),
                                     MARKET_COLUMNS, "Market average")

    def test_team_matching_ignores_formatting_without_merging_distinct_clubs(self):
        self.assertEqual(team_key("Saint-Étienne"), team_key("Saint Etienne"))
        self.assertEqual(team_key("Preußen Münster"), team_key("Preussen Munster"))
        self.assertEqual(team_key("U.A.N.L.- Tigres"), team_key("Tigres UANL"))
        self.assertNotEqual(team_key("Man City"), team_key("Man United"))

    def make_source(self, dates, odds):
        sources = pd.DataFrame({
            "Division": ["AUT"] * len(dates), "SourceDate": pd.to_datetime(dates),
            "HomeTeam": ["Tirol"] * len(dates), "AwayTeam": ["Wolfsberger AC"] * len(dates),
            "SourceResult": ["H"] * len(dates), "SourceFile": ["AUT.csv"] * len(dates),
            "SourceURL": ["https://example.test/AUT.csv"] * len(dates),
        })
        for position, column in enumerate(MARKET_COLUMNS):
            sources[column] = [row[position] for row in odds]
        return add_match_keys(sources).drop(columns=["HomeTeam", "AwayTeam"])

    def make_match(self, date="2023-07-04", odds=(2.69, 3.53, 2.43)):
        predictions = pd.DataFrame({
            "Division": ["AUT"], "MatchDate": pd.to_datetime([date]),
            "HomeTeam": ["Tirol"], "AwayTeam": ["Wolfsberger AC"], "FTResult": ["H"],
        })
        raw = predictions.copy()
        raw[["OddHome", "OddDraw", "OddAway"]] = [odds]
        return predictions, raw

    def test_transposed_source_match_requires_all_odds_and_preserves_model_date(self):
        predictions, raw = self.make_match()
        sources = self.make_source(["2023-04-07"], [(2.69, 3.53, 2.43)])
        joined = attach_market_odds(predictions, raw, sources)
        self.assertTrue(joined.loc[0, "Eligible"])
        self.assertEqual(joined.loc[0, "SourceDate"], pd.Timestamp("2023-04-07"))
        self.assertEqual(joined.loc[0, "MatchDate"], pd.Timestamp("2023-07-04"))
        self.assertEqual(joined.loc[0, "DateMatch"], "Transposed; odds verified")
        self.assertEqual(predictions.loc[0, "MatchDate"], pd.Timestamp("2023-07-04"))
        raw.loc[0, "OddAway"] = 2.44
        unmatched = attach_market_odds(predictions, raw, sources)
        self.assertFalse(unmatched.loc[0, "Eligible"])
        self.assertEqual(unmatched.loc[0, "Status"], "No verified source match")

    def test_a_missing_result_is_not_marked_eligible_in_the_source_audit(self):
        predictions, raw = self.make_match()
        predictions["FTResult"] = None
        raw["FTResult"] = None
        sources = self.make_source(["2023-04-07"], [(2.69, 3.53, 2.43)])
        sources["SourceResult"] = None
        joined = attach_market_odds(predictions, raw, sources)
        self.assertFalse(joined.loc[0, "Eligible"])
        self.assertEqual(joined.loc[0, "Status"], "Missing/invalid result")
        self.assertTrue(pd.notna(joined.loc[0, "SourceDate"]))

    def test_ambiguous_candidates_and_changed_targets_are_rejected(self):
        predictions, raw = self.make_match(date="2024-01-02")
        sources = self.make_source(["2024-01-02", "2024-02-01"], [(2.69, 3.53, 2.43)] * 2)
        with self.assertRaisesRegex(ValueError, "Ambiguous source matches"):
            attach_market_odds(predictions, raw, sources)
        raw.loc[0, "FTResult"] = "A"
        with self.assertRaisesRegex(ValueError, "saved prediction targets"):
            attach_market_odds(predictions, raw, sources)

    def test_optional_score_verification_identifies_a_game_with_missing_stored_prices(self):
        predictions, raw = self.make_match(odds=(np.nan, np.nan, np.nan))
        raw["FTHome"], raw["FTAway"] = 4, 0
        sources = self.make_source(["2023-04-07"], [(2.69, 3.53, 2.43)])
        sources["SourceHomeGoals"], sources["SourceAwayGoals"] = 4, 0
        self.assertFalse(attach_market_odds(predictions, raw, sources).loc[0, "Eligible"])
        verified = attach_market_odds(predictions, raw, sources, verify_missing_odds_with_goals=True)
        self.assertTrue(verified.loc[0, "Eligible"])
        raw["FTHome"] = 3
        rejected = attach_market_odds(predictions, raw, sources, verify_missing_odds_with_goals=True)
        self.assertFalse(rejected.loc[0, "Eligible"])

    def test_pooled_and_equal_league_scores_use_identical_market_and_model_rows(self):
        matches = pd.DataFrame({
            "Division": ["N1", "N1", "N1", "AUT"], "Tier": ["Mid"] * 4,
            "MatchDate": pd.to_datetime(["2024-01-01"] * 4), "CommonWindow": [True] * 4,
            "FTResult": ["H", "H", "H", "A"],
            "MarketHome": [1.5] * 4, "MarketDraw": [4.0] * 4, "MarketAway": [7.0] * 4,
        })
        matches = add_odds_predictions(matches, MARKET_COLUMNS, "Market average")
        for name in MODEL_NAMES:
            slug = name.lower().replace(" ", "_")
            matches[f"{slug}_prediction"] = "H"
            for label, probability in zip(LABELS, [0.2, 0.2, 0.6]):
                matches[f"{slug}_probability_{label}"] = probability
        metrics = score_comparisons(matches, "Market average")
        overall = metrics.loc[metrics.Window.eq("Common") & metrics.Period.eq("All") & metrics.Scope.eq("Tier")]
        self.assertEqual(len(overall), 5)
        self.assertTrue(overall.Matches.eq(4).all())
        self.assertTrue(overall.Accuracy.eq(0.75).all())
        self.assertTrue(overall.MeanLeagueAccuracy.eq(0.5).all())
        self.assertTrue(overall.GapToBenchmark.eq(0).all())


if __name__ == "__main__":
    unittest.main()
