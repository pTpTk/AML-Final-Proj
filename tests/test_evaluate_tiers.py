import unittest

import numpy as np
import pandas as pd

from src.evaluation import (
    LEAGUE_PATH, MODEL_NAMES, TIERS, common_end_date, prediction_scores,
    select_matches, summarize_predictions,
)
from src.models import LABELS


class TierEvaluationTests(unittest.TestCase):
    def test_groups_follow_the_ranking_rule_among_all_available_european_top_divisions(self):
        rankings = pd.read_csv(LEAGUE_PATH).sort_values("UEFARank2023")
        self.assertEqual(len(rankings), 21)
        self.assertTrue(rankings["Division"].is_unique)
        for tier, expected in [
            ("Top", rankings.iloc[:5]), ("Mid", rankings.iloc[5:10]), ("Lower", rankings.iloc[-5:]),
        ]:
            actual = rankings.loc[rankings["Tier"].eq(tier)]
            self.assertEqual(actual["Division"].tolist(), expected["Division"].tolist())
        self.assertEqual(rankings.loc[rankings["Tier"].isin(TIERS), "Division"].nunique(), 15)

    def test_selection_respects_cutoff_labels_and_available_leagues(self):
        table = pd.DataFrame([
            ("AUT", "2023-06-30", "H"), ("AUT", "2023-07-01", "D"),
            ("FIN", "2024-12-05", "A"), ("E1", "2024-01-01", "H"),
            ("FIN", "2024-12-06", "H"), ("AUT", "2024-01-01", None),
        ], columns=["Division", "MatchDate", "FTResult"])
        table["MatchDate"] = pd.to_datetime(table["MatchDate"])
        selected = select_matches(table, ["AUT", "FIN"], as_of="2024-12-05")
        self.assertEqual(selected.index.tolist(), [1, 2])

    def test_common_end_uses_the_earliest_last_match_and_requires_all_leagues(self):
        table = pd.DataFrame({
            "Division": ["AUT", "AUT", "FIN"],
            "MatchDate": pd.to_datetime(["2023-07-04", "2024-12-05", "2024-12-08"]),
        })
        self.assertEqual(common_end_date(table, ["AUT", "FIN"]), pd.Timestamp("2024-12-05"))
        with self.assertRaisesRegex(ValueError, "Missing evaluation matches"):
            common_end_date(table, ["AUT", "FIN", "SWE"])

    def test_scores_include_draw_recall_and_handle_a_single_outcome(self):
        scores = prediction_scores(["H", "D", "A"], ["H", "H", "A"], [
            [0.1, 0.2, 0.7], [0.1, 0.3, 0.6], [0.8, 0.1, 0.1],
        ])
        self.assertAlmostEqual(scores["Accuracy"], 2 / 3)
        self.assertEqual(scores["DrawRecall"], 0)
        single = prediction_scores(["H"], ["D"], [[0.1, 0.6, 0.3]])
        self.assertEqual(single["BalancedAccuracy"], 0)
        self.assertTrue(np.isfinite(single["LogLoss"]))

    def test_common_window_and_equal_league_weighting_use_the_exported_predictions(self):
        # Large league is always correct; small league is wrong. A later game
        # must enter Available scores but must not enter the Common window.
        predictions = pd.DataFrame({
            "Division": ["N1", "N1", "N1", "AUT", "AUT"],
            "Tier": ["Mid"] * 5,
            "MatchDate": pd.to_datetime(["2023-08-01"] * 4 + ["2025-08-01"]),
            "FTResult": ["H", "H", "H", "A", "H"],
        })
        for name in MODEL_NAMES:
            slug = name.lower().replace(" ", "_")
            predictions[f"{slug}_prediction"] = "H"
            for label, value in zip(LABELS, [0.2, 0.2, 0.6]):
                predictions[f"{slug}_probability_{label}"] = value
        metrics = summarize_predictions(predictions, pd.Timestamp("2024-12-05"))
        pooled = metrics.loc[
            metrics["Scope"].eq("Tier") & metrics["Period"].eq("All")
            & metrics["Model"].eq("Logistic regression")
        ].set_index("Window")
        self.assertEqual(pooled.loc["Common", "Matches"], 4)
        self.assertEqual(pooled.loc["Common", "Accuracy"], 0.75)
        self.assertEqual(pooled.loc["Common", "MeanLeagueAccuracy"], 0.5)
        self.assertEqual(pooled.loc["Available", "Matches"], 5)
        self.assertEqual(pooled.loc["Available", "Accuracy"], 0.8)
        self.assertEqual(pooled.loc["Available", "MeanLeagueAccuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
