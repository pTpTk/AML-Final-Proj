from importlib import import_module
import unittest

from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd


_step_03 = import_module('03_train_models')
evaluate = _step_03.evaluate
make_model = _step_03.make_model
season_labels = _step_03.season_labels
split_data = _step_03.split_data


class ModelTrainingTests(unittest.TestCase):
    def test_split_respects_dates_division_labels_and_future_cutoff(self):
        table = pd.DataFrame({
            "MatchDate": pd.to_datetime([
                "2000-06-30", "2000-07-01", "2023-06-30", "2023-07-01",
                "2023-08-12", "2024-01-01", "2026-08-31", "2027-01-01",
            ]),
            "Division": ["E0", "E0", "SP1", "E0", "E1", "E0", "E0", "E0"],
            "FTResult": ["H", "H", "D", "A", "H", None, "D", "H"],
        })
        train, test = split_data(table, as_of="2026-11-11")
        self.assertEqual(train.index.tolist(), [1, 2])
        self.assertEqual(test.index.tolist(), [3])
        self.assertLess(train["MatchDate"].max(), test["MatchDate"].min())

    def test_english_season_labels_change_in_july(self):
        dates = pd.Series(pd.to_datetime(["2023-06-30", "2023-07-01", "2024-05-19", "2026-08-31"]))
        self.assertEqual(season_labels(dates).tolist(), ["2022/23", "2023/24", "2023/24", "2026/27"])

    def test_imputation_uses_training_data_and_stays_fixed_at_prediction(self):
        X = pd.DataFrame({"Elo": [1., 2., 3., 4., 5., 6.], "Shots": [1., np.nan, 2., 4., 3., 5.]}).astype("float32")
        y = pd.Series(["A", "D", "H", "A", "D", "H"])
        model = make_model("Logistic regression", {"C": 1.0})
        model.fit(X, y)
        imputer = model.named_steps["simpleimputer"]
        np.testing.assert_allclose(imputer.statistics_, [3.5, 3.0])
        model.predict(pd.DataFrame({"Elo": [1000000.], "Shots": [np.nan]}))
        np.testing.assert_allclose(imputer.statistics_, [3.5, 3.0])
        self.assertEqual(imputer.indicator_.features_.tolist(), [1])
        _, probabilities, _ = evaluate(model, X, y)
        self.assertEqual(probabilities.dtype, np.dtype("float64"))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-14)

    def test_baseline_probabilities_and_scores_have_the_expected_class_order(self):
        X = pd.DataFrame({"Elo": [1., 2., 3., 4., 5., 6.]})
        y = pd.Series(["A", "A", "D", "H", "H", "H"])
        model = make_model("Prior baseline", {})
        model.fit(X, y)
        predicted, probabilities, scores = evaluate(model, X, y)
        self.assertEqual(predicted.tolist(), ["H"] * 6)
        np.testing.assert_allclose(probabilities[0], [2 / 6, 1 / 6, 3 / 6])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertEqual(scores["Accuracy"], 0.5)
        expected_loss = -(2 * np.log(2 / 6) + np.log(1 / 6) + 3 * np.log(3 / 6)) / 6
        self.assertAlmostEqual(scores["LogLoss"], expected_loss)

    def test_boosting_accepts_missing_values_without_imputation(self):
        X = pd.DataFrame({"Elo": [1., 2., np.nan, 4., 5., 6.]})
        y = pd.Series(["A", "A", "D", "H", "D", "H"])
        model = make_model("Histogram gradient boosting", {"max_leaf_nodes": 15, "l2_regularization": 1.0})
        with threadpool_limits(limits=2):
            model.fit(X, y)
            probabilities = model.predict_proba(X)
        self.assertFalse(model.early_stopping)
        self.assertTrue(np.isfinite(probabilities).all())

    def test_naive_bayes_uses_stable_precision_for_many_log_likelihoods(self):
        random = np.random.default_rng(42)
        X = pd.DataFrame(random.normal(size=(90, 214)).astype("float32"))
        y = pd.Series(["A", "D", "H"] * 30)
        model = make_model("Gaussian Naive Bayes", {"var_smoothing": 10.0})
        model.fit(X, y)
        probabilities = model.predict_proba(X)
        self.assertEqual(probabilities.dtype, np.dtype("float64"))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10)

    def test_partial_season_can_lack_an_outcome_that_is_predicted(self):
        X = pd.DataFrame({"Elo": [1., 2., 3., 4.]})
        model = make_model("Prior baseline", {})
        model.fit(X, pd.Series(["A", "A", "D", "H"]))
        # A partial season may have no away wins, even if the model predicts A.
        _, _, scores = evaluate(model, X.iloc[:2], pd.Series(["D", "H"]))
        self.assertEqual(scores["BalancedAccuracy"], 0)
        self.assertEqual(scores["Accuracy"], 0)
        self.assertTrue(np.isfinite(scores["LogLoss"]))


if __name__ == "__main__":
    unittest.main()
