from importlib import import_module
import unittest

import numpy as np
import pandas as pd


_step_03 = import_module('03_train_models')
make_model = _step_03.make_model

_step_07 = import_module('07_train_market_models')
fit_same_architecture = _step_07.fit_same_architecture
select_train_and_test = _step_07.select_train_and_test


class MarketModelTrainingTests(unittest.TestCase):
    def test_split_requires_prepared_targets_and_valid_test_odds(self):
        table = pd.DataFrame({
            "Division": ["E0"] * 7,
            "MatchDate": pd.to_datetime(["2000-06-30", "2000-07-01", "2023-07-01",
                                         "2024-12-05", "2024-12-06", "2024-01-01", "2026-07-01"]),
            "FTResult": ["H", "H", "H", "H", "H", None, "H"],
            "MarketHome": [np.nan, np.nan, 2, 2, 2, 2, 2],
            "MarketDraw": [3] * 7, "MarketAway": [4] * 7,
        })
        train, test = select_train_and_test(table, ["E0"], as_of="2026-11-23")
        self.assertEqual(train.index.tolist(), [1])
        self.assertEqual(test.index.tolist(), [2, 3, 4, 6])
        _, earlier_test = select_train_and_test(table, ["E0"], as_of="2024-12-05")
        self.assertEqual(earlier_test.index.tolist(), [2, 3])

    def test_retraining_keeps_settings_and_fits_new_preprocessing(self):
        X = pd.DataFrame({"stat": [1., 2, 3, 4, 5, 6], "elo": [1400., 1500, 1600, 1700, 1800, 1900]})
        y = pd.Series(["A", "D", "H", "H", "A", "D"])
        original = make_model("Logistic regression", {"C": .1}).fit(X, y)
        original_coefficients = original.named_steps["logisticregression"].coef_.copy()
        augmented = X.assign(market_probability=[.2, .3, .6, .8, .4, .3])
        fitted = fit_same_architecture(original, augmented, y)
        self.assertEqual(fitted.n_features_in_, 3)
        self.assertEqual(original.n_features_in_, 2)
        self.assertEqual(fitted.named_steps["logisticregression"].C, .1)
        np.testing.assert_array_equal(original.named_steps["logisticregression"].coef_, original_coefficients)


if __name__ == "__main__":
    unittest.main()
