import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data_sources
from src.data_quality import audit_raw, paired_test_mask, COUNT_COLUMNS, DETAIL_COLUMNS


class DataProvenanceTests(unittest.TestCase):
    def games(self):
        raw = pd.DataFrame({"Division": ["E0"] * 3, "MatchDate": ["2022-01-01", "2022-01-02", "2022-01-03"],
                            "MatchTime": ["15:00:00"] * 3, "HomeTeam": ["A", "B", "C"],
                            "AwayTeam": ["X"] * 3, "FTResult": ["H", "H", "H"], "HTResult": ["D"] * 3,
                            "HomeElo": [1500., np.nan, 1400], "AwayElo": [1400.] * 3,
                            "OddHome": [2., np.nan, 1.], "OddDraw": [3., np.nan, 3.], "OddAway": [4., np.nan, 4.]})
        for column in set(COUNT_COLUMNS + DETAIL_COLUMNS):
            raw[column] = 0.
        raw["FTHome"] = 1.
        return raw

    def test_external_dates_never_fill_or_replace_supplied_odds(self):
        raw = self.games()
        source = pd.DataFrame({"SourceDate": pd.to_datetime(["2022-02-01", "2022-02-02", "2022-02-03"]),
                               "SourceResult": ["H", "H", "A"], "MarketHome": [9.] * 3,
                               "MarketDraw": [9.] * 3, "MarketAway": [9.] * 3})
        prepared = data_sources.prepare_matches(raw, source)
        np.testing.assert_allclose(prepared.loc[0, data_sources.MARKET_COLUMNS].to_numpy(dtype=float), [2, 3, 4])
        self.assertTrue(prepared.loc[1:, data_sources.MARKET_COLUMNS].isna().all().all())
        self.assertTrue(pd.isna(prepared.loc[2, "FTResult"]))
        self.assertEqual(prepared.MatchDate.tolist(), source.SourceDate.tolist())
        pd.testing.assert_frame_equal(raw[data_sources.RAW_COLUMNS], prepared[data_sources.RAW_COLUMNS])

    def test_raw_only_preparation_needs_no_external_record(self):
        raw = self.games()
        prepared = data_sources.prepare_matches(raw)
        self.assertNotIn("SourceResult", prepared)
        self.assertEqual(prepared.MatchDate.tolist(), pd.to_datetime(raw.MatchDate).tolist())
        self.assertEqual(prepared.FTResult.tolist(), raw.FTResult.tolist())

    def test_missing_cells_are_separate_from_observed_domain_errors(self):
        raw = self.games()
        raw.loc[0, "HomeShots"] = -1
        fields, summary, _ = audit_raw(raw)
        fields = fields.set_index("Field")
        self.assertEqual(fields.loc["HomeElo", "Missing"], 1)
        self.assertEqual(fields.loc["HomeElo", "InvalidObserved"], 0)
        self.assertEqual(fields.loc["HomeShots", "InvalidObserved"], 1)
        self.assertEqual(fields.loc["OddHome", "Missing"], 1)
        self.assertEqual(fields.loc["OddHome", "InvalidObserved"], 1)
        summary = summary.set_index("Check")
        self.assertEqual(summary.loc["Complete but invalid odds triplet", "Matches"], 1)

    def test_download_bootstraps_empty_folder_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "new_sources"
            body = b"Date,HomeTeam,AwayTeam\n01/01/2022,A,B\n"
            with patch.object(data_sources, "SOURCE_DIR", folder), \
                 patch.object(data_sources, "DOWNLOAD_MANIFEST", folder / "download_manifest.csv"), \
                 patch.object(data_sources, "urlopen", return_value=io.BytesIO(body)) as fetch:
                result = data_sources.download_history({"test.csv": "https://example.test/test.csv"})
                self.assertEqual(result.loc[0, "Status"], "Downloaded")
                self.assertEqual(result.loc[0, "SHA256"], hashlib.sha256(body).hexdigest())
                self.assertEqual((folder / "test.csv").read_bytes(), body)
                cached = data_sources.download_history({"test.csv": "https://example.test/test.csv"})
                self.assertEqual(cached.loc[0, "Status"], "Cached")
                self.assertEqual(fetch.call_count, 1)

    def test_paired_comparison_requires_both_date_versions_to_be_held_out(self):
        table = pd.DataFrame({"RawDate": pd.to_datetime(["2023-07-04", "2023-04-07", "2023-08-01", "2023-08-01"]),
                              "SourceDate": pd.to_datetime(["2023-04-07", "2023-07-04", "2023-08-01", "2023-08-01"]),
                              "FTResult": ["H"] * 4, "SourceResult": ["H", "H", "H", "A"]})
        self.assertEqual(paired_test_mask(table, pd.Timestamp("2024-10-06")).tolist(), [False, False, True, False])


if __name__ == "__main__":
    unittest.main()
