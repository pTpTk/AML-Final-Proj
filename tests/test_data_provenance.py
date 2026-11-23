from contextlib import redirect_stdout
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest

import numpy as np
import pandas as pd


data_sources = import_module('00_download_sources')

clean_data = import_module('01_clean_data')

_step_08 = import_module('08_audit_data')
COUNT_COLUMNS = _step_08.COUNT_COLUMNS
DETAIL_COLUMNS = _step_08.DETAIL_COLUMNS
audit_raw = _step_08.audit_raw
paired_test_mask = _step_08.paired_test_mask


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
        prepared = clean_data.prepare_matches(raw, source)
        np.testing.assert_allclose(prepared.loc[0, clean_data.MARKET_COLUMNS].to_numpy(dtype=float), [2, 3, 4])
        self.assertTrue(prepared.loc[1:, clean_data.MARKET_COLUMNS].isna().all().all())
        self.assertTrue(pd.isna(prepared.loc[2, "FTResult"]))
        self.assertEqual(prepared.MatchDate.tolist(), source.SourceDate.tolist())
        pd.testing.assert_frame_equal(raw[clean_data.RAW_COLUMNS], prepared[clean_data.RAW_COLUMNS])

    def test_raw_only_preparation_needs_no_external_record(self):
        raw = self.games()
        prepared = clean_data.prepare_matches(raw)
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

    def test_raw_only_audit_ignores_stale_external_files(self):
        path = Path(__file__).resolve().parents[1] / "08_audit_data.py"
        spec = importlib.util.spec_from_file_location("audit_step", path)
        step = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw_path = folder / "Matches.csv"
            self.games().to_csv(raw_path, index=False)
            (folder / "data_preparation.json").write_text(json.dumps({"Mode": "Raw only"}))
            (folder / "data_cleaning_source_matches.csv").write_text("stale data; must not be read")
            for name in ["data_quality_external.csv", "data_quality_external_by_league.csv"]:
                (folder / name).write_text("outdated external results")
            with patch.object(step, "RAW_PATH", raw_path), patch.object(step, "REPORT_DIR", folder), \
                 patch.object(step, "build_report") as build, redirect_stdout(io.StringIO()):
                step.main()
            build.assert_called_once_with(report_dir=folder, include_sensitivity=False, allow_incomplete=True)
            self.assertTrue((folder / "data_quality_summary.csv").exists())
            self.assertFalse((folder / "data_quality_external.csv").exists())
            self.assertFalse((folder / "data_quality_external_by_league.csv").exists())


if __name__ == "__main__":
    unittest.main()
