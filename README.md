# Club football match predictability

Read **[Methodology and results](reports/study.md)** for the complete study, following the flow of `outline.pdf`: league comparisons, the supplied-odds benchmark, then models augmented with odds. This is the sole narrative report. The original PDF is an earlier draft; its corrected numbers and methodology are covered in the report.

Use Python 3.12 from this directory:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python 00_download_sources.py
.venv/bin/python 01_clean_data.py
.venv/bin/python 02_build_features.py
.venv/bin/python 03_train_models.py
.venv/bin/python 04_evaluate_leagues.py
.venv/bin/python 05_compare_market.py
.venv/bin/python 06_add_market_features.py
.venv/bin/python 07_train_market_models.py
.venv/bin/python 08_audit_data.py
.venv/bin/python 09_compare_without_external.py
```

`data/Matches.csv` is the master input. Step 00 downloads missing verification snapshots and reuses cached files; `--refresh` redownloads them. External data verifies dates/results and never replaces supplied odds or adds fixtures. Steps overwrite generated outputs, preserving the raw CSVs. Training and feature implementations live directly in `00`–`09`; later scripts import earlier ones without executing their `main()` functions.

Step 08 generates `reports/study.md` from the numerical exports. Step 09 refreshes it with the newly fitted cleaning-sensitivity comparison. To rebuild the document from completed outputs without refitting models:

```bash
.venv/bin/python tools/build_report.py
```

For a workflow using `Matches.csv` alone, skip 00, run `01_clean_data.py --raw-only`, then run 02–08. Step 09 requires source-verified preparation. Switching preparation mode requires rerunning downstream steps. Step 08 omits stale external evidence; the report builder also omits external sensitivity in raw-only mode.

After one complete source-verified run, capture and verify a repeat run with:

```bash
.venv/bin/python tools/run_study.py
.venv/bin/python tools/check_seasonal_precision.py
.venv/bin/python tools/verify_study.py
.venv/bin/python -W error -m unittest discover -s tests -v
```

`docs/` retains the CSV field dictionaries and ranking metadata used by the pipeline. `reports/` contains the unified report plus numerical CSV/JSON exports; `reports/reproduction/` retains hashes, logs and verification evidence. Models are saved under `models/`. `EloRatings.csv`, `E1_Matches.csv` and `F1_Matches.csv` are archival material, not runtime inputs.
