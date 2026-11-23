"""Measure the effect of external date/result cleaning while holding supplied odds and model settings fixed."""

from importlib import import_module
import hashlib
import json
import time

from threadpoolctl import threadpool_limits
import joblib
import numpy as np
import pandas as pd


build_report = import_module("tools.build_report").build_report

_step_00 = import_module('00_download_sources')
RAW_PATH = _step_00.RAW_PATH

_step_01 = import_module('01_clean_data')
LABELS = _step_01.LABELS
MARKET_COLUMNS = _step_01.MARKET_COLUMNS
REPORT_DIR = _step_01.REPORT_DIR
prepare_matches = _step_01.prepare_matches

_step_02 = import_module('02_build_features')
load_match_history = _step_02.load_match_history

_step_03 = import_module('03_train_models')
DATA_PATH = _step_03.DATA_PATH
FIELD_PATH = _step_03.FIELD_PATH
MODEL_DIR = _step_03.MODEL_DIR
TEST_END = _step_03.TEST_END
TEST_START = _step_03.TEST_START
TRAIN_START = _step_03.TRAIN_START
evaluate = _step_03.evaluate

_step_04 = import_module('04_evaluate_leagues')
LEAGUE_PATH = _step_04.LEAGUE_PATH
MODEL_NAMES = _step_04.MODEL_NAMES
TIERS = _step_04.TIERS
common_end_date = _step_04.common_end_date
prediction_scores = _step_04.prediction_scores

_step_05 = import_module('05_compare_market')
BENCHMARK = _step_05.BENCHMARK
add_odds_predictions = _step_05.add_odds_predictions

_step_06 = import_module('06_add_market_features')
MARKET_FIELD_PATH = _step_06.MARKET_FIELD_PATH
market_features = _step_06.market_features

_step_07 = import_module('07_train_market_models')
FEATURE_SETS = _step_07.FEATURE_SETS
fit_same_architecture = _step_07.fit_same_architecture

_step_08 = import_module('08_audit_data')
paired_test_mask = _step_08.paired_test_mask
valid_odds = _step_08.valid_odds
OUTPUT_MODELS = MODEL_DIR / "raw_only"


def summarize(table, scenario, feature_set, window):
    rows = []
    groups = [("Combined", "All", table)]
    groups += [("Tier", tier, part) for tier, part in table.groupby("Tier")]
    groups += [("League", code, part) for code, part in table.groupby("Division")]
    for scope, group, subset in groups:
        if subset.empty:
            continue
        for name in [BENCHMARK, *MODEL_NAMES]:
            slug = name.lower().replace(" ", "_")
            scores = prediction_scores(subset.FTResult, subset[f"{slug}_prediction"],
                                       subset[[f"{slug}_probability_{label}" for label in LABELS]])
            rows.append({"Scenario": scenario, "FeatureSet": feature_set, "Window": window,
                         "Scope": scope, "Group": group, "Model": name, "Matches": len(subset), **scores})
    return rows


def history_changes(raw, raw_features, base_fields, audit):
    # Source-date sorting preserves raw row IDs, so compare the same games even when dates move.
    corrected = raw.assign(MatchDate=audit.SourceDate)
    order = corrected.sort_values(["MatchDate", "MatchTime"], kind="stable", na_position="first").index
    raw_lookup = raw_features.set_index("RawRow")
    counts = np.zeros(len(base_fields), dtype=int)
    changed_rows = np.zeros(len(raw), dtype=bool)
    offset = 0
    for chunk in pd.read_csv(DATA_PATH, usecols=base_fields, chunksize=20000,
                             dtype={field: "float32" for field in base_fields}):
        row_ids = order[offset:offset + len(chunk)]
        comparison = raw_lookup.loc[row_ids, base_fields]
        different = ~np.isclose(comparison.to_numpy(dtype=float), chunk[base_fields].to_numpy(dtype=float),
                                rtol=1e-6, atol=1e-5, equal_nan=True)
        counts += different.sum(axis=0)
        changed_rows[row_ids] = different.any(axis=1)
        offset += len(chunk)
    if offset != len(raw):
        raise ValueError("Prepared feature table has a different number of matches; rerun steps 01–02.")
    pd.DataFrame({"Field": base_fields, "ChangedMatches": counts}).to_csv(
        REPORT_DIR / "external_data_feature_changes.csv", index=False)
    return changed_rows


def main():
    preparation = json.loads((REPORT_DIR / "data_preparation.json").read_text())
    if not preparation["Mode"].startswith("Source-verified"):
        raise ValueError("Step 09 requires source-verified main outputs; run steps 00–07 in that mode first.")
    if preparation["RawSHA256"] != hashlib.sha256(RAW_PATH.read_bytes()).hexdigest():
        raise ValueError("Matches.csv changed after the source-verified run; rebuild the main outputs first.")
    raw = pd.read_csv(RAW_PATH, parse_dates=["MatchDate"], dtype={"MatchTime": "str"})
    audit = pd.read_csv(REPORT_DIR / "data_cleaning_source_matches.csv", parse_dates=["MatchDate", "SourceDate"])
    if len(raw) != len(audit):
        raise ValueError("Source audit must contain exactly one record for every raw row.")
    leagues = pd.read_csv(LEAGUE_PATH)
    leagues = leagues.loc[leagues.Tier.isin(TIERS)]
    base_fields = pd.read_csv(FIELD_PATH).query("role == 'feature'")["field"].tolist()
    all_fields = pd.read_csv(MARKET_FIELD_PATH).query("role == 'feature'")["field"].tolist()
    features = {"No odds": base_fields, "With market odds": all_fields}
    # Build all raw-only features anew; no downloaded values enter this branch.
    table = load_match_history(RAW_PATH).astype({field: "float32" for field in base_fields}).copy()
    original_order = raw.sort_values(["MatchDate", "MatchTime"], kind="stable", na_position="first").index
    table["RawRow"] = original_order.to_numpy()
    changed = history_changes(raw, table, base_fields, audit)
    additions = market_features(prepare_matches(raw)).astype("float32")
    table = pd.concat([table, additions], axis=1)
    metadata = raw.loc[original_order, ["OddHome", "OddDraw", "OddAway"]].reset_index(drop=True)
    table[MARKET_COLUMNS] = metadata.to_numpy()
    known = table.FTResult.isin(LABELS)
    train = table.loc[known & table.MatchDate.ge(TRAIN_START) & table.MatchDate.lt(TEST_START)]
    test = table.loc[known & table.Division.isin(leagues.Division)
                     & table.MatchDate.ge(TEST_START) & table.MatchDate.lt(TEST_END)].copy()
    raw_end = common_end_date(test, leagues.Division)
    test = test.merge(leagues[["Division", "Tier"]], on="Division", validate="many_to_one", sort=False)
    # Prices are identical inputs to both branches. Paired scoring requires complete triplets.
    test = test.loc[valid_odds(test, MARKET_COLUMNS)].copy()
    raw_predictions = test[["RawRow", "Division", "Tier", "MatchDate", "FTResult", *MARKET_COLUMNS]].copy()
    raw_predictions = add_odds_predictions(raw_predictions, MARKET_COLUMNS, BENCHMARK)
    raw_predictions["RawDate"] = raw_predictions.MatchDate
    raw_predictions["SourceDate"] = audit.SourceDate.iloc[raw_predictions.RawRow].to_numpy()
    raw_predictions["SourceResult"] = audit.SourceResult.iloc[raw_predictions.RawRow].to_numpy()
    before = pd.read_csv(REPORT_DIR / "league_tier_predictions.csv", usecols=["MatchDate", "CommonWindow"], parse_dates=["MatchDate"])
    source_end = before.loc[before.CommonWindow, "MatchDate"].max()
    summary = {"RawInputSHA256": hashlib.sha256(RAW_PATH.read_bytes()).hexdigest(),
               "Protocol": "Fixed selected settings; raw-only refitting without retuning; supplied odds in both branches",
               "RawTrainingMatches": len(train), "RawTestMatchesWithOdds": len(test),
               "RawCommonEnd": str(raw_end.date()), "SourceCommonEnd": str(source_end.date()),
               "MatchesWithChangedBaseFeatures": int(changed.sum()),
               "MatchesWithChangedBaseFeaturesPercent": 100 * changed.mean()}
    (REPORT_DIR / "external_data_impact_settings.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)
    prediction_sets, metrics, timings = [], [], []
    with threadpool_limits(limits=2):
        for feature_set in FEATURE_SETS:
            selected = features[feature_set]
            variant = raw_predictions.copy()
            variant["FeatureSet"] = feature_set
            folder = OUTPUT_MODELS / feature_set.lower().replace(" ", "_")
            folder.mkdir(exist_ok=True)
            for name in MODEL_NAMES:
                started = time.perf_counter()
                slug = name.lower().replace(" ", "_")
                original = joblib.load(MODEL_DIR / f"{slug}.joblib")
                print(f"Raw-only fit: {feature_set}; {name}", flush=True)
                model = fit_same_architecture(original, train[selected], train.FTResult)
                predicted, probabilities, _ = evaluate(model, test[selected], test.FTResult)
                variant[f"{slug}_prediction"] = predicted
                for position, label in enumerate(LABELS):
                    variant[f"{slug}_probability_{label}"] = probabilities[:, position]
                joblib.dump(model, folder / f"{slug}.joblib", compress=3)
                timings.append({"FeatureSet": feature_set, "Model": name, "Seconds": time.perf_counter()-started})
            prediction_sets.append(variant)
            metrics.extend(summarize(variant, "Raw only", feature_set, "Native full"))
            metrics.extend(summarize(variant.loc[variant.RawDate.le(raw_end)], "Raw only", feature_set, "Native common"))
    raw_predictions = pd.concat(prediction_sets, ignore_index=True)
    raw_predictions.to_csv(REPORT_DIR / "raw_only_predictions.csv", index=False)
    source = pd.read_csv(REPORT_DIR / "market_augmented_predictions.csv", parse_dates=["MatchDate"])
    # This join follows source-verified game identities to the original row, never dates alone.
    mapping = audit[["Division", "HomeTeam", "AwayTeam", "SourceDate"]].copy()
    mapping["RawRow"] = np.arange(len(mapping))
    mapping = mapping.rename(columns={"SourceDate": "MatchDate"})
    source = source.merge(mapping, on=["Division", "HomeTeam", "AwayTeam", "MatchDate"], validate="many_to_one")
    paired = raw_predictions.merge(source, on=["RawRow", "FeatureSet"], suffixes=("", "_Verified"), validate="one_to_one")
    for feature_set in FEATURE_SETS:
        source_variant = source.loc[source.FeatureSet.eq(feature_set)]
        metrics.extend(summarize(source_variant, "External date/result cleaning", feature_set, "Native full"))
        metrics.extend(summarize(source_variant.loc[source_variant.CommonWindow], "External date/result cleaning", feature_set, "Native common"))
        variant = paired.loc[paired.FeatureSet.eq(feature_set)]
        for window, end in [("Paired common", source_end), ("Paired full", TEST_END-pd.Timedelta(days=1))]:
            matched = variant.loc[paired_test_mask(variant, end)].copy()
            metrics.extend(summarize(matched, "Raw only", feature_set, window))
            external = matched.copy()
            for name in [BENCHMARK, *MODEL_NAMES]:
                slug = name.lower().replace(" ", "_")
                for column in [f"{slug}_prediction", *[f"{slug}_probability_{label}" for label in LABELS]]:
                    external[column] = matched[f"{column}_Verified"]
            metrics.extend(summarize(external, "External date/result cleaning", feature_set, window))
    scores = pd.DataFrame(metrics)
    scores.to_csv(REPORT_DIR / "external_data_impact_metrics.csv", index=False)
    keys = ["FeatureSet", "Window", "Scope", "Group", "Model"]
    left = scores.loc[scores.Scenario.eq("Raw only")]
    right = scores.loc[scores.Scenario.eq("External date/result cleaning")]
    changes = left.merge(right, on=keys, suffixes=("_Raw", "_Cleaned"), validate="one_to_one")
    changes["AccuracyChangePP"] = 100 * (changes.Accuracy_Cleaned - changes.Accuracy_Raw)
    changes.to_csv(REPORT_DIR / "external_data_impact_changes.csv", index=False)
    primary = changes.loc[changes.Window.eq("Paired common") & changes.Scope.isin(["Tier", "Combined"]),
                          ["FeatureSet", "Group", "Model", "Matches_Raw", "Accuracy_Raw", "Accuracy_Cleaned", "AccuracyChangePP"]]
    pd.DataFrame(timings).to_csv(REPORT_DIR / "raw_only_training.csv", index=False)
    path = build_report(report_dir=REPORT_DIR, include_sensitivity=True)
    print(primary.to_string(index=False))
    print(f"Unified study report saved to {path}.")


if __name__ == "__main__":
    main()
