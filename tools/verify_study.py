"""Independently check rerun exports and compare the 63 accuracy cells in outline.pdf.

Run from any directory with the project's Python after the numbered steps finish.
Uses NumPy arithmetic, not the project's scoring/reporting functions. The extracted
PDF text and original-output backup are in reports/reproduction. Comparisons
normalize the documented report renames and retired source-audit fields.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "reproduction"
REPORTS = ROOT / "reports"
WINDOW = json.loads((REPORTS / "evaluation_window.json").read_text())
execution = json.loads((OUT / "execution.json").read_text())
BACKUP = Path(execution["BackupDirectory"])
LABELS = np.array(["A", "D", "H"])
MODELS = ["Gaussian Naive Bayes", "Logistic regression", "Histogram gradient boosting"]
checks = []


def check(name, passed, detail=""):
    checks.append({"Check": name, "Passed": bool(passed), "Detail": str(detail)})


def read(name):
    frame = pd.read_csv(REPORTS / name)
    for field in ["MatchDate", "RawDate", "SourceDate"]:
        if field in frame:
            frame[field] = pd.to_datetime(frame[field])
    return frame


def scores(frame, name):
    slug = name.lower().replace(" ", "_")
    y = frame.FTResult.to_numpy()
    predicted = frame[f"{slug}_prediction"].to_numpy()
    p = frame[[f"{slug}_probability_{label}" for label in LABELS]].to_numpy(float)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    np.testing.assert_allclose(p.sum(axis=1), 1, rtol=0, atol=1e-12)
    actual_count = np.array([(y == label).sum() for label in LABELS])
    predicted_count = np.array([(predicted == label).sum() for label in LABELS])
    true_positive = np.array([((y == label) & (predicted == label)).sum() for label in LABELS])
    recall = np.divide(true_positive, actual_count, out=np.zeros(3), where=actual_count > 0)
    denominator = actual_count + predicted_count
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros(3), where=denominator > 0)
    correct = predicted == y
    result = {"Matches": len(frame), "Accuracy": correct.mean(),
              "BalancedAccuracy": recall[actual_count > 0].mean(), "MacroF1": f1.mean(),
              "DrawRecall": recall[1],
              "LogLoss": -np.log(np.clip(p[np.arange(len(y)), np.searchsorted(LABELS, y)],
                                         np.finfo(float).eps, 1 - np.finfo(float).eps)).mean()}
    if "supplied_odds_prediction" in frame:
        market_correct = frame.supplied_odds_prediction.to_numpy() == y
        tied = frame.TiedFavourite.to_numpy(bool)
        result.update(GapToBenchmark=correct.mean() - market_correct.mean(),
                      ModelOnlyCorrect=int((correct & ~market_correct).sum()),
                      BenchmarkOnlyCorrect=int((~correct & market_correct).sum()),
                      TiedFavourites=int(tied.sum()), UniqueFavouriteMatches=int((~tied).sum()),
                      UniqueFavouriteAccuracy=correct[~tied].mean() if (~tied).any() else np.nan)
    return result


def subset(frame, row):
    selected = frame
    if "FeatureSet" in row:
        selected = selected.loc[selected.FeatureSet.eq(row.FeatureSet)]
    if row.get("Window") == "Common":
        selected = selected.loc[selected.CommonWindow]
    if row.get("Period", "All") != "All":
        selected = selected.loc[selected.MatchDate.dt.year.eq(int(row.Period))]
    if row.get("Season", "All") != "All":
        year = selected.MatchDate.dt.year - selected.MatchDate.dt.month.lt(7).astype(int)
        selected = selected.loc[year.eq(int(row.Season[:4]))]
    if "Scope" in row:
        if row.Scope == "Tier":
            selected = selected.loc[selected.Tier.eq(row.get("Group", row.get("Tier")))]
        elif row.Scope == "League":
            selected = selected.loc[selected.Division.eq(row.get("Group", row.get("Division")))]
    elif "Division" in row:
        selected = selected.loc[selected.Division.eq(row.Division)]
    return selected


def verify_metrics(filename, predictions=None, external=None):
    metrics = read(filename)
    max_error = 0.0
    cells = 0
    for _, row in metrics.iterrows():
        frame = external[(row.Scenario, row.FeatureSet, row.Window)] if external is not None else predictions
        frame = subset(frame, row)
        calculated = scores(frame, row.Model)
        if "MeanLeagueAccuracy" in row and pd.notna(row.MeanLeagueAccuracy):
            calculated["MeanLeagueAccuracy"] = np.mean([
                scores(group, row.Model)["Accuracy"] for _, group in frame.groupby("Division")])
        for field, expected in calculated.items():
            if field not in row:
                continue
            actual = row[field]
            if pd.isna(expected) and pd.isna(actual):
                continue
            error = abs(float(actual) - float(expected))
            # Step 03 predicts each season in a separate float32 batch. Its
            # logistic probabilities can differ slightly from the full-batch
            # probabilities exported in model_predictions.csv. A direct
            # seasonal-batch replay confirms the reported loss exactly.
            tolerance = (1e-7 if filename == "model_metrics.csv" and field == "LogLoss"
                         and row.get("Season") != "All" else 1e-10)
            assert error <= tolerance, (filename, row.to_dict(), field, expected)
            max_error = max(max_error, error)
            cells += 1
    check(f"Independent scores: {filename}", True,
          f"{len(metrics)} rows; {cells} checked cells; maximum absolute error {max_error:.3g}")


def markdown_tables(path):
    tables = {}
    identifier, lines = None, []
    for line in path.read_text().splitlines() + ["<!-- table: END -->"]:
        if line.startswith("<!-- table: "):
            if lines:
                columns = [value.strip() for value in lines[0].strip("|").split("|")]
                rows = [[value.strip() for value in row.strip("|").split("|")] for row in lines[2:]]
                tables[identifier] = pd.DataFrame(rows, columns=columns)
            identifier = line.removeprefix("<!-- table: ").removesuffix(" -->")
            lines = []
        elif line.startswith("| "):
            lines.append(line)
    return tables


def verify_class_tables():
    classes = read("model_class_metrics.csv")
    matrices = read("model_confusion_matrices.csv")
    frame = read("model_predictions.csv")
    for name in ["Prior baseline", *MODELS]:
        slug = name.lower().replace(" ", "_")
        y, p = frame.FTResult.to_numpy(), frame[f"{slug}_prediction"].to_numpy()
        matrix = np.array([[((y == actual) & (p == predicted)).sum() for predicted in LABELS] for actual in LABELS])
        observed = matrices.loc[matrices.Model.eq(name)].set_index("Actual").loc[LABELS, list(LABELS)].to_numpy(int)
        check(f"Independent confusion-matrix export: {name}", np.array_equal(matrix, observed))
        actual, predicted, tp = matrix.sum(axis=1), matrix.sum(axis=0), np.diag(matrix)
        precision = np.divide(tp, predicted, out=np.zeros(3), where=predicted > 0)
        recall = tp / actual
        f1 = np.divide(2 * tp, predicted + actual, out=np.zeros(3), where=(predicted + actual) > 0)
        expected = np.column_stack([precision, recall, f1, actual])
        observed = classes.loc[classes.Model.eq(name)].set_index("Result").loc[LABELS, ["precision", "recall", "f1-score", "support"]].to_numpy(float)
        check(f"Independent class-score export: {name}", np.allclose(expected, observed, rtol=0, atol=1e-10))


def verify_unified_report():
    tables = markdown_tables(REPORTS / "study.md")
    names = {"Prior baseline": "Prior baseline", "Naive Bayes": "Gaussian Naive Bayes",
             "Logistic regression": "Logistic regression", "Boosting": "Histogram gradient boosting",
             "Odds favourite": "Supplied odds"}
    top = read("top_five_model_metrics.csv").query("Season == 'All'")
    tier = read("league_tier_metrics.csv").query("Window == 'Available' and Period == 'All' and Scope == 'Tier'")
    market = read("supplied_odds_metrics.csv").query("Window == 'Available' and Period == 'All' and Scope == 'Tier'")
    changes = read("market_augmented_accuracy_changes.csv").query("Window == 'Available' and Period == 'All'")
    cells = 0
    def number(value, expected, percent=False, tolerance=0.00005001):
        nonlocal cells
        observed = float(value.rstrip('%')) / (100 if percent else 1)
        assert abs(observed - expected) <= tolerance, (value, expected)
        cells += 1
    for key, frame in [("top_five", top), ("tier_base", tier), ("tier_market", market)]:
        table = tables[key]
        assert len(table) == (5 if key == "top_five" else 3)
        for _, row in table.iterrows():
            group = re.search(r"\(([^)]+)\)$", row.League).group(1) if key == "top_five" else row.Tier
            selected = frame.loc[frame.Division.eq(group)] if key == "top_five" else frame.loc[frame.Tier.eq(group) if key == "tier_base" else frame.Group.eq(group)]
            number(row.Matches, int(selected.Matches.iloc[0]), tolerance=0)
            for column in table.columns[2:]:
                expected = float(selected.loc[selected.Model.eq(names[column]), "Accuracy"].iloc[0])
                number(row[column], expected, percent=True)
    coverage = read("league_tier_coverage.csv").query("Window == 'Available'")
    table = tables["date_coverage"]
    assert len(table) == len(coverage) == 15
    for _, row in table.iterrows():
        expected = coverage.loc[coverage.Division.eq(row.Division)].iloc[0]
        number(row.Matches, int(expected.Matches), tolerance=0)
        assert row.Tier == expected.Tier
        assert row['First available match'] == expected.FirstMatch
        assert row['Last available match'] == expected.LastMatch
    table = tables["odds_augmentation"]
    assert len(table) == 9
    for _, row in table.iterrows():
        selected = changes.loc[changes.Scope.eq("Tier") & changes.Group.eq(row.Tier) & changes.Model.eq(names[row.Model])].iloc[0]
        number(row.Matches, int(selected.Matches_NoOdds), tolerance=0)
        number(row["Without odds"], selected.Accuracy_NoOdds, percent=True)
        number(row["With odds"], selected.Accuracy_WithOdds, percent=True)
        number(row["Change (pp)"], selected.AccuracyChangePP, tolerance=0.00500001)
        expected = float(market.loc[market.Group.eq(row.Tier) & market.Model.eq("Supplied odds"), "Accuracy"].iloc[0])
        number(row["Odds favourite"], expected, percent=True)
    table = tables["pooled_augmentation"]
    assert len(table) == 4
    for _, row in table.iterrows():
        selected = changes.loc[changes.Scope.eq("Combined") & changes.Model.eq(names[row.Model])].iloc[0]
        number(row.Matches, int(selected.Matches_NoOdds), tolerance=0)
        number(row["Without odds"], selected.Accuracy_NoOdds, percent=True)
        number(row["With odds"], selected.Accuracy_WithOdds, percent=True)
        number(row["Change (pp)"], selected.AccuracyChangePP, tolerance=0.00500001)
        number(row["Log loss without"], selected.LogLoss_NoOdds)
        number(row["Log loss with"], selected.LogLoss_WithOdds)
    impact = read("external_data_impact_changes.csv").query("Window == 'Paired full' and Scope == 'Combined'")
    table = tables["cleaning_sensitivity"]
    assert len(table) == 3
    for _, row in table.iterrows():
        selected = impact.loc[impact.Model.eq(names[row.Model])]
        number(row["Paired matches"], int(selected.Matches_Raw.iloc[0]), tolerance=0)
        for column, feature in [("Without odds (pp)", "No odds"), ("With odds (pp)", "With market odds")]:
            number(row[column], float(selected.loc[selected.FeatureSet.eq(feature), "AccuracyChangePP"].iloc[0]), tolerance=0.00500001)
    check("Unified study document agrees with numerical exports", True, f"7 results/coverage tables; {cells} displayed values independently checked at their printed precision")


def verify_raw_audit(original, prepared, audit):
    n = len(original)
    detail = [f"{side}{stat}" for side in ["Home", "Away"]
              for stat in ["Shots", "Target", "Fouls", "Corners", "Yellow", "Red"]]
    odds = original[["OddHome", "OddDraw", "OddAway"]]
    valid = np.isfinite(odds).all(axis=1) & odds.gt(1).all(axis=1)
    identity = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]
    result = np.where(original.FTHome > original.FTAway, "H", np.where(original.FTHome < original.FTAway, "A", "D"))
    masks = [~original.FTResult.isin(LABELS), original[["FTHome", "FTAway"]].isna().any(axis=1),
             original[["HomeElo", "AwayElo"]].isna().any(axis=1), original[["HomeElo", "AwayElo"]].isna().all(axis=1),
             original[["HTHome", "HTAway"]].isna().any(axis=1), original[detail].isna().any(axis=1),
             original[detail].isna().all(axis=1), odds.isna().any(axis=1), odds.notna().all(axis=1) & ~valid,
             original.MatchTime.isna(), pd.to_datetime(original.MatchDate, errors="coerce").isna(),
             original.duplicated(), original.duplicated(identity), original.HomeTeam.eq(original.AwayTeam),
             original.FTResult.isin(LABELS) & original[["FTHome", "FTAway"]].notna().all(axis=1) & original.FTResult.ne(result),
             original.HTHome.gt(original.FTHome) | original.HTAway.gt(original.FTAway)]
    exported = read("data_quality_summary.csv")
    counts = np.array([m.sum() for m in masks])
    check("All 16 raw data-quality counts and percentages independently recomputed",
          np.array_equal(counts, exported.Matches) and np.allclose(100 * counts / n, exported.Percent, rtol=0, atol=1e-10))
    fields = read("data_quality_fields.csv")
    check("All 48 field missingness counts independently recomputed",
          np.array_equal(original.isna().sum().reindex(fields.Field).to_numpy(), fields.Missing))
    raw_date = pd.to_datetime(original.MatchDate)
    source_date = audit.SourceDate
    raw_train = raw_date.between("2000-07-01", "2023-06-30")
    source_train = source_date.between("2000-07-01", "2023-06-30")
    source_odds = audit[["MarketHome", "MarketDraw", "MarketAway"]]
    source_valid = np.isfinite(source_odds).all(axis=1) & source_odds.gt(1).all(axis=1)
    masks = [raw_date.ne(source_date), raw_train.ne(source_train), raw_train & ~source_train,
             source_train & ~raw_train, raw_date.between(WINDOW["StartDate"], WINDOW["AsOf"]).ne(source_date.between(WINDOW["StartDate"], WINDOW["AsOf"])),
             original.FTResult.isin(LABELS) & audit.SourceResult.isin(LABELS) & original.FTResult.ne(audit.SourceResult),
             ~valid & source_valid, valid & ~source_valid, ~source_valid]
    exported = read("data_quality_external.csv")
    counts = np.array([m.sum() for m in masks])
    check("All nine external-audit counts and percentages independently recomputed",
          np.array_equal(counts, exported.Matches) and np.allclose(100 * counts / n, exported.Percent, rtol=0, atol=1e-10))
    check("Result exclusions independently reconstructed",
          np.array_equal(prepared.FTResult.isna(), ~original.FTResult.isin(LABELS) | original.FTResult.ne(audit.SourceResult)))


def verify_evaluation_window():
    """Independently ensure all eligible dates enter every primary league cohort."""
    leagues = pd.read_csv(ROOT / "docs/league_strength_2023.csv")
    divisions = leagues.loc[leagues.Tier.isin(["Top", "Mid", "Lower"]), "Division"]
    keys = ["Division", "MatchDate", "HomeTeam", "AwayTeam"]
    table = pd.read_csv(ROOT / "data/MatchesChronological.csv", parse_dates=["MatchDate"],
                        usecols=[*keys, "FTResult", "MarketHome", "MarketDraw", "MarketAway"])
    expected = table.loc[table.Division.isin(divisions) & table.FTResult.isin(LABELS)
                         & table.MatchDate.between(WINDOW["StartDate"], WINDOW["AsOf"])]
    predictions = read("league_tier_predictions.csv")
    expected_keys = pd.MultiIndex.from_frame(expected[keys]).sort_values()
    actual_keys = pd.MultiIndex.from_frame(predictions[keys]).sort_values()
    check("All available league fixtures use the recorded July 2023–present window",
          expected_keys.equals(actual_keys), f"{len(expected)} fixtures; {WINDOW['StartDate']}–{WINDOW['AsOf']}")
    top = read("top_five_model_predictions.csv")
    tier_top = predictions.loc[predictions.Tier.eq("Top")]
    check("Big Five and Top tier use identical match identities",
          pd.MultiIndex.from_frame(top[keys]).sort_values().equals(
              pd.MultiIndex.from_frame(tier_top[keys]).sort_values()), len(top))
    premier = read("model_predictions.csv")
    premier_keys = keys[1:]
    check("Premier League and tier evaluation use identical match identities",
          pd.MultiIndex.from_frame(premier[premier_keys]).sort_values().equals(
              pd.MultiIndex.from_frame(tier_top.loc[tier_top.Division.eq("E0"), premier_keys]).sort_values()))
    valid = expected[["MarketHome", "MarketDraw", "MarketAway"]].gt(1).all(axis=1)
    valid &= np.isfinite(expected[["MarketHome", "MarketDraw", "MarketAway"]]).all(axis=1)
    market_keys = pd.MultiIndex.from_frame(expected.loc[valid, keys]).sort_values()
    for name in ["supplied_odds_predictions.csv", "market_augmented_predictions.csv"]:
        frame = read(name)
        if "FeatureSet" in frame:
            cohorts = frame.groupby("FeatureSet")
        else:
            cohorts = [("Base", frame)]
        for feature, group in cohorts:
            check(f"Full-window valid-odds cohort: {name} / {feature}",
                  market_keys.equals(pd.MultiIndex.from_frame(group[keys]).sort_values()), len(group))
    check("Available is the primary report window", WINDOW["PrimaryWindow"] == "Available")


def main(compare_previous=False):
    assert execution.get("FinishedAtUTC"), "The numbered rerun has not completed."
    check("All numbered steps and existing tests completed", all(s["ExitCode"] == 0 for s in execution["Steps"]))
    unchanged = []
    for name, original in execution["InputAndInitialHashes"].items():
        if name in ["data/MatchesChronological.csv", "data/MatchTraining.csv", "data/MarketTraining.csv",
                    "data/football_data_sources/download_manifest.csv"]:
            continue
        # The manifest retains the original code/documentation hashes as run
        # provenance; cleanup may change them without changing study inputs.
        if name.endswith(".py") or name == "README.md":
            continue
        unchanged.append(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == original)
    check("Raw input, source snapshots, outline and requirements unchanged", all(unchanged), len(unchanged))
    for name in ["MatchesChronological.csv", "MatchTraining.csv", "MarketTraining.csv"]:
        digest = hashlib.sha256((ROOT / "data" / name).read_bytes()).hexdigest()
        check(f"Regenerated data byte identity: {name}", digest == execution["InputAndInitialHashes"][f"data/{name}"], digest)
    comparisons = []
    for path in sorted((BACKUP / "reports").glob("*.csv")):
        current_path = REPORTS / path.name.replace("market_average_", "supplied_odds_")
        before = pd.read_csv(path)
        after = pd.read_csv(current_path)
        ignored = []
        old_label = "Missing raw odds but usable external averages"
        new_label = "Missing/invalid raw odds but usable external averages"
        before = before.rename(columns={old_label: new_label})
        if "Check" in before:
            before["Check"] = before["Check"].replace({old_label: new_label})
        if path.name == "data_cleaning_source_matches.csv":
            before = before.rename(columns={"EvaluationRow": "RawRow"})
            retired = [c for c in ["RawResult", "Status", "Eligible"] if c in before and c not in after]
            before = before.drop(columns=retired)
            ignored.extend(retired)
        same = before.shape == after.shape and before.columns.tolist() == after.columns.tolist()
        max_error = 0.0
        for column in before.columns if same else []:
            if column == "Seconds":
                ignored.append(column)
            elif pd.api.types.is_numeric_dtype(before[column]) and pd.api.types.is_numeric_dtype(after[column]):
                a, b = before[column].to_numpy(float), after[column].to_numpy(float)
                same &= bool(np.isclose(a, b, rtol=0, atol=1e-10, equal_nan=True).all())
                finite = np.isfinite(a) & np.isfinite(b)
                if finite.any():
                    max_error = max(max_error, float(np.max(np.abs(a[finite] - b[finite]))))
            else:
                same &= before[column].fillna("<NA>").equals(after[column].fillna("<NA>"))
        comparisons.append({"File": current_path.name, "PreviousFile": path.name, "Rows": len(after), "MatchesPrevious": same,
                            "MaxNumericDifference": max_error, "Ignored": ",".join(ignored),
                            "PreviousSHA256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "RerunSHA256": hashlib.sha256(current_path.read_bytes()).hexdigest()})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "previous_output_comparison.csv", index=False)
    if compare_previous:
        check("All CSV results preserved (elapsed seconds excluded)", comparison.MatchesPrevious.all(), len(comparison))
    model_comparisons = []
    for path in sorted((BACKUP / "models").rglob("*.joblib")):
        relative = path.relative_to(BACKUP)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        after = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        model_comparisons.append({"File": str(relative), "PreviousSHA256": before, "RerunSHA256": after,
                                  "ByteIdentical": before == after})
    pd.DataFrame(model_comparisons).to_csv(OUT / "model_artifact_comparison.csv", index=False)
    verify_unified_report()
    verify_evaluation_window()
    for metric, prediction in [
        ("model_metrics.csv", "model_predictions.csv"),
        ("top_five_model_metrics.csv", "top_five_model_predictions.csv"),
        ("league_tier_metrics.csv", "league_tier_predictions.csv"),
        ("supplied_odds_metrics.csv", "supplied_odds_predictions.csv"),
        ("market_augmented_metrics.csv", "market_augmented_predictions.csv"),
    ]:
        verify_metrics(metric, read(prediction))
    verify_class_tables()
    raw = read("raw_only_predictions.csv")
    source = read("market_augmented_predictions.csv")
    audit = read("data_cleaning_source_matches.csv")
    mapping = audit[["Division", "HomeTeam", "AwayTeam", "SourceDate"]].rename(columns={"SourceDate": "MatchDate"})
    mapping["RawRow"] = np.arange(len(mapping))
    source = source.merge(mapping, on=["Division", "HomeTeam", "AwayTeam", "MatchDate"], validate="many_to_one")
    settings = json.loads((REPORTS / "external_data_impact_settings.json").read_text())
    external = {}
    for feature in raw.FeatureSet.unique():
        raw_variant = raw.loc[raw.FeatureSet.eq(feature)]
        source_variant = source.loc[source.FeatureSet.eq(feature)]
        for scenario, frame, end in [("Raw only", raw_variant, settings["RawCommonEnd"]),
                                     ("External date/result cleaning", source_variant, settings["SourceCommonEnd"])]:
            external[(scenario, feature, "Native full")] = frame
            external[(scenario, feature, "Native common")] = frame.loc[frame.MatchDate.le(end)]
        common = raw_variant.merge(source_variant[["RawRow"]], on="RawRow", validate="one_to_one")
        for label, end in [("Paired common", settings["SourceCommonEnd"]), ("Paired full", WINDOW["AsOf"])]:
            matched = common.loc[common.RawDate.between("2023-07-01", end)
                                 & common.SourceDate.between("2023-07-01", end)
                                 & common.FTResult.eq(common.SourceResult)]
            external[("Raw only", feature, label)] = matched
            external[("External date/result cleaning", feature, label)] = source_variant.loc[source_variant.RawRow.isin(matched.RawRow)]
    verify_metrics("external_data_impact_metrics.csv", external=external)
    for filename, left, right, count_left, count_right in [
        ("market_augmented_accuracy_changes.csv", "Accuracy_NoOdds", "Accuracy_WithOdds", "Matches_NoOdds", "Matches_WithOdds"),
        ("external_data_impact_changes.csv", "Accuracy_Raw", "Accuracy_Cleaned", None, None),
    ]:
        changes = read(filename)
        check(f"Independent accuracy changes: {filename}",
              np.allclose(changes.AccuracyChangePP, 100 * (changes[right] - changes[left]), rtol=0, atol=1e-10))
        if count_left:
            check("Odds ablation uses equal cohort sizes in every exported group", changes[count_left].equals(changes[count_right]))
    control = source.loc[source.FeatureSet.eq("No odds")].sort_values("RawRow")
    odds_variant = source.loc[source.FeatureSet.eq("With market odds")].sort_values("RawRow")
    check("Odds ablation uses identical held-out match identities", np.array_equal(control.RawRow, odds_variant.RawRow))
    frozen = read("supplied_odds_predictions.csv")
    for name in ["Prior baseline", *MODELS]:
        slug = name.lower().replace(" ", "_")
        columns = [f"{slug}_prediction"] + [f"{slug}_probability_{label}" for label in LABELS]
        paired = control.merge(frozen, on=["Division", "MatchDate", "HomeTeam", "AwayTeam"], validate="one_to_one", suffixes=("", "_Frozen"))
        same = len(paired) == len(control) == len(frozen)
        same &= paired[columns[0]].equals(paired[columns[0] + "_Frozen"])
        same &= np.allclose(paired[columns[1:]].to_numpy(float), paired[[c + "_Frozen" for c in columns[1:]]].to_numpy(float), rtol=0, atol=1e-10)
        check(f"Refitted base control matches the step-03 frozen model: {name}", same)
    prepared = pd.read_csv(ROOT / "data/MatchesChronological.csv", dtype={"MatchTime": "str"})
    original = pd.read_csv(ROOT / "data/Matches.csv", dtype={"MatchTime": "str"})
    verify_raw_audit(original, prepared, audit)
    for column in ["OddHome", "OddDraw", "OddAway", "HomeElo", "AwayElo"]:
        check(f"Supplied field preserved: {column}", original[column].equals(prepared[column]))
    check("Corrected dates align with source audit", pd.to_datetime(prepared.MatchDate).equals(audit.SourceDate))
    check("Master row count preserved", len(original) == len(prepared) == 238858)
    check("Training rows independently counted", int((prepared.FTResult.isin(LABELS)
          & prepared.MatchDate.ge("2000-07-01") & prepared.MatchDate.lt("2023-07-01")).sum()) == 209501)
    current_odds = prepared[["MarketHome", "MarketDraw", "MarketAway"]].to_numpy(float)
    valid = np.isfinite(current_odds).all(axis=1) & (current_odds > 1).all(axis=1)
    features = pd.read_csv(ROOT / "data/MarketTraining.csv", usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam",
                               "MarketHomeProbability", "MarketDrawProbability", "MarketAwayProbability"])
    prepared["ValidOdds"] = valid
    aligned = features.merge(prepared[["Division", "MatchDate", "HomeTeam", "AwayTeam", "MarketHome", "MarketDraw", "MarketAway", "ValidOdds"]],
                             on=["Division", "MatchDate", "HomeTeam", "AwayTeam"], validate="one_to_one")
    observed = aligned[["MarketHomeProbability", "MarketDrawProbability", "MarketAwayProbability"]].to_numpy(float)
    implied = 1 / aligned.loc[aligned.ValidOdds, ["MarketHome", "MarketDraw", "MarketAway"]].to_numpy(float)
    implied /= implied.sum(axis=1, keepdims=True)
    check("Current probability features use inverse decimal odds", np.allclose(observed[aligned.ValidOdds], implied, rtol=0, atol=1e-12)
          and np.isnan(observed[~aligned.ValidOdds]).all())
    for filename in ["supplied_odds_predictions.csv", "market_augmented_predictions.csv", "raw_only_predictions.csv"]:
        frame = read(filename)
        odds = frame[["MarketHome", "MarketDraw", "MarketAway"]].to_numpy(float)
        favourites = np.array(["H", "D", "A"])[odds.argmin(axis=1)]
        inverse = 1 / odds
        probabilities = (inverse / inverse.sum(axis=1, keepdims=True))[:, [2, 1, 0]]
        observed = frame[[f"supplied_odds_probability_{label}" for label in LABELS]].to_numpy(float)
        check(f"Independent favourite and probability check: {filename}",
              (favourites == frame.supplied_odds_prediction).all()
              and np.allclose(probabilities, observed, rtol=0, atol=1e-12))
    outline_comparison()
    pd.DataFrame(checks).to_csv(OUT / "verification_checks.csv", index=False)
    result = {"Checks": len(checks), "Passed": sum(c["Passed"] for c in checks),
              "Failed": [c for c in checks if not c["Passed"]]}
    (OUT / "verification_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    for item in checks:
        print(item)
    assert not result["Failed"], result["Failed"]


def outline_comparison():
    text = (OUT / "outline_extracted.txt").read_text()
    percentages = iter(float(p) for p in re.findall(r"(\d+\.\d+)%", text))
    rows = []
    top = read("top_five_model_metrics.csv")
    tier = read("league_tier_metrics.csv")
    market = read("supplied_odds_metrics.csv")
    augmented = read("market_augmented_metrics.csv")

    def record(table, group, model, column, result):
        assert len(result) == 1, (table, group, model, column)
        reported = next(percentages)
        actual = 100 * float(result.Accuracy.iloc[0])
        rows.append({"OutlineTable": table, "Group": group, "Model": model, "Column": column,
                     "OutlinePercent": reported, "RerunPercent": actual, "DeltaPP": actual - reported,
                     "Matches": int(result.Matches.iloc[0]),
                     "MatchesOutlineRounding": abs(round(actual, 1) - reported) < 1e-8})

    for league in ["E0", "SP1", "D1", "I1", "F1"]:
        for model in MODELS:
            record("Top-five full window", league, model, "Accuracy", top.loc[
                top.Division.eq(league) & top.Model.eq(model) & top.Season.eq("All")])
    for group in ["Top", "Mid", "Lower"]:
        for model in MODELS:
            record("Tier base models", group, model, "Accuracy", tier.loc[
                tier.Tier.eq(group) & tier.Model.eq(model) & tier.Window.eq("Available")
                & tier.Period.eq("All") & tier.Scope.eq("Tier")])
    for group in ["Top", "Mid", "Lower"]:
        for model in MODELS + ["Supplied odds"]:
            record("Tier market comparison", group, model, "Accuracy", market.loc[
                market.Group.eq(group) & market.Model.eq(model) & market.Window.eq("Available")
                & market.Period.eq("All") & market.Scope.eq("Tier")])
    for group in ["Top", "Mid", "Lower"]:
        for model in MODELS:
            for column, choice, feature in [("Market", "Supplied odds", "No odds"),
                                            ("Old model", model, "No odds"),
                                            ("New model", model, "With market odds")]:
                record("Tier odds augmentation", group, model, column, augmented.loc[
                    augmented.Group.eq(group) & augmented.Model.eq(choice) & augmented.FeatureSet.eq(feature)
                    & augmented.Window.eq("Available") & augmented.Period.eq("All") & augmented.Scope.eq("Tier")])
    assert next(percentages, None) is None
    pd.DataFrame(rows).to_csv(OUT / "outline_number_comparison.csv", index=False)
    print(f"Outline: {sum(r['MatchesOutlineRounding'] for r in rows)} of {len(rows)} score cells match at one decimal percent.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-previous", action="store_true", help="Also require equality with historical outputs.")
    main(compare_previous=parser.parse_args().compare_previous)
