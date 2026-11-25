"""Build the single methodology/results document from the numbered study's CSV exports."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


JUSTIFICATIONS = {
    "Prior baseline": "Always predicts the most frequent training result; probabilities are training class frequencies. Establishes a minimum benchmark.",
    "Gaussian Naive Bayes": "A fast Bayesian classification baseline for numeric inputs. Its conditional-independence and Gaussian assumptions are imperfect for correlated match statistics; variance smoothing limits overconfidence.",
    "Logistic regression": "A regularized, interpretable linear multiclass model. Suitable for many correlated numerical inputs; scaling and regularization control the influence of their different units.",
    "Histogram gradient boosting": "Boosted trees capture nonlinear relationships and interactions between Elo, venue, and individual previous games. Histogram binning scales to this dataset and the model handles missing statistics directly.",
}


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["Gaussian Naive Bayes", "Logistic regression", "Histogram gradient boosting"]
ALL_MODELS = ["Prior baseline", *MODELS]
TIERS = ["Top", "Mid", "Lower"]
LABELS = ["A", "D", "H"]
NAMES = {"Prior baseline": "Prior baseline", "Gaussian Naive Bayes": "Naive Bayes",
         "Logistic regression": "Logistic regression", "Histogram gradient boosting": "Boosting"}


def markdown_table(identifier, columns, rows):
    """Give each table a stable identifier for independent numerical verification."""
    lines = [f"<!-- table: {identifier} -->", "| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(lines)


def export_class_diagnostics(predictions, report_dir):
    """Preserve the per-class scores and confusion matrices formerly printed in a separate report."""
    class_rows, matrices = [], []
    for name in ALL_MODELS:
        slug = name.lower().replace(" ", "_")
        predicted = predictions[f"{slug}_prediction"]
        report = classification_report(predictions.FTResult, predicted, labels=LABELS,
                                       output_dict=True, zero_division=0)
        class_rows.extend({"Model": name, "Result": label, **report[label]} for label in LABELS)
        matrix = pd.DataFrame(confusion_matrix(predictions.FTResult, predicted, labels=LABELS), columns=LABELS)
        matrix.insert(0, "Actual", LABELS)
        matrix.insert(0, "Model", name)
        matrices.append(matrix)
    classes = pd.DataFrame(class_rows)
    confusion = pd.concat(matrices, ignore_index=True)
    classes.to_csv(report_dir / "model_class_metrics.csv", index=False)
    confusion.to_csv(report_dir / "model_confusion_matrices.csv", index=False)
    return classes, confusion


def build_report(report_dir=None, include_sensitivity=True, allow_incomplete=False):
    """Publish study.md after step 08/09; never include a verified-date sensitivity in raw-only mode."""
    report_dir = Path(report_dir) if report_dir is not None else ROOT / "reports"
    required = ["data_preparation.json", "evaluation_window.json", "model_predictions.csv", "model_metrics.csv", "validation_metrics.csv",
                "top_five_model_metrics.csv", "league_tier_metrics.csv", "league_tier_coverage.csv",
                "league_tier_predictions.csv", "supplied_odds_metrics.csv", "market_augmented_metrics.csv",
                "market_augmented_training.csv", "market_augmented_accuracy_changes.csv", "data_quality_summary.csv"]
    missing = [name for name in required if not (report_dir / name).exists()]
    if missing:
        if allow_incomplete:
            return None
        raise FileNotFoundError("Complete steps 00–08 before building the study report. Missing: " + ", ".join(missing))
    read = lambda name: pd.read_csv(report_dir / name)
    preparation = json.loads((report_dir / "data_preparation.json").read_text())
    window = json.loads((report_dir / "evaluation_window.json").read_text())
    evaluation_start, evaluation_end = window["StartDate"], window["AsOf"]
    verified = preparation["Mode"].startswith("Source-verified")
    include_sensitivity = include_sensitivity and verified
    top = read("top_five_model_metrics.csv").query("Season == 'All'")
    tier = read("league_tier_metrics.csv").query("Window == 'Available' and Period == 'All' and Scope == 'Tier'")
    market = read("supplied_odds_metrics.csv").query("Window == 'Available' and Period == 'All'")
    augmented = read("market_augmented_metrics.csv").query("Window == 'Available' and Period == 'All'")
    changes = read("market_augmented_accuracy_changes.csv").query("Window == 'Available' and Period == 'All'")
    coverage = read("league_tier_coverage.csv").query("Window == 'Available'")
    selected = read("validation_metrics.csv").query("Selected")
    training = read("market_augmented_training.csv")
    quality = read("data_quality_summary.csv")
    rankings = pd.read_csv(ROOT / "docs/league_strength_2023.csv")
    predictions = read("model_predictions.csv")
    classes, confusion = export_class_diagnostics(predictions, report_dir)
    common_end = window["CommonEnd"]
    last_available = coverage.LastMatch.max()
    evaluation_n = int(tier.groupby("Tier").Matches.first().sum())
    date_coverage_rows = [[row.Tier, row.Division, int(row.Matches), row.FirstMatch, row.LastMatch]
                          for tier_name in TIERS for row in coverage.loc[coverage.Tier.eq(tier_name)].itertuples()]
    eligible_n = int(market.loc[market.Scope.eq("Combined") & market.Model.eq("Supplied odds"), "Matches"].iloc[0])
    training_n = int(training.TrainingMatches.iloc[0])
    prepared = pd.read_csv(ROOT / "data/MatchesChronological.csv", usecols=["Division", "MatchDate", "FTResult"], parse_dates=["MatchDate"])
    validation_n = int((prepared.Division.eq("E0") & prepared.MatchDate.ge("2021-07-01")
                        & prepared.MatchDate.lt("2023-07-01") & prepared.FTResult.isin(LABELS)).sum())

    def accuracy(frame, model):
        return float(frame.loc[frame.Model.eq(model), "Accuracy"].iloc[0])

    top_rows = []
    for code, name in [("E0", "Premier League"), ("SP1", "La Liga"), ("D1", "Bundesliga"),
                       ("I1", "Serie A"), ("F1", "Ligue 1")]:
        part = top.loc[top.Division.eq(code)]
        top_rows.append([f"{name} ({code})", int(part.Matches.iloc[0]),
                         *[f"{accuracy(part, model):.2%}" for model in ALL_MODELS]])
    top_table = markdown_table("top_five", ["League", "Matches", "Prior baseline", "Naive Bayes", "Logistic regression", "Boosting"], top_rows)
    tier_rows, market_rows, coverage_rows, group_rows = [], [], [], []
    for group in TIERS:
        part = tier.loc[tier.Tier.eq(group)]
        tier_rows.append([group, int(part.Matches.iloc[0]), *[f"{accuracy(part, model):.2%}" for model in ALL_MODELS]])
        part = market.loc[market.Scope.eq("Tier") & market.Group.eq(group)]
        market_rows.append([group, int(part.Matches.iloc[0]), *[f"{accuracy(part, model):.2%}" for model in [*MODELS, "Supplied odds"]]])
        part = coverage.loc[coverage.Tier.eq(group)]
        total = part.Matches.sum()
        coverage_rows.append([group, int(part.TrainingMatches.sum()),
                              f"{(part.CurrentEloMissing * part.Matches).sum() / total:.2%}",
                              f"{(part.DetailedStatsMissing * part.Matches).sum() / total:.2%}"])
        group_data = rankings.loc[rankings.Tier.eq(group)]
        group_rows.append([group, "; ".join(f"{row.Country} {row.Division}" for row in group_data.itertuples()),
                           ", ".join(group_data.UEFARank2023.astype(str))])
    base_table = markdown_table("tier_base", ["Tier", "Matches", "Prior baseline", "Naive Bayes", "Logistic regression", "Boosting"], tier_rows)
    market_table = markdown_table("tier_market", ["Tier", "Matches", "Naive Bayes", "Logistic regression", "Boosting", "Odds favourite"], market_rows)
    augmentation_rows, pooled_rows = [], []
    for group in TIERS:
        for name in MODELS:
            row = changes.loc[changes.Scope.eq("Tier") & changes.Group.eq(group) & changes.Model.eq(name)].iloc[0]
            benchmark = accuracy(market.loc[market.Scope.eq("Tier") & market.Group.eq(group)], "Supplied odds")
            augmentation_rows.append([group, NAMES[name], int(row.Matches_NoOdds), f"{row.Accuracy_NoOdds:.2%}",
                                      f"{benchmark:.2%}", f"{row.Accuracy_WithOdds:.2%}", f"{row.AccuracyChangePP:+.2f}"])
    for name in ALL_MODELS:
        row = changes.loc[changes.Scope.eq("Combined") & changes.Model.eq(name)].iloc[0]
        pooled_rows.append([NAMES[name], int(row.Matches_NoOdds), f"{row.Accuracy_NoOdds:.2%}",
                            f"{row.Accuracy_WithOdds:.2%}", f"{row.AccuracyChangePP:+.2f}",
                            f"{row.LogLoss_NoOdds:.4f}", f"{row.LogLoss_WithOdds:.4f}"])
    augmentation_table = markdown_table("odds_augmentation", ["Tier", "Model", "Matches", "Without odds", "Odds favourite", "With odds", "Change (pp)"], augmentation_rows)
    pooled_table = markdown_table("pooled_augmentation", ["Model", "Matches", "Without odds", "With odds", "Change (pp)", "Log loss without", "Log loss with"], pooled_rows)
    parameter_rows = []
    choices = JUSTIFICATIONS
    for name in ALL_MODELS:
        row = selected.loc[selected.Model.eq(name)].iloc[0]
        parameter_rows.append([NAMES[name], row.Parameters, f"{row.LogLoss:.4f}", choices[name]])
    quality_rows = [[row.Check, f"{int(row.Matches):,}", f"{row.Percent:.4f}%" if row.Percent < .01 else f"{row.Percent:.2f}%"] for row in quality.itertuples()
                    if row.Matches > 0]
    full_models = read("model_metrics.csv").query("Season == 'All'")
    diagnostic_rows = []
    for name in ALL_MODELS:
        row = full_models.loc[full_models.Model.eq(name)].iloc[0]
        slug = name.lower().replace(" ", "_")
        draws = predictions.FTResult.eq("D")
        draw_recall = predictions.loc[draws, f"{slug}_prediction"].eq("D").mean()
        diagnostic_rows.append([NAMES[name], int(row.Matches), f"{row.Accuracy:.2%}", f"{row.BalancedAccuracy:.2%}",
                                f"{row.MacroF1:.4f}", f"{row.LogLoss:.4f}", f"{draw_recall:.2%}"])
    market_pooled = market.loc[market.Scope.eq("Combined") & market.Model.eq("Supplied odds")].iloc[0]
    base_gaps = -100 * market.loc[market.Scope.eq("Tier") & market.Model.isin(MODELS), "GapToBenchmark"]
    # Quantitative interpretation is calculated from this run, including raw-only runs.
    winners = {NAMES[name]: top.loc[top.Model.eq(name)].sort_values("Accuracy", ascending=False).Division.iloc[0] for name in MODELS}
    winner_text = "; ".join(f"{name}: {code}" for name, code in winners.items())
    improving = int((changes.loc[changes.Scope.eq("Tier") & changes.Model.isin(MODELS), "AccuracyChangePP"] > 0).sum())
    lower = tier.loc[tier.Tier.eq("Lower")]
    lower_nb_gain = 100 * (accuracy(lower, "Gaussian Naive Bayes") - accuracy(lower, "Prior baseline"))
    best_gain = changes.loc[changes.Scope.eq("Tier") & changes.Model.isin(MODELS)].sort_values("AccuracyChangePP", ascending=False).iloc[0]
    advantages = []
    for group in TIERS:
        part = augmented.loc[augmented.Scope.eq("Tier") & augmented.Group.eq(group)
                             & augmented.FeatureSet.eq("With market odds") & augmented.Model.isin(MODELS)]
        best = part.sort_values("Accuracy", ascending=False).iloc[0]
        benchmark = accuracy(market.loc[market.Scope.eq("Tier") & market.Group.eq(group)], "Supplied odds")
        advantages.append(f"{group}: {NAMES[best.Model]} {best.Accuracy:.2%} versus favourite {benchmark:.2%}")
    pooled_best = changes.loc[changes.Scope.eq("Combined") & changes.Model.isin(MODELS)].sort_values("Accuracy_WithOdds", ascending=False).iloc[0]
    advantage_text = "; ".join(advantages)

    sensitivity = "Step 09 has not been included in this report. Run it after source-verified preparation to compare the two date treatments on identical held-out fixtures."
    if not verified:
        sensitivity = "This report uses raw-only preparation. No external date/result correction or external-cleaning sensitivity is included. Step 09 requires the source-verified main branch."
    elif include_sensitivity:
        required_sensitivity = ["external_data_impact_changes.csv", "external_data_impact_settings.json"]
        missing_sensitivity = [name for name in required_sensitivity if not (report_dir / name).exists()]
        if missing_sensitivity:
            raise FileNotFoundError("Run step 09 to include cleaning sensitivity: " + ", ".join(missing_sensitivity))
        settings = json.loads((report_dir / "external_data_impact_settings.json").read_text())
        if settings["RawInputSHA256"] != preparation["RawSHA256"]:
            raise ValueError("Cleaning-sensitivity inputs differ from the main study; rerun step 09.")
        if settings.get("EvaluationAsOf") != evaluation_end:
            raise ValueError("Cleaning sensitivity uses a different evaluation end date; rerun step 09.")
        impact = read("external_data_impact_changes.csv").query("Window == 'Paired full' and Scope == 'Combined'")
        sensitivity_rows = []
        for name in MODELS:
            part = impact.loc[impact.Model.eq(name)]
            sensitivity_rows.append([NAMES[name], int(part.Matches_Raw.iloc[0]),
                                     *[f"{part.loc[part.FeatureSet.eq(feature), 'AccuracyChangePP'].iloc[0]:+.2f}" for feature in ["No odds", "With market odds"]]])
        paired_n = int(impact.Matches_Raw.iloc[0])
        sensitivity = f"""Step 09 rebuilds histories and fits both feature variants from `Matches.csv` alone, with the selected settings fixed. Saved models can be reused with `--evaluate-only`. Both branches use the same supplied odds. The primary sensitivity comparison contains **{paired_n:,} identical fixtures** whose raw and verified dates both fall between {evaluation_start} and {evaluation_end}, with agreeing targets and valid supplied odds. This differs from the {eligible_n:,}-fixture primary odds comparison because pairing also requires agreement on eligibility under both date treatments. Secondary overlap-window scores are retained separately; their raw-date and verified-date cutoffs are {settings['RawCommonEnd']} and {settings['SourceCommonEnd']}, respectively.

{markdown_table('cleaning_sensitivity', ['Model', 'Paired matches', 'Without odds (pp)', 'With odds (pp)'], sensitivity_rows)}

Values are cleaned minus raw accuracy, in percentage points. Date/result verification changes at least one base input for **{settings['MatchesWithChangedBaseFeatures']:,} matches ({settings['MatchesWithChangedBaseFeaturesPercent']:.2f}%)**. Cleaning protects chronology; it does not improve every classifier's accuracy. Raw-only training contains {settings['RawTrainingMatches']:,} matches versus {training_n:,} in the main branch. Elo and missing statistics are preserved in both branches."""
    external_note = "Raw-only preparation preserves recorded dates and all valid supplied targets."
    external_path = report_dir / "data_quality_external.csv"
    if verified and external_path.exists():
        external = read("data_quality_external.csv").set_index("Check")
        crossings = int(external.loc["Date change moves match into/out of training", "Matches"])
        disagreements = int(external.loc["Known result disagrees with source", "Matches"])
        external_note = (f"Verification changes **{preparation['DatesChanged']:,} dates**, identifies **{disagreements} source-result disagreement**, "
                         f"and moves **{crossings:,} matches** across the July 2023 training boundary. "
                         f"Preparation excludes **{preparation['MissingOrInvalidTargets']} targets** in total, including originally missing results. "
                         "It does not fill Elo, statistics or supplied odds. External averages are retained as verification evidence only, including when they could recover a missing or invalid supplied quote.")
    class_rows = [[NAMES[row['Model']], row['Result'], f"{row['precision']:.4f}", f"{row['recall']:.4f}",
                   f"{row['f1-score']:.4f}", int(row['support'])] for row in classes.to_dict("records")]
    confusion_rows = [[NAMES[row.Model], row.Actual, row.A, row.D, row.H] for row in confusion.itertuples()]
    doc = f"""# Predictability of club football matches: methodology and results

This is the single study document, following the sequence in [outline.pdf](../../outline.pdf): compare leagues, compare predictions with the odds favourite, then add odds as model inputs. Tables are generated from the numerical exports by [tools/build_report.py](../tools/build_report.py). Percentages are accuracy unless otherwise stated; pp means percentage points. The PDF is the original design draft, not the source of the numerical results below.

**Preparation:** {preparation['Mode']}. **Training:** {training_n:,} labeled matches. **Evaluation window for all leagues:** {evaluation_start}–{evaluation_end} inclusive. **Report built:** {datetime.now(timezone.utc).date().isoformat()} UTC.

## 1. Match predictability across leagues

### 1.1 Question, data and preparation

We test whether match-outcome classifiers achieve different held-out accuracy across domestic leagues. The target has three classes: away win (A), draw (D), home win (H). Observed prediction accuracy under these models does not by itself measure how exciting a league is or explain its popularity.

`data/Matches.csv` is the master table: **{preparation['Rows']:,} matches, 38 division codes and 48 raw columns**, covering July 2000–September 2026. Results and statistics originate from Football-Data; supplied Elo values are based on ClubElo. No external fixture is appended. The primary quotes are the supplied `OddHome/OddDraw/OddAway`, including their missingness. The complete raw input hash is recorded in Appendix C.

Optional Football-Data verification snapshots identify incorrect dates and conflicting outcomes. Source dates are parsed day-first, with Latin-1 fallback; matching normalizes case, accents and punctuation and handles the explicit Tigres alias. Identity requires division, both teams and an exact or day/month-transposed date. Supplemental divisions also require all three supplied odds to agree within `0.0050001`; if all prices are missing, both final scores and the result must agree. Ambiguous or unmatched fixtures stop preparation. Two source amendments cover Santa Clara–Pacos Ferreira (2020-02-01 to 2020-02-02) and Santa Clara–Benfica (2021-01-03 to 2021-01-04). Only generated dates change; conflicting targets become missing rather than being overwritten.

{external_note}

Missing statistics, Elo and histories remain `NaN`; observed zeros remain zero. Duplicate identities stop preparation. Missing and invalid observed values are audited separately; counts can overlap. Only checks with nonzero counts are shown below. Detailed statistics mean shots, shots on target, fouls, corners and yellow/red cards for both teams.

{markdown_table('data_quality', ['Raw-data check', 'Matches', 'Share'], quality_rows)}

All 48 fields, domain errors and per-league counts remain available in [data_quality_fields.csv](data_quality_fields.csv), [data_quality_summary.csv](data_quality_summary.csv) and [data_quality_by_league.csv](data_quality_by_league.csv).

### 1.2 Features, classifiers and evaluation protocol

The base table contains **219 columns: 214 numeric inputs, four metadata fields and one target**. Inputs are current home/away Elo, two history counts and 21 fields for each team's five preceding matches. Each historical match supplies elapsed days, venue, team/opponent Elo, points (3/1/0), and goals, half-time goals, shots, shots on target, fouls, corners and yellow/red cards for and against. `Previous1` is most recent. Histories cross venues, seasons and divisions and group the supplied club names; other renamings are not inferred. Stable date/kickoff ordering puts unknown times first within a date. Shifting precedes joining, so current results and statistics are excluded. Earlier held-out games can become history for later predictions. Metadata, current statistics, precomputed form and odds-derived raw fields are excluded from the base inputs. See the [base field dictionary](../docs/match_prediction_fields.csv).

Training uses all available leagues from **2000-07-01 through 2023-06-30**. Candidate models fit dates before July 2021; selection minimizes log loss on **{validation_n:,} Premier League fixtures in July 2021–June 2023**. Models are then fitted afresh on the full training window. The grid is Naive Bayes `var_smoothing` = 0.01/1/10, logistic `C` = 0.1/1, and boosting `(max_leaf_nodes, l2_regularization)` = (15, 1)/(31, 10). Every primary evaluation uses **{evaluation_start}–{evaluation_end} inclusive**, from season 2023/24 through the run's present day, including available 2026/27 fixtures. Only available known results enter evaluation; there is no test-set tuning. The UTC run date is saved in [evaluation_window.json](evaluation_window.json).

{markdown_table('model_selection', ['Model', 'Selected parameters', 'Validation log loss', 'Reason'], parameter_rows)}

Naive Bayes and logistic regression learn median imputation, missingness indicators and standardization only on training rows. Entirely missing training columns retain zero placeholders and indicators. Naive Bayes probability calculations use float64. Logistic regression uses LBFGS with `max_iter=1000`. Boosting handles NaN directly, with 150 iterations, learning rate 0.05, minimum 50 samples per leaf, random state 42 and no random internal early stopping. Saved pipelines retain preprocessing. The prior baseline always predicts the most frequent training class, with training class frequencies as probabilities.

Accuracy is correct predictions divided by match count; balanced accuracy averages recall over actual classes present; macro F1 averages A/D/H F1, with absent-class zero handling; log loss evaluates ordered class probabilities. Draw recall is reported separately. Lower log loss is better. Exported probabilities use A/D/H order, while input odds use H/D/A. Main tier scores pool matches; equal-league means and calendar-year checks are exported separately.

### 1.3 Top-five league results

These frozen base-model scores use the same {evaluation_start}–{evaluation_end} window as the tier comparison below. The Top-tier row pools exactly these five leagues' fixtures, including the available partial 2026/27 season.

{top_table}

Highest full-window accuracy by classifier is **{winner_text}**. League rankings depend on the window and available inputs; the study does not support one unconditional ranking of league predictability, or a causal conclusion about Premier League popularity. Per-season results are in [top_five_model_metrics.csv](top_five_model_metrics.csv).

### 1.4 Fifteen leagues grouped by prior strength

There are 21 available European national top divisions. Sort their published **2023 UEFA association ranks**: Top is the highest five, Mid the next five, and Lower the lowest five available. The six intermediate divisions remain listed as unselected in [league_strength_2023.csv](../docs/league_strength_2023.csv). Grouping was fixed before examining test scores; ranks are design metadata, not model inputs. These are relative study groups, not official UEFA tiers or the weakest leagues globally. An association coefficient reflects European competition entrants, not domestic competitive balance.

{markdown_table('league_groups', ['Tier', 'Leagues', '2023 UEFA ranks'], group_rows)}

The primary `Available` comparison uses **{evaluation_start}–{evaluation_end} inclusive** for every league, giving **{evaluation_n:,} matches**. Each league contributes every available labeled fixture within those bounds. The supplied dataset ends earlier for some leagues, so a shared evaluation window does not imply equal coverage or match counts. The latest selected fixture is **{last_available}**; later dates through the cutoff have no supplied fixtures. The table below records actual coverage. Smaller leagues and different season formats also produce different counts.

{markdown_table('date_coverage', ['Tier', 'Division', 'Matches', 'First available match', 'Last available match'], date_coverage_rows)}

Secondary `Common` scores truncate every league at the earliest last available match, **{common_end}**, solely as an overlap-window sensitivity. They do not supply the primary tables. Calendar-year checks accommodate summer-season leagues. Unequal actual coverage remains a limitation when interpreting tier differences.

{base_table}

The Lower tier has lower observed accuracy, but its inputs are substantially less complete. Missingness below is measured over feature cells, not the share of fixtures missing any one field. Training counts refer to each tier's own leagues, although every model trains across all available leagues.

{markdown_table('coverage', ['Tier', 'Training fixtures in tier', 'Current Elo cells missing', 'Historical detailed-statistic cells missing'], coverage_rows)}

Coverage, outcome frequencies, training sample sizes and team-strength differences can all affect accuracy. In this run lower-tier Naive Bayes is {lower_nb_gain:+.2f} pp relative to the prior baseline. The gap is descriptive; no significance or causal popularity test is performed. Full league, calendar-year, equal-league and coverage outputs are [league_tier_metrics.csv](league_tier_metrics.csv) and [league_tier_coverage.csv](league_tier_coverage.csv).

## 2. Comparing classifiers with the supplied-odds favourite

Three outcomes do not make 33.33% the only relevant benchmark: the training-prior baseline is stronger than uniform guessing. We therefore compare each classifier against the odds favourite on **the same eligible fixtures**. All three decimal quotes must be finite and greater than one. The minimum quote selects the favourite, with ties resolved H, then D, then A.

Implied probabilities use reciprocal decimal prices and proportional removal of overround:

`pH = (1/oH) / (1/oH + 1/oD + 1/oA)`, with draw and away defined analogously.

Normalizing the decimal prices themselves would reverse their meaning. These are the supplied quotes: a historical Premier League 2023/24 sample matched source averages rather than Bet365, but bookmaker identity and quote timing are not established for every league/season. External average quotes help verify identities and provenance; they never replace a supplied input. The benchmark is called **Supplied odds**. Proportional normalization does not guarantee calibration.

The paired primary comparison contains **{eligible_n:,} eligible fixtures** in {evaluation_start}–{evaluation_end}. The CSVs retain additional overlap-window, calendar-year and tie-sensitivity results.

{market_table}

Across the nine tier/classifier base-model comparisons, the market advantage ranges from **{base_gaps.min():.2f} to {base_gaps.max():.2f} pp**. Pooled favourite accuracy is **{market_pooled.Accuracy:.2%}**, with log loss **{market_pooled.LogLoss:.4f}**. There are {int(market_pooled.TiedFavourites)} tied favourites in the primary cohort; fixed tie priority and unique-favourite sensitivity are exported in [supplied_odds_metrics.csv](supplied_odds_metrics.csv). Accuracy does not measure betting returns, because prices, stakes and transaction costs would also matter.

## 3. Adding odds as model inputs

The augmented table has **252 columns, including 247 numeric inputs**. It adds 33 probabilities: three current home/draw/away values and 30 historical team-perspective values, `{{Home|Away}}_Previous{{1..5}}_Market{{Win|Draw|Loss}}Probability`. Historical probabilities follow that team's venue and are shifted before joining. Missing or invalid odds make all three probabilities missing; original prices remain intact. See the [odds feature dictionary](../docs/market_training_fields.csv).

Step 07 clones the settings selected in step 03 and refits preprocessing and estimators afresh for **both** variants. Both retain the same **{training_n:,} labeled training rows**, including training matches without valid quotes. Test comparisons require valid supplied odds and use identical eligible games. This holds architecture, hyperparameters and cohorts fixed while adding the 33 inputs; it is not a new parameter search.

{augmentation_table}

Odds inputs improve accuracy in **{improving} of the nine** tier/classifier comparisons in this run. Pooled results on the same {eligible_n:,} fixtures are:

{pooled_table}

The largest tier gain is **{best_gain.Group} / {NAMES[best_gain.Model]} ({best_gain.AccuracyChangePP:+.2f} pp)**. Best augmented accuracy in each tier is **{advantage_text}**. Pooled {NAMES[pooled_best.Model]} reaches {pooled_best.Accuracy_WithOdds:.2%} versus {market_pooled.Accuracy:.2%} for the favourite. These differences are descriptive and do not establish statistical significance or betting profitability. Every league/year result and log-loss change is retained in [market_augmented_metrics.csv](market_augmented_metrics.csv) and [market_augmented_accuracy_changes.csv](market_augmented_accuracy_changes.csv).

## Appendix A. Sensitivity to external date/result verification

{sensitivity}

## Appendix B. Outcome-level diagnostics

The Premier League full-window diagnostics below explain why accuracy alone is incomplete: draws are rarely predicted. The baseline's balanced accuracy is one third even though its ordinary accuracy is higher.

{markdown_table('premier_diagnostics', ['Model', 'Matches', 'Accuracy', 'Balanced accuracy', 'Macro F1', 'Log loss', 'Draw recall'], diagnostic_rows)}

{markdown_table('class_metrics', ['Model', 'Actual result', 'Precision', 'Recall', 'F1', 'Support'], class_rows)}

Confusion-matrix rows are actual outcomes; the three count columns are predictions. Labels are A = away win, D = draw, H = home win. Machine-readable versions are [model_class_metrics.csv](model_class_metrics.csv) and [model_confusion_matrices.csv](model_confusion_matrices.csv).

{markdown_table('confusion_matrices', ['Model', 'Actual', 'Predicted A', 'Predicted D', 'Predicted H'], confusion_rows)}

## Appendix C. Reproducibility, evidence and interpretation

The [README](../README.md) gives setup and numbered-run commands. The implementation lives in 00–09; later scripts import earlier ones without running their `main()` functions. Steps 03–07 preserve machine-readable metrics and predictions; step 08 refreshes this document without an earlier sensitivity result, and step 09 refreshes it with the newly fitted sensitivity comparison. `tools/build_report.py` rebuilds it directly from a completed run.

Source snapshots are acquired explicitly in step 00 and reused unless missing or refreshed. The recorded full rerun on 2026-11-23 reused and hashed all 583 cached snapshots, originally acquired as `Downloaded`. Keep snapshots and `download_manifest.csv`: public URL contents can change. The raw input SHA-256 for this report is `{preparation['RawSHA256']}`. Source-verified preparation is optional; `01_clean_data.py --raw-only` followed by 02–08 uses only the master data and omits external evidence. Switching mode requires rerunning downstream steps.

The earlier full 00–09 run and subsequent layout verification are retained as historical evidence in [reproduction/](reproduction/). The evaluation-window correction reuses the fitted estimators and includes available 2026/27 fixtures in every primary comparison. Run the README's unit tests, seasonal replay and independent arithmetic verifier to check the current exports. Seasonal log loss can differ by a few billionths when reconstructed from full-batch float32 predictions; replaying the original seasonal batch reproduces it exactly. Published report precision is unaffected.

The draft PDF has 63 accuracy entries, including repeated cells. [outline_number_comparison.csv](reproduction/outline_number_comparison.csv) records every original value, replacement, cohort size and difference when the independent verifier runs. This document supersedes the PDF's numbers and corrects its ambiguous odds formula. Primary Big Five and tier results use the same requested time bounds; older draft scores cannot be attributed to one historical cause without their missing cohort and model provenance.

Missingness and provisional data remain material limitations. Elo is preserved as supplied, not independently reconstructed; the dataset describes values after June 2025 as a provisional continuation. Date verification does not recover missing technical statistics, guarantee calibrated probabilities, prove intrinsic league uncertainty or support a causal popularity conclusion. The supported findings are differences in these models' observed held-out performance under the stated data and evaluation protocol.

References: [Football-Data](https://www.football-data.co.uk/); [ClubElo](https://www.clubelo.com/); [historical 2023 UEFA association ranking](https://kassiesa.net/uefa/data/method5/crank2023.html), covering coefficients from 2018/19–2022/23 and checked on 2026-11-12; [UEFA coefficient methodology](https://www.uefa.com/nationalassociations/uefarankings/country/about/). The methodology page reflects later rule changes; the numerical ranks here come from the historical table.
"""
    output = report_dir / "study.md"
    output.write_text(doc, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-sensitivity", action="store_true", help="Omit step 09's sensitivity comparison.")
    args = parser.parse_args()
    print(build_report(include_sensitivity=not args.without_sensitivity))


if __name__ == "__main__":
    main()
