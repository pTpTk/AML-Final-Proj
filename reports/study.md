# Predictability of club football matches: methodology and results

This is the single study document, following the sequence in [outline.pdf](../../outline.pdf): compare leagues, compare predictions with the odds favourite, then add odds as model inputs. Tables are generated from the numerical exports by [tools/build_report.py](../tools/build_report.py). Percentages are accuracy unless otherwise stated; pp means percentage points. The PDF is the original design draft, not the source of the numerical results below.

**Preparation:** Source-verified dates/results; supplied odds. **Training:** 209,501 labeled matches. **Main tier window:** July 2023–2024-10-06. **Report built:** 2026-11-23 UTC.

## 1. Match predictability across leagues

### 1.1 Question, data and preparation

We test whether match-outcome classifiers achieve different held-out accuracy across domestic leagues. The target has three classes: away win (A), draw (D), home win (H). Observed prediction accuracy under these models does not by itself measure how exciting a league is or explain its popularity.

`data/Matches.csv` is the master table: **238,858 matches, 38 division codes and 48 raw columns**, covering July 2000–September 2026. Results and statistics originate from Football-Data; supplied Elo values are based on ClubElo. No external fixture is appended. The primary quotes are the supplied `OddHome/OddDraw/OddAway`, including their missingness. The complete raw input hash is recorded in Appendix C.

Optional Football-Data verification snapshots identify incorrect dates and conflicting outcomes. Source dates are parsed day-first, with Latin-1 fallback; matching normalizes case, accents and punctuation and handles the explicit Tigres alias. Identity requires division, both teams and an exact or day/month-transposed date. Supplemental divisions also require all three supplied odds to agree within `0.0050001`; if all prices are missing, both final scores and the result must agree. Ambiguous or unmatched fixtures stop preparation. Two source amendments cover Santa Clara–Pacos Ferreira (2020-02-01 to 2020-02-02) and Santa Clara–Benfica (2021-01-03 to 2021-01-04). Only generated dates change; conflicting targets become missing rather than being overwritten.

Verification changes **19,395 dates**, identifies **1 source-result disagreement**, and moves **1,075 matches** across the July 2023 training boundary. Preparation excludes **5 targets** in total, including originally missing results. It does not fill Elo, statistics or supplied odds. External averages are retained as verification evidence only, including when they could recover a missing or invalid supplied quote.

Missing statistics, Elo and histories remain `NaN`; observed zeros remain zero. Duplicate identities stop preparation. Missing and invalid observed values are audited separately; counts can overlap. Only checks with nonzero counts are shown below. Detailed statistics mean shots, shots on target, fouls, corners and yellow/red cards for both teams.

<!-- table: data_quality -->
| Raw-data check | Matches | Share |
| --- | --- | --- |
| Missing/invalid full-time result | 4 | 0.0017% |
| Missing at least one full-time goal count | 4 | 0.0017% |
| Missing at least one Elo | 98,144 | 41.09% |
| Missing both Elo values | 79,845 | 33.43% |
| Missing at least one half-time goal count | 54,580 | 22.85% |
| Missing at least one detailed statistic | 117,953 | 49.38% |
| Missing all detailed statistics | 111,259 | 46.58% |
| Missing/incomplete odds triplet | 3,042 | 1.27% |
| Complete but invalid odds triplet | 7 | 0.0029% |
| Missing kickoff time | 131,485 | 55.05% |

All 48 fields, domain errors and per-league counts remain available in [data_quality_fields.csv](data_quality_fields.csv), [data_quality_summary.csv](data_quality_summary.csv) and [data_quality_by_league.csv](data_quality_by_league.csv).

### 1.2 Features, classifiers and evaluation protocol

The base table contains **219 columns: 214 numeric inputs, four metadata fields and one target**. Inputs are current home/away Elo, two history counts and 21 fields for each team's five preceding matches. Each historical match supplies elapsed days, venue, team/opponent Elo, points (3/1/0), and goals, half-time goals, shots, shots on target, fouls, corners and yellow/red cards for and against. `Previous1` is most recent. Histories cross venues, seasons and divisions and group the supplied club names; other renamings are not inferred. Stable date/kickoff ordering puts unknown times first within a date. Shifting precedes joining, so current results and statistics are excluded. Earlier held-out games can become history for later predictions. Metadata, current statistics, precomputed form and odds-derived raw fields are excluded from the base inputs. See the [base field dictionary](../docs/match_prediction_fields.csv).

Training uses all available leagues from **2000-07-01 through 2023-06-30**. Candidate models fit dates before July 2021; selection minimizes log loss on **760 Premier League fixtures in July 2021–June 2023**. Models are then fitted afresh on the full training window. The grid is Naive Bayes `var_smoothing` = 0.01/1/10, logistic `C` = 0.1/1, and boosting `(max_leaf_nodes, l2_regularization)` = (15, 1)/(31, 10). The test window is **2023-07-01–2026-06-30**, limited to available known results and elapsed dates; there is no test-set tuning.

<!-- table: model_selection -->
| Model | Selected parameters | Validation log loss | Reason |
| --- | --- | --- | --- |
| Prior baseline | {} | 1.0635 | Always predicts the most frequent training result; probabilities are training class frequencies. Establishes a minimum benchmark. |
| Naive Bayes | {'var_smoothing': 10.0} | 0.9937 | A fast Bayesian classification baseline for numeric inputs. Its conditional-independence and Gaussian assumptions are imperfect for correlated match statistics; variance smoothing limits overconfidence. |
| Logistic regression | {'C': 1.0} | 0.9649 | A regularized, interpretable linear multiclass model. Suitable for many correlated numerical inputs; scaling and regularization control the influence of their different units. |
| Boosting | {'max_leaf_nodes': 31, 'l2_regularization': 10.0} | 0.9687 | Boosted trees capture nonlinear relationships and interactions between Elo, venue, and individual previous games. Histogram binning scales to this dataset and the model handles missing statistics directly. |

Naive Bayes and logistic regression learn median imputation, missingness indicators and standardization only on training rows. Entirely missing training columns retain zero placeholders and indicators. Naive Bayes probability calculations use float64. Logistic regression uses LBFGS with `max_iter=1000`. Boosting handles NaN directly, with 150 iterations, learning rate 0.05, minimum 50 samples per leaf, random state 42 and no random internal early stopping. Saved pipelines retain preprocessing. The prior baseline always predicts the most frequent training class, with training class frequencies as probabilities.

Accuracy is correct predictions divided by match count; balanced accuracy averages recall over actual classes present; macro F1 averages A/D/H F1, with absent-class zero handling; log loss evaluates ordered class probabilities. Draw recall is reported separately. Lower log loss is better. Exported probabilities use A/D/H order, while input odds use H/D/A. Main tier scores pool matches; equal-league means and calendar-year checks are exported separately.

### 1.3 Top-five league results

These frozen base-model scores use the full three-season test window, subject to available labeled matches. They are a different time cohort from the tier comparison below.

<!-- table: top_five -->
| League | Matches | Prior baseline | Naive Bayes | Logistic regression | Boosting |
| --- | --- | --- | --- | --- | --- |
| Premier League (E0) | 1140 | 43.16% | 52.63% | 52.63% | 52.72% |
| La Liga (SP1) | 1140 | 45.79% | 54.04% | 53.42% | 53.51% |
| Bundesliga (D1) | 917 | 42.09% | 51.69% | 52.34% | 52.45% |
| Serie A (I1) | 1140 | 40.18% | 52.89% | 52.63% | 53.16% |
| Ligue 1 (F1) | 917 | 44.06% | 50.05% | 51.69% | 50.93% |

Highest full-window accuracy by classifier is **Naive Bayes: SP1; Logistic regression: SP1; Boosting: SP1**. League rankings depend on the window and available inputs; the study does not support one unconditional ranking of league predictability, or a causal conclusion about Premier League popularity. Per-season results are in [top_five_model_metrics.csv](top_five_model_metrics.csv).

### 1.4 Fifteen leagues grouped by prior strength

There are 21 available European national top divisions. Sort their published **2023 UEFA association ranks**: Top is the highest five, Mid the next five, and Lower the lowest five available. The six intermediate divisions remain listed as unselected in [league_strength_2023.csv](../docs/league_strength_2023.csv). Grouping was fixed before examining test scores; ranks are design metadata, not model inputs. These are relative study groups, not official UEFA tiers or the weakest leagues globally. An association coefficient reflects European competition entrants, not domestic competitive balance.

<!-- table: league_groups -->
| Tier | Leagues | 2023 UEFA ranks |
| --- | --- | --- |
| Top | England E0; Spain SP1; Germany D1; Italy I1; France F1 | 1, 2, 3, 4, 5 |
| Mid | Netherlands N1; Portugal P1; Belgium B1; Scotland SC0; Austria AUT | 6, 7, 8, 9, 10 |
| Lower | Sweden SWE; Poland POL; Romania ROM; Ireland IRL; Finland FIN | 23, 24, 26, 36, 37 |

The primary comparison stops at the earliest last observed match among all 15 leagues: **2024-10-06**, giving **5,407 matches**. Secondary `Available` scores retain each league's dates through June 2026; calendar-year checks accommodate summer-season leagues.

<!-- table: tier_base -->
| Tier | Matches | Prior baseline | Naive Bayes | Logistic regression | Boosting |
| --- | --- | --- | --- | --- | --- |
| Top | 2099 | 42.97% | 52.36% | 53.03% | 52.45% |
| Mid | 1664 | 43.39% | 51.86% | 54.27% | 54.81% |
| Lower | 1644 | 43.92% | 44.16% | 45.44% | 45.13% |

The Lower tier has lower observed accuracy, but its inputs are substantially less complete. Missingness below is measured over feature cells, not the share of fixtures missing any one field. Training counts refer to each tier's own leagues, although every model trains across all available leagues.

<!-- table: coverage -->
| Tier | Training fixtures in tier | Current Elo cells missing | Historical detailed-statistic cells missing |
| --- | --- | --- | --- |
| Top | 40205 | 0.00% | 0.00% |
| Mid | 25380 | 9.95% | 15.34% |
| Lower | 13390 | 74.06% | 100.00% |

Coverage, outcome frequencies, training sample sizes and team-strength differences can all affect accuracy. In this run lower-tier Naive Bayes is +0.24 pp relative to the prior baseline. The gap is descriptive; no significance or causal popularity test is performed. Full league, calendar-year, equal-league and coverage outputs are [league_tier_metrics.csv](league_tier_metrics.csv) and [league_tier_coverage.csv](league_tier_coverage.csv).

## 2. Comparing classifiers with the supplied-odds favourite

Three outcomes do not make 33.33% the only relevant benchmark: the training-prior baseline is stronger than uniform guessing. We therefore compare each classifier against the odds favourite on **the same eligible fixtures**. All three decimal quotes must be finite and greater than one. The minimum quote selects the favourite, with ties resolved H, then D, then A.

Implied probabilities use reciprocal decimal prices and proportional removal of overround:

`pH = (1/oH) / (1/oH + 1/oD + 1/oA)`, with draw and away defined analogously.

Normalizing the decimal prices themselves would reverse their meaning. These are the supplied quotes: a historical Premier League 2023/24 sample matched source averages rather than Bet365, but bookmaker identity and quote timing are not established for every league/season. External average quotes help verify identities and provenance; they never replace a supplied input. The benchmark is called **Supplied odds**. Proportional normalization does not guarantee calibration.

The paired primary comparison contains **5,407 eligible fixtures** through 2024-10-06. The CSVs retain additional available-window, calendar-year and tie-sensitivity results.

<!-- table: tier_market -->
| Tier | Matches | Naive Bayes | Logistic regression | Boosting | Odds favourite |
| --- | --- | --- | --- | --- | --- |
| Top | 2099 | 52.36% | 53.03% | 52.45% | 54.84% |
| Mid | 1664 | 51.86% | 54.27% | 54.81% | 54.99% |
| Lower | 1644 | 44.16% | 45.44% | 45.13% | 48.42% |

Across the nine tier/classifier base-model comparisons, the market advantage ranges from **0.18 to 4.26 pp**. Pooled favourite accuracy is **52.93%**, with log loss **0.9694**. There are 10 tied favourites in the common cohort; fixed tie priority and unique-favourite sensitivity are exported in [supplied_odds_metrics.csv](supplied_odds_metrics.csv). Accuracy does not measure betting returns, because prices, stakes and transaction costs would also matter.

## 3. Adding odds as model inputs

The augmented table has **252 columns, including 247 numeric inputs**. It adds 33 probabilities: three current home/draw/away values and 30 historical team-perspective values, `{Home|Away}_Previous{1..5}_Market{Win|Draw|Loss}Probability`. Historical probabilities follow that team's venue and are shifted before joining. Missing or invalid odds make all three probabilities missing; original prices remain intact. See the [odds feature dictionary](../docs/market_training_fields.csv).

Step 07 clones the settings selected in step 03 and refits preprocessing and estimators afresh for **both** variants. Both retain the same **209,501 labeled training rows**, including training matches without valid quotes. Test comparisons require valid supplied odds and use identical eligible games. This holds architecture, hyperparameters and cohorts fixed while adding the 33 inputs; it is not a new parameter search.

<!-- table: odds_augmentation -->
| Tier | Model | Matches | Without odds | Odds favourite | With odds | Change (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Top | Naive Bayes | 2099 | 52.36% | 54.84% | 53.22% | +0.86 |
| Top | Logistic regression | 2099 | 53.03% | 54.84% | 54.36% | +1.33 |
| Top | Boosting | 2099 | 52.45% | 54.84% | 54.69% | +2.24 |
| Mid | Naive Bayes | 1664 | 51.86% | 54.99% | 52.76% | +0.90 |
| Mid | Logistic regression | 1664 | 54.27% | 54.99% | 54.99% | +0.72 |
| Mid | Boosting | 1664 | 54.81% | 54.99% | 55.53% | +0.72 |
| Lower | Naive Bayes | 1644 | 44.16% | 48.42% | 45.62% | +1.46 |
| Lower | Logistic regression | 1644 | 45.44% | 48.42% | 48.66% | +3.22 |
| Lower | Boosting | 1644 | 45.13% | 48.42% | 48.78% | +3.65 |

Odds inputs improve accuracy in **9 of the nine** tier/classifier comparisons in this run. Pooled results on the same 5,407 fixtures are:

<!-- table: pooled_augmentation -->
| Model | Matches | Without odds | With odds | Change (pp) | Log loss without | Log loss with |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | 5407 | 43.39% | 43.39% | +0.00 | 1.0762 | 1.0762 |
| Naive Bayes | 5407 | 49.71% | 50.77% | +1.05 | 1.0513 | 1.0201 |
| Logistic regression | 5407 | 51.10% | 52.82% | +1.72 | 0.9915 | 0.9693 |
| Boosting | 5407 | 50.95% | 53.15% | +2.20 | 0.9933 | 0.9702 |

The largest tier gain is **Lower / Boosting (+3.65 pp)**. Best augmented accuracy in each tier is **Top: Boosting 54.69% versus favourite 54.84%; Mid: Boosting 55.53% versus favourite 54.99%; Lower: Boosting 48.78% versus favourite 48.42%**. Pooled Boosting reaches 53.15% versus 52.93% for the favourite. These differences are descriptive and do not establish statistical significance or betting profitability. Every league/year result and log-loss change is retained in [market_augmented_metrics.csv](market_augmented_metrics.csv) and [market_augmented_accuracy_changes.csv](market_augmented_accuracy_changes.csv).

## Appendix A. Sensitivity to external date/result verification

Step 09 rebuilds histories and refits both feature variants from `Matches.csv` alone, with the selected settings fixed. Both branches use the same supplied odds. The primary sensitivity comparison contains **5,134 identical fixtures** whose raw and verified dates both fall between July 2023 and 2024-10-06, with agreeing targets and valid supplied odds. This differs from the 5,407-fixture primary odds comparison. Native raw-date and verified-date common windows end on 2024-12-05 and 2024-10-06, respectively, so their unpaired scores are not an estimate of a cleaning effect.

<!-- table: cleaning_sensitivity -->
| Model | Paired matches | Without odds (pp) | With odds (pp) |
| --- | --- | --- | --- |
| Naive Bayes | 5134 | +0.12 | +0.16 |
| Logistic regression | 5134 | +0.23 | -0.19 |
| Boosting | 5134 | +0.37 | +0.06 |

Values are cleaned minus raw accuracy, in percentage points. Date/result verification changes at least one base input for **54,369 matches (22.76%)**. Cleaning protects chronology; it does not improve every classifier's accuracy. Raw-only training contains 209,648 matches versus 209,501 in the main branch. Elo and missing statistics are preserved in both branches.

## Appendix B. Outcome-level diagnostics

The Premier League full-window diagnostics below explain why accuracy alone is incomplete: draws are rarely predicted. The baseline's balanced accuracy is one third even though its ordinary accuracy is higher.

<!-- table: premier_diagnostics -->
| Model | Matches | Accuracy | Balanced accuracy | Macro F1 | Log loss | Draw recall |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | 1140 | 43.16% | 33.33% | 0.2010 | 1.0755 | 0.00% |
| Naive Bayes | 1140 | 52.63% | 44.94% | 0.3890 | 1.0182 | 0.00% |
| Logistic regression | 1140 | 52.63% | 45.10% | 0.3900 | 0.9901 | 0.00% |
| Boosting | 1140 | 52.72% | 45.11% | 0.3909 | 0.9889 | 0.36% |

<!-- table: class_metrics -->
| Model | Actual result | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| Prior baseline | A | 0.0000 | 0.0000 | 0.0000 | 369 |
| Prior baseline | D | 0.0000 | 0.0000 | 0.0000 | 279 |
| Prior baseline | H | 0.4316 | 1.0000 | 0.6029 | 492 |
| Naive Bayes | A | 0.5322 | 0.5149 | 0.5234 | 369 |
| Naive Bayes | D | 0.0000 | 0.0000 | 0.0000 | 279 |
| Naive Bayes | H | 0.5243 | 0.8333 | 0.6436 | 492 |
| Logistic regression | A | 0.5157 | 0.5339 | 0.5246 | 369 |
| Logistic regression | D | 0.0000 | 0.0000 | 0.0000 | 279 |
| Logistic regression | H | 0.5324 | 0.8191 | 0.6453 | 492 |
| Boosting | A | 0.5093 | 0.5203 | 0.5147 | 369 |
| Boosting | D | 1.0000 | 0.0036 | 0.0071 | 279 |
| Boosting | H | 0.5354 | 0.8293 | 0.6507 | 492 |

Confusion-matrix rows are actual outcomes; the three count columns are predictions. Labels are A = away win, D = draw, H = home win. Machine-readable versions are [model_class_metrics.csv](model_class_metrics.csv) and [model_confusion_matrices.csv](model_confusion_matrices.csv).

<!-- table: confusion_matrices -->
| Model | Actual | Predicted A | Predicted D | Predicted H |
| --- | --- | --- | --- | --- |
| Prior baseline | A | 0 | 0 | 369 |
| Prior baseline | D | 0 | 0 | 279 |
| Prior baseline | H | 0 | 0 | 492 |
| Naive Bayes | A | 190 | 0 | 179 |
| Naive Bayes | D | 86 | 0 | 193 |
| Naive Bayes | H | 81 | 1 | 410 |
| Logistic regression | A | 197 | 1 | 171 |
| Logistic regression | D | 96 | 0 | 183 |
| Logistic regression | H | 89 | 0 | 403 |
| Boosting | A | 192 | 0 | 177 |
| Boosting | D | 101 | 1 | 177 |
| Boosting | H | 84 | 0 | 408 |

## Appendix C. Reproducibility, evidence and interpretation

The [README](../README.md) gives setup and numbered-run commands. The implementation lives in 00–09; later scripts import earlier ones without running their `main()` functions. Steps 03–07 preserve machine-readable metrics and predictions; step 08 refreshes this document without an earlier sensitivity result, and step 09 refreshes it with the newly fitted sensitivity comparison. `tools/build_report.py` rebuilds it directly from a completed run.

Source snapshots are acquired explicitly in step 00 and reused unless missing or refreshed. The recorded full rerun on 2026-11-23 reused and hashed all 583 cached snapshots, originally acquired as `Downloaded`. Keep snapshots and `download_manifest.csv`: public URL contents can change. The raw input SHA-256 for this report is `ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221`. Source-verified preparation is optional; `01_clean_data.py --raw-only` followed by 02–08 uses only the master data and omits external evidence. Switching mode requires rerunning downstream steps.

The verified full 00–09 run passed 35 existing tests; after cleanup, the suite contains 36 tests, including a raw-only stale-audit regression. Independent arithmetic checked 4,206 metric rows (38,058 scalar scores), with 45 verification checks passing. The subsequent file-layout cleanup preserved all model artifacts and regenerated data. These statements describe the recorded verification; logs, commands, hashes, table-by-table checks and layout evidence are in [reproduction/](reproduction/). Seasonal log loss can differ by a few billionths when reconstructed from full-batch float32 predictions; replaying the original seasonal batch reproduces it exactly. Published report precision is unaffected.

The draft PDF has 63 accuracy entries, including repeated cells; 13 agree and 50 differ at its one-decimal-percent precision. [outline_number_comparison.csv](reproduction/outline_number_comparison.csv) records every original value, replacement, cohort size and difference. This document supersedes the PDF's numbers and corrects its ambiguous odds formula. Different time windows must remain labeled; old draft scores cannot be attributed to one historical cause without their missing cohort and model provenance.

Missingness and provisional data remain material limitations. Elo is preserved as supplied, not independently reconstructed; the dataset describes values after June 2025 as a provisional continuation. Date verification does not recover missing technical statistics, guarantee calibrated probabilities, prove intrinsic league uncertainty or support a causal popularity conclusion. The supported findings are differences in these models' observed held-out performance under the stated data and evaluation protocol.

References: [Football-Data](https://www.football-data.co.uk/); [ClubElo](https://www.clubelo.com/); [historical 2023 UEFA association ranking](https://kassiesa.net/uefa/data/method5/crank2023.html), covering coefficients from 2018/19–2022/23 and checked on 2026-11-12; [UEFA coefficient methodology](https://www.uefa.com/nationalassociations/uefarankings/country/about/). The methodology page reflects later rule changes; the numerical ranks here come from the historical table.
