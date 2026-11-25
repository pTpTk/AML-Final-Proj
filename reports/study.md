# Predictability of club football matches: methodology and results

This is the single study document, following the sequence in [outline.pdf](../../outline.pdf): compare leagues, compare predictions with the odds favourite, then add odds as model inputs. Tables are generated from the numerical exports by [tools/build_report.py](../tools/build_report.py). Percentages are accuracy unless otherwise stated; pp means percentage points. The PDF is the original design draft, not the source of the numerical results below.

**Preparation:** Source-verified dates/results; supplied odds. **Training:** 209,501 labeled matches. **Evaluation window for all leagues:** 2023-07-01–2026-11-25 inclusive. **Report built:** 2026-11-25 UTC.

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

Training uses all available leagues from **2000-07-01 through 2023-06-30**. Candidate models fit dates before July 2021; selection minimizes log loss on **760 Premier League fixtures in July 2021–June 2023**. Models are then fitted afresh on the full training window. The grid is Naive Bayes `var_smoothing` = 0.01/1/10, logistic `C` = 0.1/1, and boosting `(max_leaf_nodes, l2_regularization)` = (15, 1)/(31, 10). Every primary evaluation uses **2023-07-01–2026-11-25 inclusive**, from season 2023/24 through the run's present day, including available 2026/27 fixtures. Only available known results enter evaluation; there is no test-set tuning. The UTC run date is saved in [evaluation_window.json](evaluation_window.json).

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

These frozen base-model scores use the same 2023-07-01–2026-11-25 window as the tier comparison below. The Top-tier row pools exactly these five leagues' fixtures, including the available partial 2026/27 season.

<!-- table: top_five -->
| League | Matches | Prior baseline | Naive Bayes | Logistic regression | Boosting |
| --- | --- | --- | --- | --- | --- |
| Premier League (E0) | 1160 | 43.28% | 52.59% | 52.59% | 52.67% |
| La Liga (SP1) | 1171 | 45.94% | 53.71% | 53.46% | 53.37% |
| Bundesliga (D1) | 926 | 42.44% | 51.73% | 52.38% | 52.48% |
| Serie A (I1) | 1160 | 40.17% | 53.19% | 52.84% | 53.45% |
| Ligue 1 (F1) | 936 | 43.80% | 50.00% | 51.60% | 50.85% |

Highest full-window accuracy by classifier is **Naive Bayes: SP1; Logistic regression: SP1; Boosting: I1**. League rankings depend on the window and available inputs; the study does not support one unconditional ranking of league predictability, or a causal conclusion about Premier League popularity. Per-season results are in [top_five_model_metrics.csv](top_five_model_metrics.csv).

### 1.4 Fifteen leagues grouped by prior strength

There are 21 available European national top divisions. Sort their published **2023 UEFA association ranks**: Top is the highest five, Mid the next five, and Lower the lowest five available. The six intermediate divisions remain listed as unselected in [league_strength_2023.csv](../docs/league_strength_2023.csv). Grouping was fixed before examining test scores; ranks are design metadata, not model inputs. These are relative study groups, not official UEFA tiers or the weakest leagues globally. An association coefficient reflects European competition entrants, not domestic competitive balance.

<!-- table: league_groups -->
| Tier | Leagues | 2023 UEFA ranks |
| --- | --- | --- |
| Top | England E0; Spain SP1; Germany D1; Italy I1; France F1 | 1, 2, 3, 4, 5 |
| Mid | Netherlands N1; Portugal P1; Belgium B1; Scotland SC0; Austria AUT | 6, 7, 8, 9, 10 |
| Lower | Sweden SWE; Poland POL; Romania ROM; Ireland IRL; Finland FIN | 23, 24, 26, 36, 37 |

The primary `Available` comparison uses **2023-07-01–2026-11-25 inclusive** for every league, giving **10,802 matches**. Each league contributes every available labeled fixture within those bounds. The supplied dataset ends earlier for some leagues, so a shared evaluation window does not imply equal coverage or match counts. The latest selected fixture is **2026-09-03**; later dates through the cutoff have no supplied fixtures. The table below records actual coverage. Smaller leagues and different season formats also produce different counts.

<!-- table: date_coverage -->
| Tier | Division | Matches | First available match | Last available match |
| --- | --- | --- | --- | --- |
| Top | D1 | 926 | 2023-08-18 | 2026-08-30 |
| Top | E0 | 1160 | 2023-08-11 | 2026-08-31 |
| Top | F1 | 936 | 2023-08-11 | 2026-09-03 |
| Top | I1 | 1160 | 2023-08-19 | 2026-08-31 |
| Top | SP1 | 1171 | 2023-08-11 | 2026-09-03 |
| Mid | AUT | 247 | 2023-07-28 | 2024-10-06 |
| Mid | B1 | 971 | 2023-07-28 | 2026-09-03 |
| Mid | N1 | 951 | 2023-08-11 | 2026-08-30 |
| Mid | P1 | 951 | 2023-08-11 | 2026-08-31 |
| Mid | SC0 | 681 | 2023-08-05 | 2026-09-03 |
| Lower | FIN | 248 | 2023-07-01 | 2024-10-06 |
| Lower | IRL | 232 | 2023-07-07 | 2024-10-14 |
| Lower | POL | 403 | 2023-07-21 | 2024-10-06 |
| Lower | ROM | 415 | 2023-07-14 | 2024-10-06 |
| Lower | SWE | 350 | 2023-07-01 | 2024-10-06 |

Secondary `Common` scores truncate every league at the earliest last available match, **2024-10-06**, solely as an overlap-window sensitivity. They do not supply the primary tables. Calendar-year checks accommodate summer-season leagues. Unequal actual coverage remains a limitation when interpreting tier differences.

<!-- table: tier_base -->
| Tier | Matches | Prior baseline | Naive Bayes | Logistic regression | Boosting |
| --- | --- | --- | --- | --- | --- |
| Top | 5353 | 43.13% | 52.36% | 52.62% | 52.64% |
| Mid | 3801 | 43.15% | 51.35% | 52.83% | 53.09% |
| Lower | 1648 | 43.81% | 44.05% | 45.39% | 45.08% |

The Lower tier has lower observed accuracy, but its inputs are substantially less complete. Missingness below is measured over feature cells, not the share of fixtures missing any one field. Training counts refer to each tier's own leagues, although every model trains across all available leagues.

<!-- table: coverage -->
| Tier | Training fixtures in tier | Current Elo cells missing | Historical detailed-statistic cells missing |
| --- | --- | --- | --- |
| Top | 40205 | 0.02% | 0.06% |
| Mid | 25380 | 4.70% | 6.96% |
| Lower | 13390 | 74.12% | 100.00% |

Coverage, outcome frequencies, training sample sizes and team-strength differences can all affect accuracy. In this run lower-tier Naive Bayes is +0.24 pp relative to the prior baseline. The gap is descriptive; no significance or causal popularity test is performed. Full league, calendar-year, equal-league and coverage outputs are [league_tier_metrics.csv](league_tier_metrics.csv) and [league_tier_coverage.csv](league_tier_coverage.csv).

## 2. Comparing classifiers with the supplied-odds favourite

Three outcomes do not make 33.33% the only relevant benchmark: the training-prior baseline is stronger than uniform guessing. We therefore compare each classifier against the odds favourite on **the same eligible fixtures**. All three decimal quotes must be finite and greater than one. The minimum quote selects the favourite, with ties resolved H, then D, then A.

Implied probabilities use reciprocal decimal prices and proportional removal of overround:

`pH = (1/oH) / (1/oH + 1/oD + 1/oA)`, with draw and away defined analogously.

Normalizing the decimal prices themselves would reverse their meaning. These are the supplied quotes: a historical Premier League 2023/24 sample matched source averages rather than Bet365, but bookmaker identity and quote timing are not established for every league/season. External average quotes help verify identities and provenance; they never replace a supplied input. The benchmark is called **Supplied odds**. Proportional normalization does not guarantee calibration.

The paired primary comparison contains **10,802 eligible fixtures** in 2023-07-01–2026-11-25. The CSVs retain additional overlap-window, calendar-year and tie-sensitivity results.

<!-- table: tier_market -->
| Tier | Matches | Naive Bayes | Logistic regression | Boosting | Odds favourite |
| --- | --- | --- | --- | --- | --- |
| Top | 5353 | 52.36% | 52.62% | 52.64% | 54.04% |
| Mid | 3801 | 51.35% | 52.83% | 53.09% | 53.91% |
| Lower | 1648 | 44.05% | 45.39% | 45.08% | 48.36% |

Across the nine tier/classifier base-model comparisons, the market advantage ranges from **0.82 to 4.31 pp**. Pooled favourite accuracy is **53.13%**, with log loss **0.9703**. There are 63 tied favourites in the primary cohort; fixed tie priority and unique-favourite sensitivity are exported in [supplied_odds_metrics.csv](supplied_odds_metrics.csv). Accuracy does not measure betting returns, because prices, stakes and transaction costs would also matter.

## 3. Adding odds as model inputs

The augmented table has **252 columns, including 247 numeric inputs**. It adds 33 probabilities: three current home/draw/away values and 30 historical team-perspective values, `{Home|Away}_Previous{1..5}_Market{Win|Draw|Loss}Probability`. Historical probabilities follow that team's venue and are shifted before joining. Missing or invalid odds make all three probabilities missing; original prices remain intact. See the [odds feature dictionary](../docs/market_training_fields.csv).

Step 07 clones the settings selected in step 03 and refits preprocessing and estimators afresh for **both** variants. Both retain the same **209,501 labeled training rows**, including training matches without valid quotes. Test comparisons require valid supplied odds and use identical eligible games. This holds architecture, hyperparameters and cohorts fixed while adding the 33 inputs; it is not a new parameter search.

<!-- table: odds_augmentation -->
| Tier | Model | Matches | Without odds | Odds favourite | With odds | Change (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Top | Naive Bayes | 5353 | 52.36% | 54.04% | 52.36% | +0.00 |
| Top | Logistic regression | 5353 | 52.62% | 54.04% | 53.56% | +0.93 |
| Top | Boosting | 5353 | 52.64% | 54.04% | 53.75% | +1.10 |
| Mid | Naive Bayes | 3801 | 51.35% | 53.91% | 51.70% | +0.34 |
| Mid | Logistic regression | 3801 | 52.83% | 53.91% | 53.85% | +1.03 |
| Mid | Boosting | 3801 | 53.09% | 53.91% | 54.01% | +0.92 |
| Lower | Naive Bayes | 1648 | 44.05% | 48.36% | 45.51% | +1.46 |
| Lower | Logistic regression | 1648 | 45.39% | 48.36% | 48.54% | +3.16 |
| Lower | Boosting | 1648 | 45.08% | 48.36% | 48.73% | +3.64 |

Odds inputs improve accuracy in **8 of the nine** tier/classifier comparisons in this run. Pooled results on the same 10,802 fixtures are:

<!-- table: pooled_augmentation -->
| Model | Matches | Without odds | With odds | Change (pp) | Log loss without | Log loss with |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | 10802 | 43.24% | 43.24% | +0.00 | 1.0763 | 1.0763 |
| Naive Bayes | 10802 | 50.74% | 51.08% | +0.34 | 1.0422 | 1.0177 |
| Logistic regression | 10802 | 51.59% | 52.90% | +1.31 | 0.9910 | 0.9725 |
| Boosting | 10802 | 51.65% | 53.07% | +1.43 | 0.9899 | 0.9722 |

The largest tier gain is **Lower / Boosting (+3.64 pp)**. Best augmented accuracy in each tier is **Top: Boosting 53.75% versus favourite 54.04%; Mid: Boosting 54.01% versus favourite 53.91%; Lower: Boosting 48.73% versus favourite 48.36%**. Pooled Boosting reaches 53.07% versus 53.13% for the favourite. These differences are descriptive and do not establish statistical significance or betting profitability. Every league/year result and log-loss change is retained in [market_augmented_metrics.csv](market_augmented_metrics.csv) and [market_augmented_accuracy_changes.csv](market_augmented_accuracy_changes.csv).

## Appendix A. Sensitivity to external date/result verification

Step 09 rebuilds histories and fits both feature variants from `Matches.csv` alone, with the selected settings fixed. Saved models can be reused with `--evaluate-only`. Both branches use the same supplied odds. The primary sensitivity comparison contains **10,631 identical fixtures** whose raw and verified dates both fall between 2023-07-01 and 2026-11-25, with agreeing targets and valid supplied odds. This differs from the 10,802-fixture primary odds comparison because pairing also requires agreement on eligibility under both date treatments. Secondary overlap-window scores are retained separately; their raw-date and verified-date cutoffs are 2024-12-05 and 2024-10-06, respectively.

<!-- table: cleaning_sensitivity -->
| Model | Paired matches | Without odds (pp) | With odds (pp) |
| --- | --- | --- | --- |
| Naive Bayes | 10631 | +0.01 | +0.13 |
| Logistic regression | 10631 | +0.08 | -0.18 |
| Boosting | 10631 | +0.31 | -0.04 |

Values are cleaned minus raw accuracy, in percentage points. Date/result verification changes at least one base input for **54,369 matches (22.76%)**. Cleaning protects chronology; it does not improve every classifier's accuracy. Raw-only training contains 209,648 matches versus 209,501 in the main branch. Elo and missing statistics are preserved in both branches.

## Appendix B. Outcome-level diagnostics

The Premier League full-window diagnostics below explain why accuracy alone is incomplete: draws are rarely predicted. The baseline's balanced accuracy is one third even though its ordinary accuracy is higher.

<!-- table: premier_diagnostics -->
| Model | Matches | Accuracy | Balanced accuracy | Macro F1 | Log loss | Draw recall |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | 1160 | 43.28% | 33.33% | 0.2014 | 1.0749 | 0.00% |
| Naive Bayes | 1160 | 52.59% | 44.82% | 0.3883 | 1.0182 | 0.00% |
| Logistic regression | 1160 | 52.59% | 44.98% | 0.3892 | 0.9917 | 0.00% |
| Boosting | 1160 | 52.67% | 44.99% | 0.3900 | 0.9891 | 0.35% |

<!-- table: class_metrics -->
| Model | Actual result | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- | --- |
| Prior baseline | A | 0.0000 | 0.0000 | 0.0000 | 375 |
| Prior baseline | D | 0.0000 | 0.0000 | 0.0000 | 283 |
| Prior baseline | H | 0.4328 | 1.0000 | 0.6041 | 502 |
| Naive Bayes | A | 0.5319 | 0.5120 | 0.5217 | 375 |
| Naive Bayes | D | 0.0000 | 0.0000 | 0.0000 | 283 |
| Naive Bayes | H | 0.5238 | 0.8327 | 0.6431 | 502 |
| Logistic regression | A | 0.5142 | 0.5307 | 0.5223 | 375 |
| Logistic regression | D | 0.0000 | 0.0000 | 0.0000 | 283 |
| Logistic regression | H | 0.5324 | 0.8187 | 0.6452 | 502 |
| Boosting | A | 0.5065 | 0.5173 | 0.5119 | 375 |
| Boosting | D | 1.0000 | 0.0035 | 0.0070 | 283 |
| Boosting | H | 0.5361 | 0.8287 | 0.6510 | 502 |

Confusion-matrix rows are actual outcomes; the three count columns are predictions. Labels are A = away win, D = draw, H = home win. Machine-readable versions are [model_class_metrics.csv](model_class_metrics.csv) and [model_confusion_matrices.csv](model_confusion_matrices.csv).

<!-- table: confusion_matrices -->
| Model | Actual | Predicted A | Predicted D | Predicted H |
| --- | --- | --- | --- | --- |
| Prior baseline | A | 0 | 0 | 375 |
| Prior baseline | D | 0 | 0 | 283 |
| Prior baseline | H | 0 | 0 | 502 |
| Naive Bayes | A | 192 | 0 | 183 |
| Naive Bayes | D | 86 | 0 | 197 |
| Naive Bayes | H | 83 | 1 | 418 |
| Logistic regression | A | 199 | 1 | 175 |
| Logistic regression | D | 97 | 0 | 186 |
| Logistic regression | H | 91 | 0 | 411 |
| Boosting | A | 194 | 0 | 181 |
| Boosting | D | 103 | 1 | 179 |
| Boosting | H | 86 | 0 | 416 |

## Appendix C. Reproducibility, evidence and interpretation

The [README](../README.md) gives setup and numbered-run commands. The implementation lives in 00–09; later scripts import earlier ones without running their `main()` functions. Steps 03–07 preserve machine-readable metrics and predictions; step 08 refreshes this document without an earlier sensitivity result, and step 09 refreshes it with the newly fitted sensitivity comparison. `tools/build_report.py` rebuilds it directly from a completed run.

Source snapshots are acquired explicitly in step 00 and reused unless missing or refreshed. The recorded full rerun on 2026-11-23 reused and hashed all 583 cached snapshots, originally acquired as `Downloaded`. Keep snapshots and `download_manifest.csv`: public URL contents can change. The raw input SHA-256 for this report is `ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221`. Source-verified preparation is optional; `01_clean_data.py --raw-only` followed by 02–08 uses only the master data and omits external evidence. Switching mode requires rerunning downstream steps.

The earlier full 00–09 run and subsequent layout verification are retained as historical evidence in [reproduction/](reproduction/). The evaluation-window correction reuses the fitted estimators and includes available 2026/27 fixtures in every primary comparison. Run the README's unit tests, seasonal replay and independent arithmetic verifier to check the current exports. Seasonal log loss can differ by a few billionths when reconstructed from full-batch float32 predictions; replaying the original seasonal batch reproduces it exactly. Published report precision is unaffected.

The draft PDF has 63 accuracy entries, including repeated cells. [outline_number_comparison.csv](reproduction/outline_number_comparison.csv) records every original value, replacement, cohort size and difference when the independent verifier runs. This document supersedes the PDF's numbers and corrects its ambiguous odds formula. Primary Big Five and tier results use the same requested time bounds; older draft scores cannot be attributed to one historical cause without their missing cohort and model provenance.

Missingness and provisional data remain material limitations. Elo is preserved as supplied, not independently reconstructed; the dataset describes values after June 2025 as a provisional continuation. Date verification does not recover missing technical statistics, guarantee calibrated probabilities, prove intrinsic league uncertainty or support a causal popularity conclusion. The supported findings are differences in these models' observed held-out performance under the stated data and evaluation protocol.

References: [Football-Data](https://www.football-data.co.uk/); [ClubElo](https://www.clubelo.com/); [historical 2023 UEFA association ranking](https://kassiesa.net/uefa/data/method5/crank2023.html), covering coefficients from 2018/19–2022/23 and checked on 2026-11-12; [UEFA coefficient methodology](https://www.uefa.com/nationalassociations/uefarankings/country/about/). The methodology page reflects later rule changes; the numerical ranks here come from the historical table.
