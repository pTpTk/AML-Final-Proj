# Club football match predictability / 足球比赛结果的可预测性

## English

The analysis follows [the outline](../outline.pdf): compare predictability across leagues, compare classifiers with the odds favourite, then add odds as model inputs. It keeps Gaussian Naive Bayes, logistic regression, histogram gradient boosting and a class-prior baseline. The neural-network and standalone Bet365 experiments are removed.

**`data/Matches.csv` is the master match dataset. The primary analysis uses its `OddHome/OddDraw/OddAway` prices, including their original missingness. No external odds fill or replace them.** External Football-Data files are optional verification data for correcting dates and checking result labels; their acquisition is an explicit source-code step. No downloaded match is added to the master match list.

### Why external data was introduced

The original odds audit used Football-Data's league/season files to establish what the generic `OddHome/OddDraw/OddAway` columns represented. Comparing them with explicitly named bookmaker and market-average fields showed that the audited Premier League 2023/24 prices matched `AvgH/AvgD/AvgA`, rather than Bet365's `B365H/B365D/B365A`. This justified checking the odds' provenance before interpreting the market benchmark; it did not establish one source or quote timing for every league and season.

Those source files also provide match dates and results, making it possible to check the master table's chronology. This matters because a day/month transposition can produce a valid-looking date that ordinary parsing accepts. An incorrect date can select the wrong previous five games or put a match on the wrong side of the training/test boundary. Verification identified 19,395 date differences and 1,075 matches crossing that boundary, which motivates retaining an optional external-cleaning step.

The current workflow therefore uses external files as verification evidence while keeping the supplied odds as the primary inputs. Missing odds, Elo and statistics remain explicit. Step 00 documents acquisition and records file hashes; `01_clean_data.py --raw-only` provides a workflow without external verification. Step 09 measures the effect of that choice on identical held-out games, rather than assuming verification must improve accuracy.

### Run the numbered steps

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

Step 00 derives URLs from the divisions/seasons in `Matches.csv`, downloads the required public [Football-Data](https://www.football-data.co.uk/) CSVs to **`data/football_data_sources/`**, and records URLs, SHA-256 hashes, sizes, check timestamps and downloaded/cached status in `download_manifest.csv`. It works from an empty directory. Existing snapshots are reused; `--refresh` redownloads them. A failed download is recorded and stops the step. The implementation is in [src/data_sources.py](src/data_sources.py). This run fetched all **583 files afresh**; the old `bet365_sources` directory is no longer a dependency. Public URLs can change over time: retain snapshots and their manifest to reproduce exact results. Only step 00 needs network access after installing dependencies.

**To run using `Matches.csv` alone**, skip step 00 and replace step 01 with:

```bash
.venv/bin/python 01_clean_data.py --raw-only
```

Then run steps 02–08 normally. In this mode there are no external date corrections or source-result exclusions. Step 09 is an additional comparison that requires a completed source-verified run; its raw branch independently rebuilds features and refits models from `Matches.csv` alone, without overwriting the main outputs. Switching the preparation mode requires rerunning downstream steps. Steps overwrite their own generated outputs; raw input CSVs remain intact. `reports/data_preparation.json` records the mode, input hash, changes and odds provenance.

| File | Behavior | Main outputs |
| --- | --- | --- |
| [00_download_sources.py](00_download_sources.py) | Explicitly acquire public verification files, with retry, caching and provenance. | `data/football_data_sources/*.csv`, `download_manifest.csv` |
| [01_clean_data.py](01_clean_data.py) | Prepare the supplied odds; by default verify dates/results against the downloaded files. `--raw-only` skips verification. | `data/MatchesChronological.csv`; `reports/data_preparation.json`, `data_cleaning_*.csv` |
| [02_build_features.py](02_build_features.py) | Build pre-match Elo and five separate historical games per team. | `data/MatchTraining.csv` |
| [03_train_models.py](03_train_models.py) | Select parameters on earlier validation, refit on all training leagues, evaluate the Premier League. | `models/*.joblib`; `reports/model_evaluation.md`, `model_*.csv`, `validation_metrics.csv` |
| [04_evaluate_leagues.py](04_evaluate_leagues.py) | Evaluate frozen models on the top five leagues and 15 leagues grouped into three tiers. | `reports/top_five_evaluation.md`, `top_five_model_*.csv`, `league_tier_evaluation.md`, `league_tier_*.csv` |
| [05_compare_market.py](05_compare_market.py) | Compare every classifier and the favourite from the supplied odds on identical eligible games. | `reports/market_average_evaluation.md`, `market_average_*.csv` (historical filenames) |
| [06_add_market_features.py](06_add_market_features.py) | Add 3 current and 30 historical probabilities from the supplied odds. Check row alignment. | `data/MarketTraining.csv`; `docs/market_training_fields.csv` |
| [07_train_market_models.py](07_train_market_models.py) | Fit the same selected classifiers with/without the 33 odds features on identical training rows. | `models/market_augmented/`; `reports/market_augmented_*` |
| [08_audit_data.py](08_audit_data.py) | Audit every raw field, missingness, invalid values, consistency and optional external discrepancies. | [reports/data_quality.md](reports/data_quality.md); `data_quality_*.csv` |
| [09_compare_without_external.py](09_compare_without_external.py) | Refit a raw-only branch with fixed model settings and compare against source-verified dates/results. Both branches use the supplied odds. | [reports/external_data_impact.md](reports/external_data_impact.md); `external_data_*.csv`, `raw_only_predictions.csv`; `models/raw_only/` |

The numbered files contain the analysis steps; `src/` holds reusable functions and `tests/` checks meaningful behavior. `EloRatings.csv`, `E1_Matches.csv` and `F1_Matches.csv` are retained source material but are not runtime inputs. Classifier inputs use the Elo values already in `Matches.csv`.

### Features and protocol

The raw file has **238,858 matches, 38 division codes and 48 columns**, covering July 2000–September 2026. Results/statistics originated from Football-Data and supplied Elo values are based on [ClubElo](https://www.clubelo.com/). The base table has **219 columns: 214 numeric inputs, four metadata fields and one target**. Metadata (`Division`, `MatchDate`, `HomeTeam`, `AwayTeam`) and target (`FTResult`: H/D/A) are excluded from numeric inputs.

The 214 inputs are current home/away Elo, two history counts, and 21 fields for each team's five previous games. Each game includes elapsed days, historical venue, team/opponent Elo, result points (3/1/0), and for/against goals, half-time goals, shots, shots on target, fouls, corners, yellow cards and red cards. `Previous1` is most recent. Histories cross seasons/divisions and include both venues. The field dictionary is [docs/match_prediction_fields.csv](docs/match_prediction_fields.csv).

The odds-augmented table has **252 columns, including 247 inputs**. Its 33 extra inputs are current `MarketHomeProbability/MarketDrawProbability/MarketAwayProbability` and `{Home|Away}_Previous{1..5}_Market{Win|Draw|Loss}Probability`. Historical probabilities follow the team's perspective and venue. See [docs/market_training_fields.csv](docs/market_training_fields.csv).

- Training: all available leagues, **2000-07-01–2023-06-30**.
- Parameter selection: fit candidates before July 2021, validate on Premier League (`E0`) games in July 2021–June 2023; select by validation log loss and refit on the whole training window. No test-set tuning.
- Evaluation: **2023-07-01–2026-06-30**, limited by available data. Calendar-year leagues use the same bounds. The top-five report also breaks results down by season.
- The primary tier comparison stops at the earliest last match among the 15 leagues. With verified dates this is **2024-10-06**; raw recorded dates give **2024-12-05**. `Available` scores retain each league's available test dates and calendar-year checks.
- The final odds experiment clones the selected settings and fits both preprocessing and estimators afresh. Both variants retain identical training rows, including games with missing odds. Paired test comparisons require complete, finite decimal odds above one.
- Step 09 holds model settings and supplied odds fixed. Its primary accuracy comparison uses identical matches whose **raw and verified dates are both in the held-out window**, with agreeing targets. It separates this from native-window results with different cohorts. It does not redo parameter selection.

| Tier | Leagues | 2023 UEFA ranks |
| --- | --- | --- |
| Top | England E0, Spain SP1, Germany D1, Italy I1, France F1 | 1–5 |
| Mid | Netherlands N1, Portugal P1, Belgium B1, Scotland SC0, Austria AUT | 6–10 |
| Lower | Sweden SWE, Poland POL, Romania ROM, Ireland IRL, Finland FIN | 23, 24, 26, 36, 37 |

The fixed grouping is documented in [selection evidence](docs/league_tier_selection.md) and [ranking data](docs/league_strength_2023.csv). These are externally sourced study-design metadata, not downloaded match features. “Lower” is relative to available European top divisions. Reports include accuracy, balanced accuracy, macro F1, log loss, draw recall, pooled/equal-league accuracy and coverage where applicable. Exported probability columns use A/D/H order; input odds use H/D/A.

### Cleansing and reasons for optional external data

1. **Keep missingness explicit.** Unavailable statistics, Elo and history stay `NaN`; observed zero stays zero. Invalid/unknown H/D/A targets are excluded from supervision. Duplicate match identities stop preparation rather than being silently discarded. The raw audit separates missing cells from invalid observed values and flags cross-field contradictions.
2. **Validate supplied odds.** All three values must be finite and greater than one. If any fails, all three derived probabilities are missing; original odds columns remain intact. No quote is downloaded to fill a gap. A match without valid odds remains usable for training through the appropriate missing-value handling, but is excluded from paired odds-benchmark evaluation.
3. **Optionally verify chronology and targets.** This is the reason to download external files: ISO-looking raw dates can be syntactically valid while having day/month transpositions, which alter chronological splits and team histories. Match league, both teams and exact/transposed date candidates; supplemental leagues additionally require all three stored odds within `0.0050001` for rounding. If all stored prices are missing, require both final goal counts and result instead. Reject ambiguous/unmatched identities. Two explicit source amendments cover Santa Clara–Pacos Ferreira (2020-02-01 → 2020-02-02) and Santa Clara–Benfica (2021-01-03 → 2021-01-04). Source dates replace dates only in the generated copy; conflicting targets become missing rather than being overwritten with another result. Raw-only mode preserves recorded dates and labels except invalid/missing target handling.
4. **Audit provenance without substituting quotes.** Source dates are parsed day-first, with Latin-1 fallback where necessary. Matching keys normalize case, accents and punctuation, with one explicit Tigres alias; display names stay intact. Source URL/hash and row-level evidence are exported. Complete `AvgCH/CD/CA`, then `AvgH/D/A`, then `BbAvH/D/A` source averages help identify supplemental games and audit quote provenance. They do not become model inputs. The old audit proved market-average rather than Bet365 matches for Premier League 2023/24; that sample does not establish one bookmaker or quote timing across every league/season.
5. **Build histories before modeling, with shifts.** Stable sort by prepared date/kickoff; missing kickoff times sort first within their date. Shift each team's games before joining, so current results/statistics/odds never enter its historical blocks. Earlier held-out games may become history for later predictions. Current statistics, metadata, precomputed form and other raw odds-derived fields are excluded from base inputs. Histories group supplied team names; other club renamings are not inferred.
6. **Learn preprocessing only from training rows.** Naive Bayes/logistic regression use median imputation, missingness indicators and standardization. Entirely missing training columns are retained with zero placeholders and indicators. Naive Bayes uses float64 probability calculations; boosting handles NaN natively and disables its random internal early-stopping split. Saved pipelines retain preprocessing.

For decimal odds `oH/oD/oA`, use **`pH = (1/oH) / (1/oH + 1/oD + 1/oA)`**, analogously for draw/away. Raw odds divided by their sum would reverse the meaning. The favourite is the minimum-odds outcome; ties follow home/draw/away priority. Proportional removal of overround does not guarantee calibration. The benchmark is labelled **Supplied odds**, without assuming verified closing-average timing. Accuracy is not betting profitability or evidence about causes of league popularity.

### Missing/invalid data in Matches.csv

| Check | Matches | Share |
| --- | --- | --- |
| Missing full-time result and goal counts | 4 | 0.0017% |
| Missing at least one Elo | 98,144 | 41.09% |
| Missing both Elo values | 79,845 | 33.43% |
| Missing at least one half-time goal count | 54,580 | 22.85% |
| Missing at least one detailed statistic | 117,953 | 49.38% |
| Missing all detailed statistics | 111,259 | 46.58% |
| Missing home/draw/away odds triplet | 3,042 | 1.27% |
| Complete but invalid odds triplet | 7 | 0.0029% |
| Missing kickoff time | 131,485 | 55.05% |
| Unparseable dates or duplicate match identities | 0 | 0% |

“Detailed statistics” means both teams' shots, shots on target, fouls, corners and yellow/red cards. Missingness differs sharply by league; counts are not disjoint. The seven invalid complete triplets have home odds at most one. Other unused price fields contain one invalid `MaxAway`, four invalid `HandiHome` and three invalid `HandiAway` values. No supplied H/D/A result disagrees with its supplied full-time goal counts. See the [full audit](reports/data_quality.md) and field/league CSVs for every column.

External verification finds **19,395 date changes (8.12%)**, **one result disagreement**, and **1,075 matches crossing the training boundary** (611 leave, 464 enter: raw training has 209,648 labeled rows, verified training 209,501). The large Elo/statistics gaps are **not** filled by this verification. External averages could recover 173 missing and seven invalid raw quote triplets, but 24,749 matches with usable supplied odds lack usable source averages; replacing all supplied odds would therefore reduce coverage. The primary flow preserves them. Exact model/feature effects are measured in [the raw-only comparison](reports/external_data_impact.md).

With model settings fixed, the common paired comparison contains **5,134 identical held-out matches**. External date/result cleaning changes pooled accuracy as follows (percentage points; cleaned minus raw):

| Classifier | Without odds features | With supplied-odds features |
| --- | --- | --- |
| Gaussian Naive Bayes | +0.12 pp | +0.16 pp |
| Logistic regression | +0.23 pp | −0.19 pp |
| Histogram gradient boosting | +0.37 pp | +0.06 pp |

Across individual tier/model combinations, the changes range from **−0.33 to +0.92 pp**. The lower-tier group remains less predictable. **54,369 matches (22.76%)** have at least one changed base history input. Verification primarily protects chronological validity; it does not guarantee higher accuracy. The paired cohort deliberately differs from the full primary cohort because both versions of a game's date must lie inside the common held-out window.

Elo is preserved as supplied, not independently reconstructed; values after June 2025 are described by the dataset as a provisional continuation. Optional verification improves chronology but does not make the whole dataset complete or eliminate all quality concerns. Small accuracy differences are descriptive, not significance claims. Use regenerated reports rather than copying draft outline numbers.

## 中文

分析按[提纲](../outline.pdf)进行：比较各联赛的可预测性，与赔率热门结果比较，再将赔率加入模型。保留高斯朴素贝叶斯、逻辑回归、直方图梯度提升及类别先验基线，移除神经网络和单独的 Bet365 实验。

**`data/Matches.csv` 是比赛主表。主分析直接使用其中的 `OddHome/OddDraw/OddAway`，保留原始缺失情况；外部赔率不会填补或替换这些值。** 外部 Football-Data 文件仅用于可选的日期、赛果核验及来源审计，并有明确的下载步骤。外部来源中额外的比赛不会加入主表。

### 为什么引入外部数据

最初的赔率审计使用 Football-Data 的联赛/赛季原始文件，核对名称较笼统的 `OddHome/OddDraw/OddAway` 究竟代表什么。将其与明确标注的单家和市场平均赔率比较后，发现被核验的英超 2023/24 报价对应 `AvgH/AvgD/AvgA`，而非 Bet365 的 `B365H/B365D/B365A`。这说明解释市场基准前有必要核对来源，但不能据此推断所有联赛、赛季都使用同一来源或同一报价时间。

这些来源文件还包含比赛日期和赛果，因此也可用于核验主表的时间顺序。月日互换可能生成语法完全有效的日期，普通日期解析无法发现；错误日期会改变“最近五场”的选择，也可能把比赛划到训练/测试边界的错误一侧。核验发现 19,395 个日期差异及 1,075 场跨越该边界的比赛，因此保留可选的外部清洗步骤有实际依据。

当前流程将外部文件用作核验证据，主要赔率输入仍取自主表；缺失赔率、Elo 和技术统计保持明确的缺失状态。第 00 步显式下载并记录文件哈希，`01_clean_data.py --raw-only` 支持不做外部核验的流程，第 09 步在相同留出比赛上衡量两种选择的影响，而不预设核验一定提高准确率。

### 运行及各步骤行为

使用 Python 3.12，在本目录运行：

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

第 00 步根据 `Matches.csv` 的联赛/赛季生成 URL，把公开的 [Football-Data](https://www.football-data.co.uk/) 文件下载至 **`data/football_data_sources/`**，在 `download_manifest.csv` 中记录 URL、SHA-256、大小、核验时间和下载/缓存状态。无需预先存在该目录；重复运行复用缓存，`--refresh` 强制重下载。下载失败会记录并停止。下载实现位于 [src/data_sources.py](src/data_sources.py)。本次已从空目录重新下载全部 **583 个文件**，不依赖旧 `bet365_sources`。公开 URL 内容可能变化；精确复现需保留快照和清单。安装依赖后，只有第 00 步需要联网。

**仅使用 `Matches.csv` 时**，跳过第 00 步，并将第 01 步改为：

```bash
.venv/bin/python 01_clean_data.py --raw-only
```

随后正常运行 02–08。此模式不做外部日期修正或来源冲突排除。第 09 步是额外的敏感性比较，需先完成外部核验流程；其中原始分支仅从 `Matches.csv` 重建历史并重新训练，不覆盖主分析。切换准备模式后需重跑下游步骤。生成文件会被对应步骤覆盖，原始 CSV 不变；`reports/data_preparation.json` 记录模式、输入哈希、改动和赔率来源。

| 编号 | 行为 | 主要输出 |
| --- | --- | --- |
| 00 | 显式下载公开核验文件，支持重试、缓存并记录来源。 | `data/football_data_sources/`、`download_manifest.csv` |
| 01 | 准备原始赔率；默认使用下载文件核验日期和赛果，`--raw-only` 可跳过。 | `data/MatchesChronological.csv`；`data_preparation.json`、`data_cleaning_*.csv` |
| 02 | 构建赛前 Elo 和双方各自最近五场比赛的独立历史。 | `data/MatchTraining.csv` |
| 03 | 在更早的验证集选参，使用全部训练联赛重新拟合并在英超测试。 | `models/*.joblib`；`model_evaluation.md`、`model_*.csv`、`validation_metrics.csv` |
| 04 | 使用固定模型比较五大联赛和三个层级的 15 个联赛。 | `top_five_evaluation.md`、`top_five_model_*.csv`、`league_tier_*` |
| 05 | 在同一批有效比赛上，比较分类器和原始赔率热门结果。 | `market_average_*`（历史文件名） |
| 06 | 从原始赔率增加本场 3 个及历史 30 个概率，并检查行对齐。 | `data/MarketTraining.csv`；`docs/market_training_fields.csv` |
| 07 | 按同一组已选参数、相同训练行拟合有/无赔率特征的模型。 | `models/market_augmented/`；`market_augmented_*` |
| 08 | 审计每列缺失、无效值、跨字段一致性及可用的外部差异。 | [数据质量报告](reports/data_quality.md)；`data_quality_*.csv` |
| 09 | 固定参数，在纯原始数据分支重建历史并重新拟合；两分支均使用相同原始赔率。 | [外部清洗影响报告](reports/external_data_impact.md)；`external_data_*`、`raw_only_predictions.csv`、`models/raw_only/` |

报告位于 `reports/`。编号文件保留分析步骤，`src/` 存放复用函数，`tests/` 验证关键行为。`EloRatings.csv`、`E1_Matches.csv`、`F1_Matches.csv` 作为来源材料保留，运行时不需要读取；模型使用主表已有 Elo。

### 特征和评估方法

主表包含 **238,858 场比赛、38 个联赛/级别代码、48 列**，覆盖 2000 年 7 月至 2026 年 9 月。比赛结果和统计来自 Football-Data，原始 Elo 基于 [ClubElo](https://www.clubelo.com/)。基础表有 **219 列：214 个数值输入、4 个元数据字段和 1 个目标**。`Division`、`MatchDate`、`HomeTeam`、`AwayTeam` 及目标 `FTResult`（H/D/A）不作为数值输入。

214 个特征包括当前双方 Elo、两个历史场次数量，以及双方各自最近五场比赛每场 21 个字段：间隔天数、当时主客场、球队/对手 Elo、赛果积分（3/1/0），以及双方进球、半场进球、射门、射正、犯规、角球、黄牌、红牌。`Previous1` 为最近一场，历史跨赛季和级别并合并主客场经历。详见[基础字段字典](docs/match_prediction_fields.csv)。

扩展表有 **252 列、247 个数值输入**。新增本场 `MarketHomeProbability/MarketDrawProbability/MarketAwayProbability`，以及 `{Home|Away}_Previous{1..5}_Market{Win|Draw|Loss}Probability`，合计 33 个概率。历史胜平负按该球队当时主客场转换视角。详见[扩展字段字典](docs/market_training_fields.csv)。

- 训练：所有可用联赛，**2000-07-01 至 2023-06-30**。
- 选参：2021 年 7 月之前拟合候选模型，在 2021 年 7 月至 2023 年 6 月英超上按对数损失选参，再用完整训练窗口拟合；不使用测试集调参。
- 测试：**2023-07-01 至 2026-06-30**，受实际覆盖限制；自然年联赛使用相同边界。五大联赛另有分赛季结果。
- 层级主比较截止于 15 个联赛各自最后一场比赛中的最早日期：核验日期为 **2024-10-06**，原始记录日期为 **2024-12-05**。`Available` 保留各联赛全部可用测试日期，另输出自然年检查。
- 赔率实验复制已选设置，重新拟合预处理和分类器；两个特征版本训练行相同，均保留缺赔率比赛。与赔率基准配对测试要求三项赔率全部有限且大于 1。
- 第 09 步固定模型参数和原始赔率，主要比较限定同一比赛在**原始日期和来源日期下都属于留出测试期**且赛果一致，另外报告两分支各自自然窗口的结果；不重新搜索参数。

Top 为英格兰 E0、西班牙 SP1、德国 D1、意大利 I1、法国 F1（2023 UEFA 排名 1–5）；Mid 为荷兰 N1、葡萄牙 P1、比利时 B1、苏格兰 SC0、奥地利 AUT（6–10）；Lower 为瑞典 SWE、波兰 POL、罗马尼亚 ROM、爱尔兰 IRL、芬兰 FIN（23、24、26、36、37）。依据见[分组说明](docs/league_tier_selection.md)及[排名表](docs/league_strength_2023.csv)。排名是外部研究设计元数据，不是下载的比赛输入特征；Lower 仅相对于可用欧洲顶级联赛。按需报告准确率、平衡准确率、宏 F1、对数损失、平局召回、比赛合并/联赛等权平均及覆盖率。导出概率按 A/D/H 排列，赔率输入为 H/D/A。

### 清洗方法及外部核验的理由

1. **明确保留缺失。**未知技术统计、Elo 和不足五场历史保持 `NaN`，真实零值保留。无效或缺失的 H/D/A 目标不参与监督训练。重复比赛身份会使程序停止，不静默去重。审计区分缺失与非缺失但无效的观测值，并标记跨字段矛盾。
2. **校验原始赔率。**三项必须全部有限且大于 1；任一无效时三个衍生概率均置为缺失，原赔率列不改动。不下载报价补缺。缺赔率比赛仍可在适当预处理下用于训练，但不参加与赔率基准的配对评估。
3. **可选的日期、赛果核验。**下载的主要理由是部分 ISO 格式日期虽然语法有效，却存在月日互换，会改变训练划分和历史。匹配联赛、双方球队及原/互换日期候选；补充联赛还要求三项原赔率在 `0.0050001` 绝对误差内一致。若全部原赔率缺失，则要求双方全场进球及赛果一致。拒绝多义或缺失匹配。另明确修正 Santa Clara–Pacos Ferreira（2020-02-01 → 2020-02-02）及 Santa Clara–Benfica（2021-01-03 → 2021-01-04）。只修改生成副本中的日期，冲突目标置缺失，不直接改写为另一个赛果。纯原始模式保留日期和有效标签。
4. **核对来源，但不替换报价。**来源按日/月/年解析，必要时用 Latin-1。匹配键忽略大小写、重音、标点，并处理一条明确的 Tigres 别名；展示名称保留。记录文件 URL/哈希和逐场证据。来源中完整的 `AvgCH/CD/CA`、`AvgH/D/A`、`BbAvH/D/A` 按顺序用于补充比赛身份匹配和报价来源审计，不作为模型输入。旧审计在英超 2023/24 中发现原赔率对应市场平均值而非 Bet365；不能由此推断所有联赛/赛季都使用同一家或同一报价时间。
5. **先移位再加入历史。**按准备后的日期和开球时间稳定排序，缺时间排在当日最前。球队历史先移位后关联，避免本场赛果、统计和赔率进入本场历史。较早测试比赛可成为较晚比赛的历史。本场统计、元数据、预计算状态及其他赔率衍生列不作为基础模型输入。历史按原球队名分组，不推断其他更名。
6. **预处理只在训练行学习。**朴素贝叶斯/逻辑回归使用中位数、缺失指示和标准化；整列缺失时保留零占位及缺失指示。朴素贝叶斯使用 float64 概率计算；提升树原生处理 NaN 并关闭内部随机早停划分。预处理随模型保存。

十进制赔率 `oH/oD/oA` 转换为 **`pH = (1/oH) / (1/oH + 1/oD + 1/oA)`**，平/客同理。直接用原赔率除以其和会颠倒含义。赔率最低者为热门，并列按主/平/客固定顺序处理。按比例去除超额概率不保证校准。基准标为 **Supplied odds（原始赔率）**，不假设已经核验为市场平均临场报价。准确率不代表投注盈利或联赛受欢迎程度的原因。

### 主表缺失和无效值

| 检查项 | 比赛数 | 比例 |
| --- | --- | --- |
| 全场赛果及进球缺失 | 4 | 0.0017% |
| 至少一方 Elo 缺失 | 98,144 | 41.09% |
| 双方 Elo 均缺失 | 79,845 | 33.43% |
| 至少一方半场进球缺失 | 54,580 | 22.85% |
| 至少一项详细技术统计缺失 | 117,953 | 49.38% |
| 全部详细技术统计缺失 | 111,259 | 46.58% |
| 主/平/客赔率缺失 | 3,042 | 1.27% |
| 赔率完整但无效 | 7 | 0.0029% |
| 开球时间缺失 | 131,485 | 55.05% |
| 日期无法解析或比赛身份重复 | 0 | 0% |

详细统计指双方射门、射正、犯规、角球、黄红牌；各项缺失可重叠，联赛间差异很大。7 场无效完整报价的主胜赔率不大于 1。未用于主分析的其他赔率列另有 `MaxAway` 1 个、`HandiHome` 4 个、`HandiAway` 3 个无效值。所有已有 H/D/A 标签均与原表全场进球一致。逐列、逐联赛详情见[完整审计](reports/data_quality.md)及 CSV。

外部核验发现 **19,395 个日期差异（8.12%）**、**1 个赛果冲突**，以及 **1,075 场比赛跨越训练边界**（611 场移出、464 场移入；原始有标签训练行 209,648，核验后 209,501）。外部核验**不会**补齐大面积缺失的 Elo 或技术统计。外部平均赔率可恢复 173 个缺失和 7 个无效原始报价，但有 24,749 场具有可用原赔率的比赛缺少有效来源平均报价，全面替换会降低覆盖。主流程保留原赔率；具体历史特征及模型变化见[纯原始数据比较](reports/external_data_impact.md)。

固定模型参数后，共同配对比较包含 **5,134 场完全相同的留出比赛**。外部日期/赛果清洗带来的合并准确率变化如下（清洗后减原始，单位为百分点）：

| 分类器 | 无赔率特征 | 含原始赔率特征 |
| --- | --- | --- |
| 高斯朴素贝叶斯 | +0.12 | +0.16 |
| 逻辑回归 | +0.23 | −0.19 |
| 直方图梯度提升 | +0.37 | +0.06 |

逐层级/模型的变化范围为 **−0.33 至 +0.92 个百分点**，低层级仍较难预测。共有 **54,369 场比赛（22.76%）** 的至少一个基础历史输入发生变化。核验主要保障时间顺序有效性，不保证提高准确率。配对样本要求原始日期和核验日期都位于共同留出窗口，因此与主报告完整测试样本数量不同。

Elo 按主表保留，不独立重算；数据集将 2025 年 6 月以后的值描述为暂定延续。可选核验改善时间顺序，但不意味着数据完整或所有质量问题都已解决。小幅准确率差异不构成显著性结论；写作请使用重新生成的报告，而非照抄提纲草稿。

检查命令 / Verification:

```bash
.venv/bin/python -W error -m unittest discover -s tests -v
```
