# Effect of external date/result cleaning / 外部清洗的影响

The raw branch rebuilds histories and refits all classifiers using Matches.csv alone. Both branches use its original odds. Selected hyperparameters are held fixed; this is not a new hyperparameter search. Primary scores use identical games whose raw AND source dates are held out between July 2023 and 2024-10-06, with agreeing results and valid supplied odds. The raw native common cutoff is 2024-12-05, so native common scores use different cohorts and must not be interpreted as a paired accuracy change. 54,369 matches (22.76%) have at least one changed base feature after date/result cleaning. Elo and detailed statistics remain as supplied; missing values are not externally filled. 原始分支仅使用 Matches.csv 重建历史并重新拟合；两分支使用相同原始赔率和固定模型参数。主要比较要求同一比赛在原始日期及来源日期下均属于测试期，并且赛果一致。因此配对表与各自自然窗口的结果需分别解读。

## Paired accuracy changes / 配对准确率变化

| FeatureSet | Group | Model | Matches_Raw | Accuracy_Raw | Accuracy_Cleaned | AccuracyChangePP |
| --- | --- | --- | --- | --- | --- | --- |
| No odds | All | Supplied odds | 5134 | 0.5312 | 0.5312 | 0.0000 |
| No odds | All | Prior baseline | 5134 | 0.4334 | 0.4334 | 0.0000 |
| No odds | All | Gaussian Naive Bayes | 5134 | 0.4988 | 0.5000 | 0.1169 |
| No odds | All | Logistic regression | 5134 | 0.5123 | 0.5146 | 0.2337 |
| No odds | All | Histogram gradient boosting | 5134 | 0.5095 | 0.5132 | 0.3701 |
| No odds | Lower | Supplied odds | 1413 | 0.4848 | 0.4848 | 0.0000 |
| No odds | Lower | Prior baseline | 1413 | 0.4374 | 0.4374 | 0.0000 |
| No odds | Lower | Gaussian Naive Bayes | 1413 | 0.4388 | 0.4409 | 0.2123 |
| No odds | Lower | Logistic regression | 1413 | 0.4558 | 0.4572 | 0.1415 |
| No odds | Lower | Histogram gradient boosting | 1413 | 0.4444 | 0.4536 | 0.9200 |
| No odds | Mid | Supplied odds | 1622 | 0.5493 | 0.5493 | 0.0000 |
| No odds | Mid | Prior baseline | 1622 | 0.4346 | 0.4346 | 0.0000 |
| No odds | Mid | Gaussian Naive Bayes | 1622 | 0.5173 | 0.5210 | 0.3699 |
| No odds | Mid | Logistic regression | 1622 | 0.5370 | 0.5444 | 0.7398 |
| No odds | Mid | Histogram gradient boosting | 1622 | 0.5425 | 0.5506 | 0.8015 |
| No odds | Top | Supplied odds | 2099 | 0.5484 | 0.5484 | 0.0000 |
| No odds | Top | Prior baseline | 2099 | 0.4297 | 0.4297 | 0.0000 |
| No odds | Top | Gaussian Naive Bayes | 2099 | 0.5250 | 0.5236 | -0.1429 |
| No odds | Top | Logistic regression | 2099 | 0.5312 | 0.5303 | -0.0953 |
| No odds | Top | Histogram gradient boosting | 2099 | 0.5279 | 0.5245 | -0.3335 |
| With market odds | All | Supplied odds | 5134 | 0.5312 | 0.5312 | 0.0000 |
| With market odds | All | Prior baseline | 5134 | 0.4334 | 0.4334 | 0.0000 |
| With market odds | All | Gaussian Naive Bayes | 5134 | 0.5078 | 0.5093 | 0.1558 |
| With market odds | All | Logistic regression | 5134 | 0.5323 | 0.5304 | -0.1948 |
| With market odds | All | Histogram gradient boosting | 5134 | 0.5323 | 0.5329 | 0.0584 |
| With market odds | Lower | Supplied odds | 1413 | 0.4848 | 0.4848 | 0.0000 |
| With market odds | Lower | Prior baseline | 1413 | 0.4374 | 0.4374 | 0.0000 |
| With market odds | Lower | Gaussian Naive Bayes | 1413 | 0.4544 | 0.4544 | 0.0000 |
| With market odds | Lower | Logistic regression | 1413 | 0.4890 | 0.4883 | -0.0708 |
| With market odds | Lower | Histogram gradient boosting | 1413 | 0.4862 | 0.4869 | 0.0708 |
| With market odds | Mid | Supplied odds | 1622 | 0.5493 | 0.5493 | 0.0000 |
| With market odds | Mid | Prior baseline | 1622 | 0.4346 | 0.4346 | 0.0000 |
| With market odds | Mid | Gaussian Naive Bayes | 1622 | 0.5265 | 0.5277 | 0.1233 |
| With market odds | Mid | Logistic regression | 1622 | 0.5518 | 0.5499 | -0.1850 |
| With market odds | Mid | Histogram gradient boosting | 1622 | 0.5536 | 0.5549 | 0.1233 |
| With market odds | Top | Supplied odds | 2099 | 0.5484 | 0.5484 | 0.0000 |
| With market odds | Top | Prior baseline | 2099 | 0.4297 | 0.4297 | 0.0000 |
| With market odds | Top | Gaussian Naive Bayes | 2099 | 0.5293 | 0.5322 | 0.2859 |
| With market odds | Top | Logistic regression | 2099 | 0.5465 | 0.5436 | -0.2859 |
| With market odds | Top | Histogram gradient boosting | 2099 | 0.5469 | 0.5469 | 0.0000 |

## Raw-only run settings / 运行参数

| Setting | Value |
| --- | --- |
| RawInputSHA256 | ef224cf2c252f07a842b3bcfd4ba5c718c25cedd8937ffa174a74b86b5ba4221 |
| Protocol | Fixed selected settings; raw-only refitting without retuning; supplied odds in both branches |
| RawTrainingMatches | 209648 |
| RawTestMatchesWithOdds | 10549 |
| RawCommonEnd | 2024-12-05 |
| SourceCommonEnd | 2024-10-06 |
| MatchesWithChangedBaseFeatures | 54369 |
| MatchesWithChangedBaseFeaturesPercent | 22.7621 |

## Raw-only fitting time / 拟合耗时

| FeatureSet | Model | Seconds |
| --- | --- | --- |
| No odds | Prior baseline | 0.2162 |
| No odds | Gaussian Naive Bayes | 8.1899 |
| No odds | Logistic regression | 72.7230 |
| No odds | Histogram gradient boosting | 76.9222 |
| With market odds | Prior baseline | 0.2021 |
| With market odds | Gaussian Naive Bayes | 9.7202 |
| With market odds | Logistic regression | 72.5794 |
| With market odds | Histogram gradient boosting | 83.5541 |
