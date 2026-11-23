# Match outcome classification

Training: 209,501 matches across all available leagues, July 2000–June 2023. Premier League test: 1,140 matches, July 2023–June 2026. Parameters minimize log loss on earlier Premier League validation (July 2021–June 2023); candidate fitting uses only dates before July 2021. Preprocessing is fitted on training rows. All histories use source-corrected dates. Scores are descriptive; see README for limitations.

## Model choices

| Model | Reason |
| --- | --- |
| Prior baseline | Always predicts the most frequent training result; probabilities are training class frequencies. Establishes a minimum benchmark. |
| Gaussian Naive Bayes | A fast Bayesian classification baseline for numeric inputs. Its conditional-independence and Gaussian assumptions are imperfect for correlated match statistics; variance smoothing limits overconfidence. |
| Logistic regression | A regularized, interpretable linear multiclass model. Suitable for many correlated numerical inputs; scaling and regularization control the influence of their different units. |
| Histogram gradient boosting | Boosted trees capture nonlinear relationships and interactions between Elo, venue, and individual previous games. Histogram binning scales to this dataset and the model handles missing statistics directly. |

## Earlier validation

| Model | Parameters | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | Selected |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | {} | 0.4566 | 0.3333 | 0.2090 | 1.0635 | True |
| Gaussian Naive Bayes | {'var_smoothing': 0.01} | 0.4987 | 0.4525 | 0.3762 | 5.7214 | False |
| Gaussian Naive Bayes | {'var_smoothing': 1.0} | 0.5145 | 0.4609 | 0.3881 | 2.1532 | False |
| Gaussian Naive Bayes | {'var_smoothing': 10.0} | 0.5263 | 0.4344 | 0.3792 | 0.9937 | True |
| Logistic regression | {'C': 0.1} | 0.5645 | 0.4746 | 0.4179 | 0.9650 | False |
| Logistic regression | {'C': 1.0} | 0.5645 | 0.4746 | 0.4179 | 0.9649 | True |
| Histogram gradient boosting | {'max_leaf_nodes': 15, 'l2_regularization': 1.0} | 0.5408 | 0.4486 | 0.3957 | 0.9719 | False |
| Histogram gradient boosting | {'max_leaf_nodes': 31, 'l2_regularization': 10.0} | 0.5447 | 0.4554 | 0.4007 | 0.9687 | True |

## Premier League scores

| Model | Season | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss |
| --- | --- | --- | --- | --- | --- | --- |
| Prior baseline | All | 1140 | 0.4316 | 0.3333 | 0.2010 | 1.0755 |
| Prior baseline | 2023/24 | 380 | 0.4605 | 0.3333 | 0.2102 | 1.0604 |
| Prior baseline | 2024/25 | 380 | 0.4079 | 0.3333 | 0.1931 | 1.0860 |
| Prior baseline | 2025/26 | 380 | 0.4263 | 0.3333 | 0.1993 | 1.0802 |
| Gaussian Naive Bayes | All | 1140 | 0.5263 | 0.4494 | 0.3890 | 1.0182 |
| Gaussian Naive Bayes | 2023/24 | 380 | 0.5763 | 0.4727 | 0.4178 | 0.9645 |
| Gaussian Naive Bayes | 2024/25 | 380 | 0.5237 | 0.4546 | 0.3935 | 1.0355 |
| Gaussian Naive Bayes | 2025/26 | 380 | 0.4789 | 0.4178 | 0.3527 | 1.0544 |
| Logistic regression | All | 1140 | 0.5263 | 0.4510 | 0.3900 | 0.9901 |
| Logistic regression | 2023/24 | 380 | 0.5711 | 0.4697 | 0.4156 | 0.9273 |
| Logistic regression | 2024/25 | 380 | 0.5132 | 0.4456 | 0.3841 | 1.0107 |
| Logistic regression | 2025/26 | 380 | 0.4947 | 0.4362 | 0.3688 | 1.0325 |
| Histogram gradient boosting | All | 1140 | 0.5272 | 0.4511 | 0.3909 | 0.9889 |
| Histogram gradient boosting | 2023/24 | 380 | 0.5842 | 0.4830 | 0.4324 | 0.9299 |
| Histogram gradient boosting | 2024/25 | 380 | 0.5184 | 0.4499 | 0.3870 | 0.9995 |
| Histogram gradient boosting | 2025/26 | 380 | 0.4789 | 0.4178 | 0.3512 | 1.0372 |

## Prior baseline: class scores

| Result | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| A | 0.0000 | 0.0000 | 0.0000 | 369.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 279.0000 |
| H | 0.4316 | 1.0000 | 0.6029 | 492.0000 |

## Prior baseline: confusion matrix (actual rows)

| Actual | A | D | H |
| --- | --- | --- | --- |
| A | 0 | 0 | 369 |
| D | 0 | 0 | 279 |
| H | 0 | 0 | 492 |

## Gaussian Naive Bayes: class scores

| Result | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| A | 0.5322 | 0.5149 | 0.5234 | 369.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 279.0000 |
| H | 0.5243 | 0.8333 | 0.6436 | 492.0000 |

## Gaussian Naive Bayes: confusion matrix (actual rows)

| Actual | A | D | H |
| --- | --- | --- | --- |
| A | 190 | 0 | 179 |
| D | 86 | 0 | 193 |
| H | 81 | 1 | 410 |

## Logistic regression: class scores

| Result | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| A | 0.5157 | 0.5339 | 0.5246 | 369.0000 |
| D | 0.0000 | 0.0000 | 0.0000 | 279.0000 |
| H | 0.5324 | 0.8191 | 0.6453 | 492.0000 |

## Logistic regression: confusion matrix (actual rows)

| Actual | A | D | H |
| --- | --- | --- | --- |
| A | 197 | 1 | 171 |
| D | 96 | 0 | 183 |
| H | 89 | 0 | 403 |

## Histogram gradient boosting: class scores

| Result | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| A | 0.5093 | 0.5203 | 0.5147 | 369.0000 |
| D | 1.0000 | 0.0036 | 0.0071 | 279.0000 |
| H | 0.5354 | 0.8293 | 0.6507 | 492.0000 |

## Histogram gradient boosting: confusion matrix (actual rows)

| Actual | A | D | H |
| --- | --- | --- | --- |
| A | 192 | 0 | 177 |
| D | 101 | 1 | 177 |
| H | 84 | 0 | 408 |
