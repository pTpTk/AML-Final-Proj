# Adding market probabilities

Both feature sets use the same 209,501 labeled training rows before July 2023. The selected settings from step 03 are cloned and fitted afresh, including preprocessing. No test-set tuning. Primary test window ends 2024-10-06; every comparison uses identical eligible rows. The control uses 214 numeric features; the augmented models use 247. Market features use normalized inverse decimal odds. Historical quotes follow the team's venue and are shifted before joining. All quotes are supplied OddHome/OddDraw/OddAway values from Matches.csv; missing quotes stay missing. Results are descriptive and can differ from the draft outline after date preparation and the June 2026 cutoff. Detailed league/year metrics are in market_augmented_metrics.csv.

## Tier accuracy

| Tier | Model | Matches | Model without market odds | Market favourite | Model with market odds | Change (pp) |
| --- | --- | --- | --- | --- | --- | --- |
| Top | Prior baseline | 2099 | 42.97% | 54.84% | 42.97% | +0.00 |
| Top | Gaussian Naive Bayes | 2099 | 52.36% | 54.84% | 53.22% | +0.86 |
| Top | Logistic regression | 2099 | 53.03% | 54.84% | 54.36% | +1.33 |
| Top | Histogram gradient boosting | 2099 | 52.45% | 54.84% | 54.69% | +2.24 |
| Mid | Prior baseline | 1664 | 43.39% | 54.99% | 43.39% | +0.00 |
| Mid | Gaussian Naive Bayes | 1664 | 51.86% | 54.99% | 52.76% | +0.90 |
| Mid | Logistic regression | 1664 | 54.27% | 54.99% | 54.99% | +0.72 |
| Mid | Histogram gradient boosting | 1664 | 54.81% | 54.99% | 55.53% | +0.72 |
| Lower | Prior baseline | 1644 | 43.92% | 48.42% | 43.92% | +0.00 |
| Lower | Gaussian Naive Bayes | 1644 | 44.16% | 48.42% | 45.62% | +1.46 |
| Lower | Logistic regression | 1644 | 45.44% | 48.42% | 48.66% | +3.22 |
| Lower | Histogram gradient boosting | 1644 | 45.13% | 48.42% | 48.78% | +3.65 |

## Pooled accuracy changes

| Group | Model | Matches | NoOdds | WithOdds | AccuracyChangePP |
| --- | --- | --- | --- | --- | --- |
| All | Prior baseline | 5407 | 0.4339 | 0.4339 | 0.0000 |
| All | Gaussian Naive Bayes | 5407 | 0.4971 | 0.5077 | 1.0542 |
| All | Logistic regression | 5407 | 0.5110 | 0.5282 | 1.7200 |
| All | Histogram gradient boosting | 5407 | 0.5095 | 0.5315 | 2.2009 |

## Training runs

| FeatureSet | Model | TrainingMatches | Features | Seconds |
| --- | --- | --- | --- | --- |
| No odds | Prior baseline | 209501 | 214 | 0.2255 |
| No odds | Gaussian Naive Bayes | 209501 | 214 | 8.4086 |
| No odds | Logistic regression | 209501 | 214 | 66.6999 |
| No odds | Histogram gradient boosting | 209501 | 214 | 74.2310 |
| With market odds | Prior baseline | 209501 | 247 | 0.2037 |
| With market odds | Gaussian Naive Bayes | 209501 | 247 | 9.9331 |
| With market odds | Logistic regression | 209501 | 247 | 80.3259 |
| With market odds | Histogram gradient boosting | 209501 | 247 | 81.8476 |
