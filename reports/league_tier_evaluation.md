# Predictability across 15 leagues

Primary comparison: July 2023 through 2024-10-06, the earliest last match among all 15 leagues. Secondary Available results stop by June 2026. Tiers use published 2023 UEFA ranks. Coverage differences and outcome frequencies can affect accuracy; observed gaps are not significance or popularity claims. Full results, including calendar-year checks, are in league_tier_metrics.csv. Preparation: Source-verified dates/results; supplied odds. Odds inputs: Matches.csv: OddHome/OddDraw/OddAway. Dates changed: 19,395. Missing/invalid targets: 5.

## Tier accuracy

| Tier | Matches | Prior baseline | Gaussian Naive Bayes | Logistic regression | Histogram gradient boosting |
| --- | --- | --- | --- | --- | --- |
| Top | 2099 | 0.4297 | 0.5236 | 0.5303 | 0.5245 |
| Mid | 1664 | 0.4339 | 0.5186 | 0.5427 | 0.5481 |
| Lower | 1644 | 0.4392 | 0.4416 | 0.4544 | 0.4513 |

## Coverage

| Window | Division | Matches | FirstMatch | LastMatch | TrainingMatches | CurrentEloMissing | DetailedStatsMissing | AllFeaturesMissing | CompleteFiveGameHistory | HomeWinRate | DrawRate | AwayWinRate | Country | League | UEFARank2023 | Coefficient2023 | Tier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common | AUT | 247 | 2023-07-28 | 2024-10-06 | 2059 | 0.5121 | 1.0000 | 0.7106 | 0.9595 | 0.3846 | 0.2915 | 0.3239 | Austria | Bundesliga | 10 | 34.0000 | Mid |
| Common | B1 | 392 | 2023-07-28 | 2024-10-06 | 5935 | 0.0459 | 0.0094 | 0.0115 | 0.9872 | 0.4490 | 0.2653 | 0.2857 | Belgium | Pro League | 8 | 42.2000 | Mid |
| Common | D1 | 360 | 2023-08-18 | 2024-10-06 | 6910 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4278 | 0.2583 | 0.3139 | Germany | Bundesliga | 3 | 82.4810 | Top |
| Common | E0 | 450 | 2023-08-11 | 2024-10-06 | 8650 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4489 | 0.2289 | 0.3222 | England | Premier League | 1 | 109.5700 | Top |
| Common | F1 | 369 | 2023-08-11 | 2024-10-06 | 8145 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4065 | 0.2629 | 0.3306 | France | Ligue 1 | 5 | 61.1640 | Top |
| Common | FIN | 248 | 2023-07-01 | 2024-10-06 | 2106 | 0.8831 | 1.0000 | 0.7474 | 0.9597 | 0.4476 | 0.2339 | 0.3185 | Finland | Veikkausliiga | 37 | 10.2500 | Lower |
| Common | I1 | 450 | 2023-08-19 | 2024-10-06 | 8252 | 0.0000 | 0.0000 | 0.0001 | 1.0000 | 0.4178 | 0.3022 | 0.2800 | Italy | Serie A | 4 | 81.9260 | Top |
| Common | IRL | 228 | 2023-07-07 | 2024-10-06 | 2135 | 1.0000 | 1.0000 | 0.7570 | 1.0000 | 0.4430 | 0.2851 | 0.2719 | Ireland | Premier Division | 36 | 10.3750 | Lower |
| Common | N1 | 376 | 2023-08-11 | 2024-10-06 | 6676 | 0.0000 | 0.0040 | 0.0039 | 0.9867 | 0.4388 | 0.2420 | 0.3191 | Netherlands | Eredivisie | 6 | 59.9000 | Mid |
| Common | P1 | 377 | 2023-08-11 | 2024-10-06 | 6014 | 0.0557 | 0.0080 | 0.0131 | 0.9735 | 0.4483 | 0.2440 | 0.3077 | Portugal | Primeira Liga | 7 | 56.2160 | Mid |
| Common | POL | 403 | 2023-07-21 | 2024-10-06 | 3164 | 0.6563 | 1.0000 | 0.7258 | 0.9529 | 0.4417 | 0.2878 | 0.2705 | Poland | Ekstraklasa | 24 | 20.7500 | Lower |
| Common | ROM | 415 | 2023-07-14 | 2024-10-06 | 3225 | 0.8494 | 1.0000 | 0.7432 | 0.9759 | 0.4169 | 0.3205 | 0.2627 | Romania | Liga I | 26 | 20.5000 | Lower |
| Common | SC0 | 272 | 2023-08-05 | 2024-10-06 | 4696 | 0.0000 | 0.0000 | 0.0005 | 1.0000 | 0.4301 | 0.2647 | 0.3051 | Scotland | Premiership | 9 | 36.4000 | Mid |
| Common | SP1 | 470 | 2023-08-11 | 2024-10-06 | 8248 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4426 | 0.2851 | 0.2723 | Spain | La Liga | 2 | 92.9980 | Top |
| Common | SWE | 350 | 2023-07-01 | 2024-10-06 | 2760 | 0.4386 | 1.0000 | 0.7003 | 0.9857 | 0.4543 | 0.2000 | 0.3457 | Sweden | Allsvenskan | 23 | 23.7500 | Lower |
| Available | AUT | 247 | 2023-07-28 | 2024-10-06 | 2059 | 0.5121 | 1.0000 | 0.7106 | 0.9595 | 0.3846 | 0.2915 | 0.3239 | Austria | Bundesliga | 10 | 34.0000 | Mid |
| Available | B1 | 935 | 2023-07-28 | 2026-05-24 | 5935 | 0.0193 | 0.0064 | 0.0070 | 0.9893 | 0.4299 | 0.2674 | 0.3027 | Belgium | Pro League | 8 | 42.2000 | Mid |
| Available | D1 | 917 | 2023-08-18 | 2026-05-16 | 6910 | 0.0000 | 0.0016 | 0.0017 | 0.9945 | 0.4209 | 0.2541 | 0.3250 | Germany | Bundesliga | 3 | 82.4810 | Top |
| Available | E0 | 1140 | 2023-08-11 | 2026-05-24 | 8650 | 0.0009 | 0.0013 | 0.0014 | 0.9956 | 0.4316 | 0.2447 | 0.3237 | England | Premier League | 1 | 109.5700 | Top |
| Available | F1 | 917 | 2023-08-11 | 2026-05-17 | 8145 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4406 | 0.2377 | 0.3217 | France | Ligue 1 | 5 | 61.1640 | Top |
| Available | FIN | 248 | 2023-07-01 | 2024-10-06 | 2106 | 0.8831 | 1.0000 | 0.7474 | 0.9597 | 0.4476 | 0.2339 | 0.3185 | Finland | Veikkausliiga | 37 | 10.2500 | Lower |
| Available | I1 | 1140 | 2023-08-19 | 2026-05-24 | 8252 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4018 | 0.2798 | 0.3184 | Italy | Serie A | 4 | 81.9260 | Top |
| Available | IRL | 232 | 2023-07-07 | 2024-10-14 | 2135 | 1.0000 | 1.0000 | 0.7570 | 1.0000 | 0.4353 | 0.2931 | 0.2716 | Ireland | Premier Division | 36 | 10.3750 | Lower |
| Available | N1 | 918 | 2023-08-11 | 2026-05-17 | 6676 | 0.0000 | 0.0033 | 0.0032 | 0.9891 | 0.4434 | 0.2538 | 0.3028 | Netherlands | Eredivisie | 6 | 59.9000 | Mid |
| Available | P1 | 918 | 2023-08-11 | 2026-05-16 | 6014 | 0.0370 | 0.0049 | 0.0079 | 0.9891 | 0.4259 | 0.2582 | 0.3159 | Portugal | Primeira Liga | 7 | 56.2160 | Mid |
| Available | POL | 403 | 2023-07-21 | 2024-10-06 | 3164 | 0.6563 | 1.0000 | 0.7258 | 0.9529 | 0.4417 | 0.2878 | 0.2705 | Poland | Ekstraklasa | 24 | 20.7500 | Lower |
| Available | ROM | 415 | 2023-07-14 | 2024-10-06 | 3225 | 0.8494 | 1.0000 | 0.7432 | 0.9759 | 0.4169 | 0.3205 | 0.2627 | Romania | Liga I | 26 | 20.5000 | Lower |
| Available | SC0 | 656 | 2023-08-05 | 2026-05-17 | 4696 | 0.0000 | 0.0000 | 0.0005 | 1.0000 | 0.4466 | 0.2530 | 0.3003 | Scotland | Premiership | 9 | 36.4000 | Mid |
| Available | SP1 | 1140 | 2023-08-11 | 2026-05-24 | 8248 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4579 | 0.2605 | 0.2816 | Spain | La Liga | 2 | 92.9980 | Top |
| Available | SWE | 350 | 2023-07-01 | 2024-10-06 | 2760 | 0.4386 | 1.0000 | 0.7003 | 0.9857 | 0.4543 | 0.2000 | 0.3457 | Sweden | Allsvenskan | 23 | 23.7500 | Lower |

## League scores in the common window

| Window | Period | Scope | Tier | Division | Model | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | DrawRecall | MeanLeagueAccuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common | All | League | Top | D1 | Prior baseline | 360 | 0.4278 | 0.3333 | 0.1997 | 1.0783 | 0.0000 | nan |
| Common | All | League | Top | D1 | Gaussian Naive Bayes | 360 | 0.5083 | 0.4424 | 0.3797 | 1.0077 | 0.0000 | nan |
| Common | All | League | Top | D1 | Logistic regression | 360 | 0.5222 | 0.4564 | 0.3920 | 0.9703 | 0.0000 | nan |
| Common | All | League | Top | D1 | Histogram gradient boosting | 360 | 0.5139 | 0.4482 | 0.3897 | 0.9756 | 0.0108 | nan |
| Common | All | League | Top | E0 | Prior baseline | 450 | 0.4489 | 0.3333 | 0.2065 | 1.0666 | 0.0000 | nan |
| Common | All | League | Top | E0 | Gaussian Naive Bayes | 450 | 0.5644 | 0.4730 | 0.4148 | 0.9722 | 0.0000 | nan |
| Common | All | League | Top | E0 | Logistic regression | 450 | 0.5644 | 0.4743 | 0.4158 | 0.9277 | 0.0000 | nan |
| Common | All | League | Top | E0 | Histogram gradient boosting | 450 | 0.5756 | 0.4854 | 0.4298 | 0.9300 | 0.0097 | nan |
| Common | All | League | Top | F1 | Prior baseline | 369 | 0.4065 | 0.3333 | 0.1927 | 1.0880 | 0.0000 | nan |
| Common | All | League | Top | F1 | Gaussian Naive Bayes | 369 | 0.4715 | 0.4127 | 0.3478 | 1.0517 | 0.0000 | nan |
| Common | All | League | Top | F1 | Logistic regression | 369 | 0.4851 | 0.4279 | 0.3651 | 1.0364 | 0.0000 | nan |
| Common | All | League | Top | F1 | Histogram gradient boosting | 369 | 0.4743 | 0.4169 | 0.3534 | 1.0283 | 0.0000 | nan |
| Common | All | League | Top | I1 | Prior baseline | 450 | 0.4178 | 0.3333 | 0.1964 | 1.0863 | 0.0000 | nan |
| Common | All | League | Top | I1 | Gaussian Naive Bayes | 450 | 0.5178 | 0.4742 | 0.3959 | 1.0411 | 0.0000 | nan |
| Common | All | League | Top | I1 | Logistic regression | 450 | 0.5200 | 0.4810 | 0.4025 | 0.9903 | 0.0074 | nan |
| Common | All | League | Top | I1 | Histogram gradient boosting | 450 | 0.5178 | 0.4799 | 0.4072 | 0.9890 | 0.0147 | nan |
| Common | All | League | Top | SP1 | Prior baseline | 470 | 0.4426 | 0.3333 | 0.2045 | 1.0739 | 0.0000 | nan |
| Common | All | League | Top | SP1 | Gaussian Naive Bayes | 470 | 0.5426 | 0.4748 | 0.4026 | 1.0209 | 0.0000 | nan |
| Common | All | League | Top | SP1 | Logistic regression | 470 | 0.5489 | 0.4906 | 0.4141 | 0.9516 | 0.0000 | nan |
| Common | All | League | Top | SP1 | Histogram gradient boosting | 470 | 0.5298 | 0.4671 | 0.3960 | 0.9563 | 0.0000 | nan |
| Common | All | League | Mid | AUT | Prior baseline | 247 | 0.3846 | 0.3333 | 0.1852 | 1.1000 | 0.0000 | nan |
| Common | All | League | Mid | AUT | Gaussian Naive Bayes | 247 | 0.3927 | 0.3426 | 0.2049 | 1.1125 | 0.0278 | nan |
| Common | All | League | Mid | AUT | Logistic regression | 247 | 0.4696 | 0.4281 | 0.3502 | 1.0350 | 0.0000 | nan |
| Common | All | League | Mid | AUT | Histogram gradient boosting | 247 | 0.4777 | 0.4357 | 0.3590 | 1.0326 | 0.0000 | nan |
| Common | All | League | Mid | B1 | Prior baseline | 392 | 0.4490 | 0.3333 | 0.2066 | 1.0695 | 0.0000 | nan |
| Common | All | League | Mid | B1 | Gaussian Naive Bayes | 392 | 0.5051 | 0.4467 | 0.3851 | 1.1858 | 0.0096 | nan |
| Common | All | League | Mid | B1 | Logistic regression | 392 | 0.5281 | 0.4669 | 0.4013 | 1.0040 | 0.0096 | nan |
| Common | All | League | Mid | B1 | Histogram gradient boosting | 392 | 0.5153 | 0.4529 | 0.3855 | 1.0040 | 0.0000 | nan |
| Common | All | League | Mid | N1 | Prior baseline | 376 | 0.4388 | 0.3333 | 0.2033 | 1.0721 | 0.0000 | nan |
| Common | All | League | Mid | N1 | Gaussian Naive Bayes | 376 | 0.5585 | 0.4848 | 0.4200 | 0.9615 | 0.0000 | nan |
| Common | All | League | Mid | N1 | Logistic regression | 376 | 0.5798 | 0.5010 | 0.4338 | 0.8894 | 0.0000 | nan |
| Common | All | League | Mid | N1 | Histogram gradient boosting | 376 | 0.5957 | 0.5131 | 0.4451 | 0.8940 | 0.0000 | nan |
| Common | All | League | Mid | P1 | Prior baseline | 377 | 0.4483 | 0.3333 | 0.2063 | 1.0681 | 0.0000 | nan |
| Common | All | League | Mid | P1 | Gaussian Naive Bayes | 377 | 0.5703 | 0.4843 | 0.4283 | 0.9600 | 0.0109 | nan |
| Common | All | League | Mid | P1 | Logistic regression | 377 | 0.5889 | 0.5067 | 0.4589 | 0.8985 | 0.0435 | nan |
| Common | All | League | Mid | P1 | Histogram gradient boosting | 377 | 0.5968 | 0.5085 | 0.4490 | 0.9039 | 0.0109 | nan |
| Common | All | League | Mid | SC0 | Prior baseline | 272 | 0.4301 | 0.3333 | 0.2005 | 1.0778 | 0.0000 | nan |
| Common | All | League | Mid | SC0 | Gaussian Naive Bayes | 272 | 0.5257 | 0.4646 | 0.3953 | 0.9848 | 0.0000 | nan |
| Common | All | League | Mid | SC0 | Logistic regression | 272 | 0.5147 | 0.4525 | 0.3862 | 0.9484 | 0.0000 | nan |
| Common | All | League | Mid | SC0 | Histogram gradient boosting | 272 | 0.5257 | 0.4576 | 0.3896 | 0.9523 | 0.0000 | nan |
| Common | All | League | Lower | FIN | Prior baseline | 248 | 0.4476 | 0.3333 | 0.2061 | 1.0676 | 0.0000 | nan |
| Common | All | League | Lower | FIN | Gaussian Naive Bayes | 248 | 0.4476 | 0.3361 | 0.2165 | 1.0932 | 0.0172 | nan |
| Common | All | League | Lower | FIN | Logistic regression | 248 | 0.4798 | 0.3793 | 0.3114 | 1.0392 | 0.0000 | nan |
| Common | All | League | Lower | FIN | Histogram gradient boosting | 248 | 0.4919 | 0.3895 | 0.3230 | 1.0373 | 0.0000 | nan |
| Common | All | League | Lower | IRL | Prior baseline | 228 | 0.4430 | 0.3333 | 0.2047 | 1.0738 | 0.0000 | nan |
| Common | All | League | Lower | IRL | Gaussian Naive Bayes | 228 | 0.4474 | 0.3387 | 0.2159 | 1.0709 | 0.0000 | nan |
| Common | All | League | Lower | IRL | Logistic regression | 228 | 0.4518 | 0.3545 | 0.2640 | 1.0439 | 0.0000 | nan |
| Common | All | League | Lower | IRL | Histogram gradient boosting | 228 | 0.4430 | 0.3520 | 0.2707 | 1.0443 | 0.0000 | nan |
| Common | All | League | Lower | POL | Prior baseline | 403 | 0.4417 | 0.3333 | 0.2042 | 1.0745 | 0.0000 | nan |
| Common | All | League | Lower | POL | Gaussian Naive Bayes | 403 | 0.4392 | 0.3335 | 0.2144 | 1.0999 | 0.0172 | nan |
| Common | All | League | Lower | POL | Logistic regression | 403 | 0.4442 | 0.3601 | 0.2854 | 1.0668 | 0.0000 | nan |
| Common | All | League | Lower | POL | Histogram gradient boosting | 403 | 0.4417 | 0.3535 | 0.2734 | 1.0755 | 0.0000 | nan |
| Common | All | League | Lower | ROM | Prior baseline | 415 | 0.4169 | 0.3333 | 0.1961 | 1.0881 | 0.0000 | nan |
| Common | All | League | Lower | ROM | Gaussian Naive Bayes | 415 | 0.4289 | 0.3464 | 0.2241 | 1.0990 | 0.0301 | nan |
| Common | All | League | Lower | ROM | Logistic regression | 415 | 0.4265 | 0.3603 | 0.2694 | 1.0764 | 0.0000 | nan |
| Common | All | League | Lower | ROM | Histogram gradient boosting | 415 | 0.4217 | 0.3553 | 0.2642 | 1.0813 | 0.0000 | nan |
| Common | All | League | Lower | SWE | Prior baseline | 350 | 0.4543 | 0.3333 | 0.2083 | 1.0619 | 0.0000 | nan |
| Common | All | League | Lower | SWE | Gaussian Naive Bayes | 350 | 0.4514 | 0.3319 | 0.2139 | 1.1496 | 0.0000 | nan |
| Common | All | League | Lower | SWE | Logistic regression | 350 | 0.4829 | 0.3813 | 0.3349 | 1.0458 | 0.0000 | nan |
| Common | All | League | Lower | SWE | Histogram gradient boosting | 350 | 0.4743 | 0.3704 | 0.3184 | 1.0438 | 0.0000 | nan |
