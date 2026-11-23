# Supplied odds versus classifiers

The benchmark uses OddHome/OddDraw/OddAway supplied in Matches.csv. Bookmaker identity and closing/pre-closing timing are not established for every row. The lowest quote selects the favourite; ties use home/draw/away priority. Probabilities are (1/odds) divided by the sum of all three inverse odds. Every model is scored on the same eligible games, using prepared dates and the common window from step 04. Accuracy does not measure betting returns. All windows, calendar-year checks and tie sensitivity are in market_average_metrics.csv.

## Tier comparison

| Window | Period | Scope | Group | Model | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | DrawRecall | GapToBenchmark | ModelOnlyCorrect | BenchmarkOnlyCorrect | TiedFavourites | UniqueFavouriteMatches | UniqueFavouriteAccuracy | MeanLeagueAccuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common | All | Tier | Lower | Supplied odds | 1644 | 0.4842 | 0.4189 | 0.3548 | 1.0184 | 0.0000 | 0.0000 | 0 | 0 | 4 | 1640 | 0.4848 | 0.4925 |
| Common | All | Tier | Lower | Prior baseline | 1644 | 0.4392 | 0.3333 | 0.2034 | 1.0741 | 0.0000 | -0.0450 | 147 | 221 | 4 | 1640 | 0.4396 | 0.4407 |
| Common | All | Tier | Lower | Gaussian Naive Bayes | 1644 | 0.4416 | 0.3379 | 0.2182 | 1.1052 | 0.0158 | -0.0426 | 154 | 224 | 4 | 1640 | 0.4421 | 0.4429 |
| Common | All | Tier | Lower | Logistic regression | 1644 | 0.4544 | 0.3691 | 0.2975 | 1.0574 | 0.0000 | -0.0298 | 121 | 170 | 4 | 1640 | 0.4549 | 0.4570 |
| Common | All | Tier | Lower | Histogram gradient boosting | 1644 | 0.4513 | 0.3647 | 0.2910 | 1.0601 | 0.0000 | -0.0328 | 114 | 168 | 4 | 1640 | 0.4518 | 0.4545 |
| Common | All | Tier | Mid | Supplied odds | 1664 | 0.5499 | 0.4827 | 0.4143 | 0.9391 | 0.0000 | 0.0000 | 0 | 0 | 2 | 1662 | 0.5499 | 0.5457 |
| Common | All | Tier | Mid | Prior baseline | 1664 | 0.4339 | 0.3333 | 0.2017 | 1.0756 | 0.0000 | -0.1160 | 123 | 316 | 2 | 1662 | 0.4338 | 0.4302 |
| Common | All | Tier | Mid | Gaussian Naive Bayes | 1664 | 0.5186 | 0.4491 | 0.3901 | 1.0402 | 0.0093 | -0.0312 | 76 | 128 | 2 | 1662 | 0.5193 | 0.5105 |
| Common | All | Tier | Mid | Logistic regression | 1664 | 0.5427 | 0.4747 | 0.4123 | 0.9497 | 0.0116 | -0.0072 | 50 | 62 | 2 | 1662 | 0.5427 | 0.5362 |
| Common | All | Tier | Mid | Histogram gradient boosting | 1664 | 0.5481 | 0.4768 | 0.4106 | 0.9522 | 0.0023 | -0.0018 | 57 | 60 | 2 | 1662 | 0.5481 | 0.5423 |
| Common | All | Tier | Top | Supplied odds | 2099 | 0.5484 | 0.4868 | 0.4154 | 0.9552 | 0.0018 | 0.0000 | 0 | 0 | 4 | 2095 | 0.5484 | 0.5469 |
| Common | All | Tier | Top | Prior baseline | 2099 | 0.4297 | 0.3333 | 0.2004 | 1.0782 | 0.0000 | -0.1186 | 144 | 393 | 4 | 2095 | 0.4296 | 0.4287 |
| Common | All | Tier | Top | Gaussian Naive Bayes | 2099 | 0.5236 | 0.4575 | 0.3904 | 1.0179 | 0.0000 | -0.0248 | 66 | 118 | 4 | 2095 | 0.5241 | 0.5209 |
| Common | All | Tier | Top | Logistic regression | 2099 | 0.5303 | 0.4676 | 0.3997 | 0.9729 | 0.0018 | -0.0181 | 51 | 89 | 4 | 2095 | 0.5308 | 0.5281 |
| Common | All | Tier | Top | Histogram gradient boosting | 2099 | 0.5245 | 0.4615 | 0.3974 | 0.9736 | 0.0071 | -0.0238 | 50 | 100 | 4 | 2095 | 0.5246 | 0.5223 |

## League comparison

| Window | Period | Scope | Group | Model | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | DrawRecall | GapToBenchmark | ModelOnlyCorrect | BenchmarkOnlyCorrect | TiedFavourites | UniqueFavouriteMatches | UniqueFavouriteAccuracy | MeanLeagueAccuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Common | All | League | AUT | Supplied odds | 247 | 0.4939 | 0.4610 | 0.3842 | 0.9934 | 0.0000 | 0.0000 | 0 | 0 | 1 | 246 | 0.4959 | nan |
| Common | All | League | AUT | Prior baseline | 247 | 0.3846 | 0.3333 | 0.1852 | 1.1000 | 0.0000 | -0.1093 | 23 | 50 | 1 | 246 | 0.3862 | nan |
| Common | All | League | AUT | Gaussian Naive Bayes | 247 | 0.3927 | 0.3426 | 0.2049 | 1.1125 | 0.0278 | -0.1012 | 25 | 50 | 1 | 246 | 0.3943 | nan |
| Common | All | League | AUT | Logistic regression | 247 | 0.4696 | 0.4281 | 0.3502 | 1.0350 | 0.0000 | -0.0243 | 15 | 21 | 1 | 246 | 0.4715 | nan |
| Common | All | League | AUT | Histogram gradient boosting | 247 | 0.4777 | 0.4357 | 0.3590 | 1.0326 | 0.0000 | -0.0162 | 17 | 21 | 1 | 246 | 0.4797 | nan |
| Common | All | League | B1 | Supplied odds | 392 | 0.5230 | 0.4651 | 0.3947 | 0.9959 | 0.0000 | 0.0000 | 0 | 0 | 1 | 391 | 0.5217 | nan |
| Common | All | League | B1 | Prior baseline | 392 | 0.4490 | 0.3333 | 0.2066 | 1.0695 | 0.0000 | -0.0740 | 42 | 71 | 1 | 391 | 0.4476 | nan |
| Common | All | League | B1 | Gaussian Naive Bayes | 392 | 0.5051 | 0.4467 | 0.3851 | 1.1858 | 0.0096 | -0.0179 | 17 | 24 | 1 | 391 | 0.5064 | nan |
| Common | All | League | B1 | Logistic regression | 392 | 0.5281 | 0.4669 | 0.4013 | 1.0040 | 0.0096 | 0.0051 | 10 | 8 | 1 | 391 | 0.5269 | nan |
| Common | All | League | B1 | Histogram gradient boosting | 392 | 0.5153 | 0.4529 | 0.3855 | 1.0040 | 0.0000 | -0.0077 | 9 | 12 | 1 | 391 | 0.5141 | nan |
| Common | All | League | D1 | Supplied odds | 360 | 0.5472 | 0.4806 | 0.4137 | 0.9499 | 0.0000 | 0.0000 | 0 | 0 | 2 | 358 | 0.5503 | nan |
| Common | All | League | D1 | Prior baseline | 360 | 0.4278 | 0.3333 | 0.1997 | 1.0783 | 0.0000 | -0.1194 | 26 | 69 | 2 | 358 | 0.4302 | nan |
| Common | All | League | D1 | Gaussian Naive Bayes | 360 | 0.5083 | 0.4424 | 0.3797 | 1.0077 | 0.0000 | -0.0389 | 14 | 28 | 2 | 358 | 0.5112 | nan |
| Common | All | League | D1 | Logistic regression | 360 | 0.5222 | 0.4564 | 0.3920 | 0.9703 | 0.0000 | -0.0250 | 9 | 18 | 2 | 358 | 0.5251 | nan |
| Common | All | League | D1 | Histogram gradient boosting | 360 | 0.5139 | 0.4482 | 0.3897 | 0.9756 | 0.0108 | -0.0333 | 8 | 20 | 2 | 358 | 0.5168 | nan |
| Common | All | League | E0 | Supplied odds | 450 | 0.5822 | 0.4959 | 0.4331 | 0.9096 | 0.0000 | 0.0000 | 0 | 0 | 1 | 449 | 0.5813 | nan |
| Common | All | League | E0 | Prior baseline | 450 | 0.4489 | 0.3333 | 0.2065 | 1.0666 | 0.0000 | -0.1333 | 38 | 98 | 1 | 449 | 0.4477 | nan |
| Common | All | League | E0 | Gaussian Naive Bayes | 450 | 0.5644 | 0.4730 | 0.4148 | 0.9722 | 0.0000 | -0.0178 | 17 | 25 | 1 | 449 | 0.5635 | nan |
| Common | All | League | E0 | Logistic regression | 450 | 0.5644 | 0.4743 | 0.4158 | 0.9277 | 0.0000 | -0.0178 | 14 | 22 | 1 | 449 | 0.5635 | nan |
| Common | All | League | E0 | Histogram gradient boosting | 450 | 0.5756 | 0.4854 | 0.4298 | 0.9300 | 0.0097 | -0.0067 | 18 | 21 | 1 | 449 | 0.5746 | nan |
| Common | All | League | F1 | Supplied odds | 369 | 0.5122 | 0.4542 | 0.3867 | 1.0010 | 0.0000 | 0.0000 | 0 | 0 | 1 | 368 | 0.5109 | nan |
| Common | All | League | F1 | Prior baseline | 369 | 0.4065 | 0.3333 | 0.1927 | 1.0880 | 0.0000 | -0.1057 | 28 | 67 | 1 | 368 | 0.4049 | nan |
| Common | All | League | F1 | Gaussian Naive Bayes | 369 | 0.4715 | 0.4127 | 0.3478 | 1.0517 | 0.0000 | -0.0407 | 10 | 25 | 1 | 368 | 0.4728 | nan |
| Common | All | League | F1 | Logistic regression | 369 | 0.4851 | 0.4279 | 0.3651 | 1.0364 | 0.0000 | -0.0271 | 9 | 19 | 1 | 368 | 0.4864 | nan |
| Common | All | League | F1 | Histogram gradient boosting | 369 | 0.4743 | 0.4169 | 0.3534 | 1.0283 | 0.0000 | -0.0379 | 9 | 23 | 1 | 368 | 0.4728 | nan |
| Common | All | League | FIN | Supplied odds | 248 | 0.5605 | 0.4758 | 0.4135 | 0.9957 | 0.0000 | 0.0000 | 0 | 0 | 1 | 247 | 0.5628 | nan |
| Common | All | League | FIN | Prior baseline | 248 | 0.4476 | 0.3333 | 0.2061 | 1.0676 | 0.0000 | -0.1129 | 20 | 48 | 1 | 247 | 0.4494 | nan |
| Common | All | League | FIN | Gaussian Naive Bayes | 248 | 0.4476 | 0.3361 | 0.2165 | 1.0932 | 0.0172 | -0.1129 | 20 | 48 | 1 | 247 | 0.4494 | nan |
| Common | All | League | FIN | Logistic regression | 248 | 0.4798 | 0.3793 | 0.3114 | 1.0392 | 0.0000 | -0.0806 | 20 | 40 | 1 | 247 | 0.4818 | nan |
| Common | All | League | FIN | Histogram gradient boosting | 248 | 0.4919 | 0.3895 | 0.3230 | 1.0373 | 0.0000 | -0.0685 | 19 | 36 | 1 | 247 | 0.4939 | nan |
| Common | All | League | I1 | Supplied odds | 450 | 0.5311 | 0.4970 | 0.4107 | 0.9759 | 0.0000 | 0.0000 | 0 | 0 | 0 | 450 | 0.5311 | nan |
| Common | All | League | I1 | Prior baseline | 450 | 0.4178 | 0.3333 | 0.1964 | 1.0863 | 0.0000 | -0.1133 | 33 | 84 | 0 | 450 | 0.4178 | nan |
| Common | All | League | I1 | Gaussian Naive Bayes | 450 | 0.5178 | 0.4742 | 0.3959 | 1.0411 | 0.0000 | -0.0133 | 11 | 17 | 0 | 450 | 0.5178 | nan |
| Common | All | League | I1 | Logistic regression | 450 | 0.5200 | 0.4810 | 0.4025 | 0.9903 | 0.0074 | -0.0111 | 7 | 12 | 0 | 450 | 0.5200 | nan |
| Common | All | League | I1 | Histogram gradient boosting | 450 | 0.5178 | 0.4799 | 0.4072 | 0.9890 | 0.0147 | -0.0133 | 7 | 13 | 0 | 450 | 0.5178 | nan |
| Common | All | League | IRL | Supplied odds | 228 | 0.5044 | 0.4522 | 0.3773 | 0.9940 | 0.0000 | 0.0000 | 0 | 0 | 0 | 228 | 0.5044 | nan |
| Common | All | League | IRL | Prior baseline | 228 | 0.4430 | 0.3333 | 0.2047 | 1.0738 | 0.0000 | -0.0614 | 21 | 35 | 0 | 228 | 0.4430 | nan |
| Common | All | League | IRL | Gaussian Naive Bayes | 228 | 0.4474 | 0.3387 | 0.2159 | 1.0709 | 0.0000 | -0.0570 | 22 | 35 | 0 | 228 | 0.4474 | nan |
| Common | All | League | IRL | Logistic regression | 228 | 0.4518 | 0.3545 | 0.2640 | 1.0439 | 0.0000 | -0.0526 | 18 | 30 | 0 | 228 | 0.4518 | nan |
| Common | All | League | IRL | Histogram gradient boosting | 228 | 0.4430 | 0.3520 | 0.2707 | 1.0443 | 0.0000 | -0.0614 | 18 | 32 | 0 | 228 | 0.4430 | nan |
| Common | All | League | N1 | Supplied odds | 376 | 0.5798 | 0.4995 | 0.4329 | 0.8995 | 0.0000 | 0.0000 | 0 | 0 | 0 | 376 | 0.5798 | nan |
| Common | All | League | N1 | Prior baseline | 376 | 0.4388 | 0.3333 | 0.2033 | 1.0721 | 0.0000 | -0.1410 | 25 | 78 | 0 | 376 | 0.4388 | nan |
| Common | All | League | N1 | Gaussian Naive Bayes | 376 | 0.5585 | 0.4848 | 0.4200 | 0.9615 | 0.0000 | -0.0213 | 13 | 21 | 0 | 376 | 0.5585 | nan |
| Common | All | League | N1 | Logistic regression | 376 | 0.5798 | 0.5010 | 0.4338 | 0.8894 | 0.0000 | 0.0000 | 9 | 9 | 0 | 376 | 0.5798 | nan |
| Common | All | League | N1 | Histogram gradient boosting | 376 | 0.5957 | 0.5131 | 0.4451 | 0.8940 | 0.0000 | 0.0160 | 11 | 5 | 0 | 376 | 0.5957 | nan |
| Common | All | League | P1 | Supplied odds | 377 | 0.5915 | 0.5011 | 0.4398 | 0.8958 | 0.0000 | 0.0000 | 0 | 0 | 0 | 377 | 0.5915 | nan |
| Common | All | League | P1 | Prior baseline | 377 | 0.4483 | 0.3333 | 0.2063 | 1.0681 | 0.0000 | -0.1432 | 14 | 68 | 0 | 377 | 0.4483 | nan |
| Common | All | League | P1 | Gaussian Naive Bayes | 377 | 0.5703 | 0.4843 | 0.4283 | 0.9600 | 0.0109 | -0.0212 | 10 | 18 | 0 | 377 | 0.5703 | nan |
| Common | All | League | P1 | Logistic regression | 377 | 0.5889 | 0.5067 | 0.4589 | 0.8985 | 0.0435 | -0.0027 | 13 | 14 | 0 | 377 | 0.5889 | nan |
| Common | All | League | P1 | Histogram gradient boosting | 377 | 0.5968 | 0.5085 | 0.4490 | 0.9039 | 0.0109 | 0.0053 | 11 | 9 | 0 | 377 | 0.5968 | nan |
| Common | All | League | POL | Supplied odds | 403 | 0.4442 | 0.3826 | 0.3173 | 1.0459 | 0.0000 | 0.0000 | 0 | 0 | 0 | 403 | 0.4442 | nan |
| Common | All | League | POL | Prior baseline | 403 | 0.4417 | 0.3333 | 0.2042 | 1.0745 | 0.0000 | -0.0025 | 39 | 40 | 0 | 403 | 0.4417 | nan |
| Common | All | League | POL | Gaussian Naive Bayes | 403 | 0.4392 | 0.3335 | 0.2144 | 1.0999 | 0.0172 | -0.0050 | 41 | 43 | 0 | 403 | 0.4392 | nan |
| Common | All | League | POL | Logistic regression | 403 | 0.4442 | 0.3601 | 0.2854 | 1.0668 | 0.0000 | 0.0000 | 28 | 28 | 0 | 403 | 0.4442 | nan |
| Common | All | League | POL | Histogram gradient boosting | 403 | 0.4417 | 0.3535 | 0.2734 | 1.0755 | 0.0000 | -0.0025 | 29 | 30 | 0 | 403 | 0.4417 | nan |
| Common | All | League | ROM | Supplied odds | 415 | 0.4506 | 0.4090 | 0.3314 | 1.0363 | 0.0000 | 0.0000 | 0 | 0 | 2 | 413 | 0.4528 | nan |
| Common | All | League | ROM | Prior baseline | 415 | 0.4169 | 0.3333 | 0.1961 | 1.0881 | 0.0000 | -0.0337 | 29 | 43 | 2 | 413 | 0.4189 | nan |
| Common | All | League | ROM | Gaussian Naive Bayes | 415 | 0.4289 | 0.3464 | 0.2241 | 1.0990 | 0.0301 | -0.0217 | 33 | 42 | 2 | 413 | 0.4310 | nan |
| Common | All | League | ROM | Logistic regression | 415 | 0.4265 | 0.3603 | 0.2694 | 1.0764 | 0.0000 | -0.0241 | 27 | 37 | 2 | 413 | 0.4286 | nan |
| Common | All | League | ROM | Histogram gradient boosting | 415 | 0.4217 | 0.3553 | 0.2642 | 1.0813 | 0.0000 | -0.0289 | 24 | 36 | 2 | 413 | 0.4237 | nan |
| Common | All | League | SC0 | Supplied odds | 272 | 0.5404 | 0.4760 | 0.4073 | 0.9224 | 0.0000 | 0.0000 | 0 | 0 | 0 | 272 | 0.5404 | nan |
| Common | All | League | SC0 | Prior baseline | 272 | 0.4301 | 0.3333 | 0.2005 | 1.0778 | 0.0000 | -0.1103 | 19 | 49 | 0 | 272 | 0.4301 | nan |
| Common | All | League | SC0 | Gaussian Naive Bayes | 272 | 0.5257 | 0.4646 | 0.3953 | 0.9848 | 0.0000 | -0.0147 | 11 | 15 | 0 | 272 | 0.5257 | nan |
| Common | All | League | SC0 | Logistic regression | 272 | 0.5147 | 0.4525 | 0.3862 | 0.9484 | 0.0000 | -0.0257 | 3 | 10 | 0 | 272 | 0.5147 | nan |
| Common | All | League | SC0 | Histogram gradient boosting | 272 | 0.5257 | 0.4576 | 0.3896 | 0.9523 | 0.0000 | -0.0147 | 9 | 13 | 0 | 272 | 0.5257 | nan |
| Common | All | League | SP1 | Supplied odds | 470 | 0.5617 | 0.4981 | 0.4243 | 0.9471 | 0.0075 | 0.0000 | 0 | 0 | 0 | 470 | 0.5617 | nan |
| Common | All | League | SP1 | Prior baseline | 470 | 0.4426 | 0.3333 | 0.2045 | 1.0739 | 0.0000 | -0.1191 | 19 | 75 | 0 | 470 | 0.4426 | nan |
| Common | All | League | SP1 | Gaussian Naive Bayes | 470 | 0.5426 | 0.4748 | 0.4026 | 1.0209 | 0.0000 | -0.0191 | 14 | 23 | 0 | 470 | 0.5426 | nan |
| Common | All | League | SP1 | Logistic regression | 470 | 0.5489 | 0.4906 | 0.4141 | 0.9516 | 0.0000 | -0.0128 | 12 | 18 | 0 | 470 | 0.5489 | nan |
| Common | All | League | SP1 | Histogram gradient boosting | 470 | 0.5298 | 0.4671 | 0.3960 | 0.9563 | 0.0000 | -0.0319 | 8 | 23 | 0 | 470 | 0.5298 | nan |
| Common | All | League | SWE | Supplied odds | 350 | 0.5029 | 0.4052 | 0.3606 | 0.9974 | 0.0000 | 0.0000 | 0 | 0 | 1 | 349 | 0.5014 | nan |
| Common | All | League | SWE | Prior baseline | 350 | 0.4543 | 0.3333 | 0.2083 | 1.0619 | 0.0000 | -0.0486 | 38 | 55 | 1 | 349 | 0.4527 | nan |
| Common | All | League | SWE | Gaussian Naive Bayes | 350 | 0.4514 | 0.3319 | 0.2139 | 1.1496 | 0.0000 | -0.0514 | 38 | 56 | 1 | 349 | 0.4499 | nan |
| Common | All | League | SWE | Logistic regression | 350 | 0.4829 | 0.3813 | 0.3349 | 1.0458 | 0.0000 | -0.0200 | 28 | 35 | 1 | 349 | 0.4814 | nan |
| Common | All | League | SWE | Histogram gradient boosting | 350 | 0.4743 | 0.3704 | 0.3184 | 1.0438 | 0.0000 | -0.0286 | 24 | 34 | 1 | 349 | 0.4728 | nan |

## Coverage

| Division | Tier | Status | Matches |
| --- | --- | --- | --- |
| AUT | Mid | Eligible | 247 |
| B1 | Mid | Eligible | 935 |
| D1 | Top | Eligible | 917 |
| E0 | Top | Eligible | 1140 |
| F1 | Top | Eligible | 917 |
| FIN | Lower | Eligible | 248 |
| I1 | Top | Eligible | 1140 |
| IRL | Lower | Eligible | 232 |
| N1 | Mid | Eligible | 918 |
| P1 | Mid | Eligible | 918 |
| POL | Lower | Eligible | 403 |
| ROM | Lower | Eligible | 415 |
| SC0 | Mid | Eligible | 656 |
| SP1 | Top | Eligible | 1140 |
| SWE | Lower | Eligible | 350 |
