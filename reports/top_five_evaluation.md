# Predictability across the top five leagues

Frozen models; test dates July 2023–June 2026. No refitting or parameter selection. Earlier held-out games can become history for later games. Preparation: Source-verified dates/results; supplied odds. Odds inputs: Matches.csv: OddHome/OddDraw/OddAway. Dates changed: 19,395. Missing/invalid targets: 5.

## All seasons

| Division | Season | Model | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | DrawRecall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | All | Prior baseline | 917 | 0.4209 | 0.3333 | 0.1975 | 1.0810 | 0.0000 |
| D1 | All | Gaussian Naive Bayes | 917 | 0.5169 | 0.4481 | 0.3844 | 1.0104 | 0.0000 |
| D1 | All | Logistic regression | 917 | 0.5234 | 0.4553 | 0.3909 | 0.9771 | 0.0000 |
| D1 | All | Histogram gradient boosting | 917 | 0.5245 | 0.4552 | 0.3929 | 0.9796 | 0.0043 |
| E0 | All | Prior baseline | 1140 | 0.4316 | 0.3333 | 0.2010 | 1.0755 | 0.0000 |
| E0 | All | Gaussian Naive Bayes | 1140 | 0.5263 | 0.4494 | 0.3890 | 1.0182 | 0.0000 |
| E0 | All | Logistic regression | 1140 | 0.5263 | 0.4510 | 0.3900 | 0.9901 | 0.0000 |
| E0 | All | Histogram gradient boosting | 1140 | 0.5272 | 0.4511 | 0.3909 | 0.9889 | 0.0036 |
| F1 | All | Prior baseline | 917 | 0.4406 | 0.3333 | 0.2039 | 1.0710 | 0.0000 |
| F1 | All | Gaussian Naive Bayes | 917 | 0.5005 | 0.4202 | 0.3645 | 1.0208 | 0.0000 |
| F1 | All | Logistic regression | 917 | 0.5169 | 0.4383 | 0.3818 | 1.0096 | 0.0000 |
| F1 | All | Histogram gradient boosting | 917 | 0.5093 | 0.4308 | 0.3768 | 1.0052 | 0.0046 |
| I1 | All | Prior baseline | 1140 | 0.4018 | 0.3333 | 0.1911 | 1.0915 | 0.0000 |
| I1 | All | Gaussian Naive Bayes | 1140 | 0.5289 | 0.4779 | 0.4043 | 1.0283 | 0.0000 |
| I1 | All | Logistic regression | 1140 | 0.5263 | 0.4786 | 0.4075 | 0.9849 | 0.0063 |
| I1 | All | Histogram gradient boosting | 1140 | 0.5316 | 0.4838 | 0.4160 | 0.9883 | 0.0125 |
| SP1 | All | Prior baseline | 1140 | 0.4579 | 0.3333 | 0.2094 | 1.0652 | 0.0000 |
| SP1 | All | Gaussian Naive Bayes | 1140 | 0.5404 | 0.4569 | 0.3933 | 1.0110 | 0.0000 |
| SP1 | All | Logistic regression | 1140 | 0.5342 | 0.4585 | 0.3943 | 0.9697 | 0.0000 |
| SP1 | All | Histogram gradient boosting | 1140 | 0.5351 | 0.4553 | 0.3949 | 0.9667 | 0.0067 |

## By season

| Division | Season | Model | Matches | Accuracy | BalancedAccuracy | MacroF1 | LogLoss | DrawRecall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 | 2023/24 | Prior baseline | 306 | 0.4379 | 0.3333 | 0.2030 | 1.0743 | 0.0000 |
| D1 | 2023/24 | Gaussian Naive Bayes | 306 | 0.5131 | 0.4481 | 0.3837 | 1.0045 | 0.0000 |
| D1 | 2023/24 | Logistic regression | 306 | 0.5196 | 0.4566 | 0.3910 | 0.9642 | 0.0000 |
| D1 | 2023/24 | Histogram gradient boosting | 306 | 0.5163 | 0.4523 | 0.3934 | 0.9677 | 0.0123 |
| D1 | 2024/25 | Prior baseline | 305 | 0.3869 | 0.3333 | 0.1860 | 1.0959 | 0.0000 |
| D1 | 2024/25 | Gaussian Naive Bayes | 305 | 0.4885 | 0.4306 | 0.3627 | 1.0253 | 0.0000 |
| D1 | 2024/25 | Logistic regression | 305 | 0.4918 | 0.4342 | 0.3680 | 1.0023 | 0.0000 |
| D1 | 2024/25 | Histogram gradient boosting | 305 | 0.5049 | 0.4453 | 0.3759 | 1.0046 | 0.0000 |
| D1 | 2025/26 | Prior baseline | 306 | 0.4379 | 0.3333 | 0.2030 | 1.0727 | 0.0000 |
| D1 | 2025/26 | Gaussian Naive Bayes | 306 | 0.5490 | 0.4710 | 0.4066 | 1.0016 | 0.0000 |
| D1 | 2025/26 | Logistic regression | 306 | 0.5588 | 0.4795 | 0.4141 | 0.9648 | 0.0000 |
| D1 | 2025/26 | Histogram gradient boosting | 306 | 0.5523 | 0.4726 | 0.4091 | 0.9666 | 0.0000 |
| E0 | 2023/24 | Prior baseline | 380 | 0.4605 | 0.3333 | 0.2102 | 1.0604 | 0.0000 |
| E0 | 2023/24 | Gaussian Naive Bayes | 380 | 0.5763 | 0.4727 | 0.4178 | 0.9645 | 0.0000 |
| E0 | 2023/24 | Logistic regression | 380 | 0.5711 | 0.4697 | 0.4156 | 0.9273 | 0.0000 |
| E0 | 2023/24 | Histogram gradient boosting | 380 | 0.5842 | 0.4830 | 0.4324 | 0.9299 | 0.0122 |
| E0 | 2024/25 | Prior baseline | 380 | 0.4079 | 0.3333 | 0.1931 | 1.0860 | 0.0000 |
| E0 | 2024/25 | Gaussian Naive Bayes | 380 | 0.5237 | 0.4546 | 0.3935 | 1.0355 | 0.0000 |
| E0 | 2024/25 | Logistic regression | 380 | 0.5132 | 0.4456 | 0.3841 | 1.0107 | 0.0000 |
| E0 | 2024/25 | Histogram gradient boosting | 380 | 0.5184 | 0.4499 | 0.3870 | 0.9995 | 0.0000 |
| E0 | 2025/26 | Prior baseline | 380 | 0.4263 | 0.3333 | 0.1993 | 1.0802 | 0.0000 |
| E0 | 2025/26 | Gaussian Naive Bayes | 380 | 0.4789 | 0.4178 | 0.3527 | 1.0544 | 0.0000 |
| E0 | 2025/26 | Logistic regression | 380 | 0.4947 | 0.4362 | 0.3688 | 1.0325 | 0.0000 |
| E0 | 2025/26 | Histogram gradient boosting | 380 | 0.4789 | 0.4178 | 0.3512 | 1.0372 | 0.0000 |
| F1 | 2023/24 | Prior baseline | 306 | 0.3922 | 0.3333 | 0.1878 | 1.0945 | 0.0000 |
| F1 | 2023/24 | Gaussian Naive Bayes | 306 | 0.4673 | 0.4135 | 0.3435 | 1.0601 | 0.0000 |
| F1 | 2023/24 | Logistic regression | 306 | 0.4804 | 0.4274 | 0.3595 | 1.0521 | 0.0000 |
| F1 | 2023/24 | Histogram gradient boosting | 306 | 0.4706 | 0.4179 | 0.3503 | 1.0471 | 0.0000 |
| F1 | 2024/25 | Prior baseline | 305 | 0.4689 | 0.3333 | 0.2128 | 1.0557 | 0.0000 |
| F1 | 2024/25 | Gaussian Naive Bayes | 305 | 0.5213 | 0.4187 | 0.3728 | 0.9904 | 0.0000 |
| F1 | 2024/25 | Logistic regression | 305 | 0.5410 | 0.4387 | 0.3933 | 0.9782 | 0.0000 |
| F1 | 2024/25 | Histogram gradient boosting | 305 | 0.5344 | 0.4321 | 0.3853 | 0.9796 | 0.0000 |
| F1 | 2025/26 | Prior baseline | 306 | 0.4608 | 0.3333 | 0.2103 | 1.0627 | 0.0000 |
| F1 | 2025/26 | Gaussian Naive Bayes | 306 | 0.5131 | 0.4341 | 0.3768 | 1.0117 | 0.0000 |
| F1 | 2025/26 | Logistic regression | 306 | 0.5294 | 0.4540 | 0.3925 | 0.9984 | 0.0000 |
| F1 | 2025/26 | Histogram gradient boosting | 306 | 0.5229 | 0.4473 | 0.3941 | 0.9889 | 0.0133 |
| I1 | 2023/24 | Prior baseline | 380 | 0.4184 | 0.3333 | 0.1967 | 1.0854 | 0.0000 |
| I1 | 2023/24 | Gaussian Naive Bayes | 380 | 0.5289 | 0.4810 | 0.4047 | 1.0327 | 0.0000 |
| I1 | 2023/24 | Logistic regression | 380 | 0.5342 | 0.4909 | 0.4147 | 0.9808 | 0.0089 |
| I1 | 2023/24 | Histogram gradient boosting | 380 | 0.5342 | 0.4918 | 0.4221 | 0.9763 | 0.0179 |
| I1 | 2024/25 | Prior baseline | 380 | 0.3974 | 0.3333 | 0.1896 | 1.0938 | 0.0000 |
| I1 | 2024/25 | Gaussian Naive Bayes | 380 | 0.5316 | 0.4842 | 0.4090 | 1.0251 | 0.0000 |
| I1 | 2024/25 | Logistic regression | 380 | 0.5211 | 0.4785 | 0.4069 | 0.9721 | 0.0093 |
| I1 | 2024/25 | Histogram gradient boosting | 380 | 0.5263 | 0.4823 | 0.4123 | 0.9784 | 0.0093 |
| I1 | 2025/26 | Prior baseline | 380 | 0.3895 | 0.3333 | 0.1869 | 1.0954 | 0.0000 |
| I1 | 2025/26 | Gaussian Naive Bayes | 380 | 0.5263 | 0.4690 | 0.3989 | 1.0272 | 0.0000 |
| I1 | 2025/26 | Logistic regression | 380 | 0.5237 | 0.4675 | 0.3999 | 1.0019 | 0.0000 |
| I1 | 2025/26 | Histogram gradient boosting | 380 | 0.5342 | 0.4781 | 0.4131 | 1.0103 | 0.0101 |
| SP1 | 2023/24 | Prior baseline | 380 | 0.4395 | 0.3333 | 0.2035 | 1.0750 | 0.0000 |
| SP1 | 2023/24 | Gaussian Naive Bayes | 380 | 0.5342 | 0.4638 | 0.3941 | 1.0220 | 0.0000 |
| SP1 | 2023/24 | Logistic regression | 380 | 0.5395 | 0.4792 | 0.4064 | 0.9566 | 0.0000 |
| SP1 | 2023/24 | Histogram gradient boosting | 380 | 0.5184 | 0.4529 | 0.3848 | 0.9622 | 0.0000 |
| SP1 | 2024/25 | Prior baseline | 380 | 0.4447 | 0.3333 | 0.2052 | 1.0706 | 0.0000 |
| SP1 | 2024/25 | Gaussian Naive Bayes | 380 | 0.5447 | 0.4682 | 0.4043 | 1.0071 | 0.0000 |
| SP1 | 2024/25 | Logistic regression | 380 | 0.5447 | 0.4711 | 0.4059 | 0.9712 | 0.0000 |
| SP1 | 2024/25 | Histogram gradient boosting | 380 | 0.5553 | 0.4810 | 0.4238 | 0.9588 | 0.0206 |
| SP1 | 2025/26 | Prior baseline | 380 | 0.4895 | 0.3333 | 0.2191 | 1.0500 | 0.0000 |
| SP1 | 2025/26 | Gaussian Naive Bayes | 380 | 0.5421 | 0.4370 | 0.3793 | 1.0038 | 0.0000 |
| SP1 | 2025/26 | Logistic regression | 380 | 0.5184 | 0.4239 | 0.3688 | 0.9814 | 0.0000 |
| SP1 | 2025/26 | Histogram gradient boosting | 380 | 0.5316 | 0.4299 | 0.3737 | 0.9790 | 0.0000 |
