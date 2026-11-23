# Evidence for the league groups

Source checked on 12 November 2026:

- [Bert Kassies: UEFA Country Ranking 2023](https://kassiesa.net/uefa/data/method5/crank2023.html).
  This historical table gives the ranks and five-season coefficients in
  `league_strength_2023.csv`, covering 2018/19 through 2022/23.
- [UEFA: how association club coefficients are calculated](https://www.uefa.com/nationalassociations/uefarankings/country/about/).
  UEFA explains that an association's season coefficient uses its clubs' results
  in UEFA club competitions, divided by the number of participating clubs.
  Five season coefficients determine the association ranking. The current
  methodology page includes later rule changes; the numerical values here
  come from the historical 2023 table above.

The dataset contains 21 European national top divisions. The CSV includes all
21, including the six left out of this experiment, so the selection is auditable.
Sort these divisions by their **2023** association rank:

1. **Top:** the five highest-ranked available divisions (ranks 1–5).
2. **Mid:** the next five available divisions (ranks 6–10).
3. **Lower:** the five lowest-ranked available divisions (ranks 23, 24, 26, 36, 37).

This selects Netherlands, Portugal, Belgium, Scotland, and Austria for the mid
group, and Sweden, Poland, Romania, Ireland, and Finland for the lower group.
The rules were fixed before examining these leagues' model scores.

The ranking was complete before the evaluation starts in July 2023. Neither
post-June-2023 match results nor incomplete Elo averages determine the groups.
These ranks are analysis metadata and are **not added to the fitted models' inputs**.

The terminology is relative to the available European leagues. These are not
official UEFA tiers or a global ranking. In particular, “lower” does not mean
the five weakest leagues in the world. A UEFA association coefficient measures
the performance of European competition entrants, not the average strength,
competitive balance, or predictability of every club in its domestic division.
Keeping the comparison within UEFA provides one common published measure;
mixing confederations would require an additional cross-confederation measure.

Accuracy can also reflect missing statistics, training sample sizes, outcome
frequencies, and team-strength gaps. The evaluation therefore reports coverage
and baseline scores, uses a common date window, and checks calendar-year results.
