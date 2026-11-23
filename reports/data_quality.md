# Matches.csv data quality / 原始数据质量

238,858 matches, 48 columns. Missing values and invalid observed values are counted separately; percentages use all matches as denominator. Detailed statistics mean shots, shots on target, fouls, corners, yellow/red cards for both teams. Cross-field contradictions are flags, not automatic repairs. External checks validate dates/results and compare quote availability; they do not imply the missing statistics were recovered. 缺失与非缺失但无效的数值分别统计；百分比均以全部比赛为分母。跨字段矛盾仅标记，不自动修复。外部核验不代表已补齐缺失的 Elo 或技术统计。

## Missing and invalid matches / 比赛层面的缺失与无效值

| Check | Matches | Percent |
| --- | --- | --- |
| Missing/invalid full-time result | 4 | 0.0017 |
| Missing at least one full-time goal count | 4 | 0.0017 |
| Missing at least one Elo | 98144 | 41.0888 |
| Missing both Elo values | 79845 | 33.4278 |
| Missing at least one half-time goal count | 54580 | 22.8504 |
| Missing at least one detailed statistic | 117953 | 49.3821 |
| Missing all detailed statistics | 111259 | 46.5796 |
| Missing/incomplete odds triplet | 3042 | 1.2736 |
| Complete but invalid odds triplet | 7 | 0.0029 |
| Missing kickoff time | 131485 | 55.0474 |
| Unparseable date | 0 | 0.0000 |
| Duplicate full row (after first) | 0 | 0.0000 |
| Duplicate match identity (after first) | 0 | 0.0000 |
| Same home and away team | 0 | 0.0000 |
| Known result disagrees with supplied goals | 0 | 0.0000 |
| Half-time goals exceed full-time goals | 0 | 0.0000 |

## Each field / 各字段

| Field | Rows | Missing | MissingPercent | InvalidObserved | Rule |
| --- | --- | --- | --- | --- | --- |
| Division | 238858 | 0 | 0.0000 | 0 | Nonempty identifier |
| MatchDate | 238858 | 0 | 0.0000 | 0 | %Y-%m-%d |
| MatchTime | 238858 | 131485 | 55.0474 | 0 | %H:%M:%S |
| HomeTeam | 238858 | 0 | 0.0000 | 0 | Nonempty identifier |
| AwayTeam | 238858 | 0 | 0.0000 | 0 | Nonempty identifier |
| HomeElo | 238858 | 88960 | 37.2439 | 0 | Finite Elo > 0 |
| AwayElo | 238858 | 89029 | 37.2728 | 0 | Finite Elo > 0 |
| Form3Home | 238858 | 1524 | 0.6380 | 0 | Finite numeric value |
| Form5Home | 238858 | 1535 | 0.6426 | 0 | Finite numeric value |
| Form3Away | 238858 | 1523 | 0.6376 | 0 | Finite numeric value |
| Form5Away | 238858 | 1534 | 0.6422 | 0 | Finite numeric value |
| FTHome | 238858 | 4 | 0.0017 | 0 | Finite nonnegative integer |
| FTAway | 238858 | 4 | 0.0017 | 0 | Finite nonnegative integer |
| FTResult | 238858 | 4 | 0.0017 | 0 | H, D or A |
| HTHome | 238858 | 54580 | 22.8504 | 0 | Finite nonnegative integer |
| HTAway | 238858 | 54580 | 22.8504 | 0 | Finite nonnegative integer |
| HTResult | 238858 | 54580 | 22.8504 | 0 | H, D or A |
| HomeShots | 238858 | 116375 | 48.7214 | 0 | Finite nonnegative integer |
| AwayShots | 238858 | 116372 | 48.7202 | 0 | Finite nonnegative integer |
| HomeTarget | 238858 | 117181 | 49.0589 | 0 | Finite nonnegative integer |
| AwayTarget | 238858 | 117178 | 49.0576 | 0 | Finite nonnegative integer |
| HomeFouls | 238858 | 117137 | 49.0404 | 0 | Finite nonnegative integer |
| AwayFouls | 238858 | 117137 | 49.0404 | 0 | Finite nonnegative integer |
| HomeCorners | 238858 | 116747 | 48.8772 | 0 | Finite nonnegative integer |
| AwayCorners | 238858 | 116747 | 48.8772 | 0 | Finite nonnegative integer |
| HomeYellow | 238858 | 111260 | 46.5800 | 0 | Finite nonnegative integer |
| AwayYellow | 238858 | 111259 | 46.5796 | 0 | Finite nonnegative integer |
| HomeRed | 238858 | 111259 | 46.5796 | 0 | Finite nonnegative integer |
| AwayRed | 238858 | 111261 | 46.5804 | 0 | Finite nonnegative integer |
| OddHome | 238858 | 3042 | 1.2736 | 7 | Finite decimal odds > 1 |
| OddDraw | 238858 | 3042 | 1.2736 | 0 | Finite decimal odds > 1 |
| OddAway | 238858 | 3042 | 1.2736 | 0 | Finite decimal odds > 1 |
| MaxHome | 238858 | 27647 | 11.5747 | 0 | Finite decimal odds > 1 |
| MaxDraw | 238858 | 27647 | 11.5747 | 0 | Finite decimal odds > 1 |
| MaxAway | 238858 | 27647 | 11.5747 | 1 | Finite decimal odds > 1 |
| Over25 | 238858 | 82171 | 34.4016 | 0 | Finite decimal odds > 1 |
| Under25 | 238858 | 82172 | 34.4020 | 0 | Finite decimal odds > 1 |
| MaxOver25 | 238858 | 82171 | 34.4016 | 0 | Finite decimal odds > 1 |
| MaxUnder25 | 238858 | 82172 | 34.4020 | 0 | Finite decimal odds > 1 |
| HandiSize | 238858 | 73846 | 30.9163 | 0 | Finite numeric value |
| HandiHome | 238858 | 74107 | 31.0255 | 4 | Finite decimal odds > 1 |
| HandiAway | 238858 | 74131 | 31.0356 | 3 | Finite decimal odds > 1 |
| C_LTH | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |
| C_LTA | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |
| C_VHD | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |
| C_VAD | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |
| C_HTB | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |
| C_PHB | 238858 | 126256 | 52.8582 | 0 | Finite numeric value |

## External checks / 外部核验

| Check | Matches | Percent |
| --- | --- | --- |
| Date differs from external source | 19395 | 8.1199 |
| Date change moves match into/out of training | 1075 | 0.4501 |
| In raw training only | 611 | 0.2558 |
| In source-date training only | 464 | 0.1943 |
| Date change moves match into/out of test | 1075 | 0.4501 |
| Known result disagrees with source | 1 | 0.0004 |
| Missing raw odds but usable external averages | 180 | 0.0754 |
| Usable raw odds but no usable external averages | 24749 | 10.3614 |
| No usable external market averages | 27618 | 11.5625 |
