"""Report missing/invalid values in Matches.csv and, when available, external repair counts."""

import pandas as pd

from src.data_sources import RAW_PATH
from src.data_quality import audit_raw, audit_external
from src.models import REPORT_DIR
from src.reporting import write_report


def main():
    raw = pd.read_csv(RAW_PATH, dtype={"MatchTime": "str"})
    fields, summary, leagues = audit_raw(raw)
    REPORT_DIR.mkdir(exist_ok=True)
    fields.to_csv(REPORT_DIR / "data_quality_fields.csv", index=False)
    summary.to_csv(REPORT_DIR / "data_quality_summary.csv", index=False)
    leagues.to_csv(REPORT_DIR / "data_quality_by_league.csv", index=False)
    tables = [("Missing and invalid matches / 比赛层面的缺失与无效值", summary),
              ("Each field / 各字段", fields)]
    source_path = REPORT_DIR / "data_cleaning_source_matches.csv"
    if source_path.exists():
        audit = pd.read_csv(source_path, parse_dates=["MatchDate", "SourceDate"])
        external, by_league = audit_external(raw, audit)
        external.to_csv(REPORT_DIR / "data_quality_external.csv", index=False)
        by_league.to_csv(REPORT_DIR / "data_quality_external_by_league.csv", index=False)
        tables.append(("External checks / 外部核验", external))
    write_report("data_quality.md", "Matches.csv data quality / 原始数据质量",
                 f"{len(raw):,} matches, {len(raw.columns)} columns. Missing values and invalid observed "
                 "values are counted separately; percentages use all matches as denominator. "
                 "Detailed statistics mean shots, shots on target, fouls, corners, yellow/red cards for both teams. "
                 "Cross-field contradictions are flags, not automatic repairs. External checks validate dates/results "
                 "and compare quote availability; they do not imply the missing statistics were recovered. "
                 "缺失与非缺失但无效的数值分别统计；百分比均以全部比赛为分母。跨字段矛盾仅标记，" 
                 "不自动修复。外部核验不代表已补齐缺失的 Elo 或技术统计。", tables)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
