"""Replay seasonal prediction batches and independently calculate reported log loss."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parents[1]
out = root / "reports" / "reproduction"
features = pd.read_csv(root / "docs/match_prediction_fields.csv").query("role == 'feature'").field.tolist()
selected = []
for chunk in pd.read_csv(root / "data/MatchTraining.csv", parse_dates=["MatchDate"], chunksize=20000,
                         dtype={field: "float32" for field in features}):
    selected.append(chunk.loc[chunk.Division.eq("E0") & chunk.MatchDate.ge("2023-07-01")
                              & chunk.MatchDate.lt("2026-07-01") & chunk.FTResult.isin(["A", "D", "H"])])
test = pd.concat(selected)
exported = pd.read_csv(root / "reports/model_predictions.csv")
metrics = pd.read_csv(root / "reports/model_metrics.csv")
records = []
with threadpool_limits(limits=2):
    for model_name in metrics.Model.unique():
        slug = model_name.lower().replace(" ", "_")
        model = joblib.load(root / "models" / f"{slug}.joblib")
        for start in [2023, 2024, 2025]:
            season = f"{start}/{(start + 1) % 100:02d}"
            frame = test.loc[test.MatchDate.ge(f"{start}-07-01") & test.MatchDate.lt(f"{start + 1}-07-01")]
            order = [list(model.classes_).index(label) for label in ["A", "D", "H"]]
            p = model.predict_proba(frame[features])[:, order].astype(float)
            p /= p.sum(axis=1, keepdims=True)
            y = np.searchsorted(np.array(["A", "D", "H"]), frame.FTResult.to_numpy())
            loss = -np.log(np.clip(p[np.arange(len(frame)), y], np.finfo(float).eps, 1 - np.finfo(float).eps)).mean()
            reported = float(metrics.loc[metrics.Model.eq(model_name) & metrics.Season.eq(season), "LogLoss"].iloc[0])
            q = exported.loc[exported.Season.eq(season), [f"{slug}_probability_{label}" for label in ["A", "D", "H"]]].to_numpy(float)
            records.append({"Model": model_name, "Season": season, "ReportedLogLoss": reported,
                            "ReplayedLogLoss": loss, "AbsoluteLossDifference": abs(loss - reported),
                            "MaxProbabilityDifferenceFromFullBatchExport": float(np.abs(p - q).max())})
result = pd.DataFrame(records)
result.to_csv(out / "seasonal_batch_precision.csv", index=False)
print(result.to_string(index=False))
assert result.AbsoluteLossDifference.le(1e-10).all()
