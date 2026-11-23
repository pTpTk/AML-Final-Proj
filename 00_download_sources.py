"""Download public Football-Data snapshots explicitly; Matches.csv defines the requests."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
import argparse
import hashlib
import time

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "data" / "football_data_sources"
DOWNLOAD_MANIFEST = SOURCE_DIR / "download_manifest.csv"
RAW_PATH = ROOT / "data" / "Matches.csv"

STANDARD_DIVISIONS = [
    "E0", "E1", "E2", "E3", "EC", "SC0", "SC1", "SC2", "SC3",
    "F1", "F2", "D1", "D2", "I1", "I2", "SP1", "SP2", "B1", "N1", "P1", "T1", "G1",
]


def historical_source_urls(raw):
    """Request only league/season files represented in the original dataset."""
    urls = {}
    for code, group in raw.groupby("Division"):
        if code in STANDARD_DIVISIONS:
            years = group["MatchDate"].dt.year - group["MatchDate"].dt.month.lt(7).astype(int)
            for year in sorted(years.unique()):
                season = f"{year % 100:02d}{(year + 1) % 100:02d}"
                urls[f"{season}_{code}.csv"] = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
        else:
            source_code = {"DEN": "DNK", "JAP": "JPN", "SUI": "SWZ"}.get(code, code)
            urls[f"{code}.csv"] = f"https://www.football-data.co.uk/new/{source_code}.csv"
    return urls


def download_history(urls, refresh=False):
    """Fetch required public CSVs and record URLs, hashes, sizes and retrieval status."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    def download(item):
        filename, url = item
        path = SOURCE_DIR / filename
        status = "Cached"
        if refresh or not path.exists():
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=40) as response:
                        body = response.read()
                    if not body or b"<html" in body[:500].lower():
                        raise ValueError(f"Expected a CSV from {url}")
                    temporary = path.with_suffix(".csv.part")
                    temporary.write_bytes(body)
                    temporary.replace(path)
                    status = "Downloaded"
                    break
                except (OSError, HTTPError):
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
        body = path.read_bytes()
        return {"File": filename, "URL": url, "SHA256": hashlib.sha256(body).hexdigest(),
                "Bytes": len(body), "Status": status,
                "CheckedAtUTC": datetime.now(timezone.utc).isoformat()}

    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Consume every result so failures never leave missing files unnoticed.
        futures = [(item, pool.submit(download, item)) for item in urls.items()]
        for position, ((filename, url), future) in enumerate(futures, start=1):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append(filename)
                rows.append({"File": filename, "URL": url, "Status": "Failed", "Error": str(error)})
            if position % 40 == 0 or position == len(urls):
                print(f"Source files {position}/{len(urls)}; failures: {len(failures)}", flush=True)
    manifest = pd.DataFrame(rows)
    manifest.to_csv(DOWNLOAD_MANIFEST, index=False)
    if failures:
        raise RuntimeError(f"Could not obtain {len(failures)} required sources; see {DOWNLOAD_MANIFEST}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Redownload existing snapshots as well.")
    args = parser.parse_args()
    raw = pd.read_csv(RAW_PATH, usecols=["Division", "MatchDate"], parse_dates=["MatchDate"])
    urls = historical_source_urls(raw)
    manifest = download_history(urls, refresh=args.refresh)
    print(manifest["Status"].value_counts().to_string())
    print(f"Download provenance saved to {DOWNLOAD_MANIFEST}")


if __name__ == "__main__":
    main()
