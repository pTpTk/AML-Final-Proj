"""Download public Football-Data snapshots explicitly; Matches.csv defines the requests."""

import argparse
import pandas as pd

from src.data_sources import RAW_PATH, DOWNLOAD_MANIFEST, historical_source_urls, download_history


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
