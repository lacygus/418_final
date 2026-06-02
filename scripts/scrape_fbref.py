import logging
import os
import time
from io import StringIO
from typing import Dict

import pandas as pd
import requests

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_fbref.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

LEAGUES = {
    "EPL": "https://fbref.com/en/comps/9/stats/Premier-League-Stats",
    "LaLiga": "https://fbref.com/en/comps/12/stats/La-Liga-Stats",
    "Bundesliga": "https://fbref.com/en/comps/20/stats/Bundesliga-Stats",
    "SerieA": "https://fbref.com/en/comps/11/stats/Serie-A-Stats",
    "Ligue1": "https://fbref.com/en/comps/13/stats/Ligue-1-Stats",
}

DELAY = 4.0  # FBref rate limit -- be safe


def fetch_html(url: str) -> str:
    """Fetch raw HTML from a URL with rate limiting."""
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    logging.info("GET %s status=%d", url, r.status_code)
    return r.text


def extract_stats_table(html: str) -> pd.DataFrame:
    """Parse standard player stats table from FBref HTML."""
    # FBref hides many tables inside HTML comments
    html = html.replace("<!--", "").replace("-->", "")
    tables = pd.read_html(StringIO(html), attrs={"id": "stats_standard"})
    if not tables:
        raise RuntimeError("stats_standard table not found")
    df = tables[0]

    # FBref tables have multi-level columns -- flatten
    df.columns = ["_".join([str(c) for c in col if "Unnamed" not in str(c)]).strip("_")
                  for col in df.columns]

    # Drop repeated header rows
    df = df[df["Player"] != "Player"].reset_index(drop=True)
    return df


def scrape_league(name: str, url: str) -> pd.DataFrame:
    """Scrape one league's standard stats."""
    html = fetch_html(url)
    df = extract_stats_table(html)
    df["league"] = name
    logging.info("%s: %d players", name, len(df))
    print(f"  {name}: {len(df)} players")
    return df


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    all_dfs = []
    for name, url in LEAGUES.items():
        try:
            df = scrape_league(name, url)
            df.to_csv(f"data/raw/fbref_{name}.csv", index=False)
            all_dfs.append(df)
        except Exception as e:
            logging.error("Failed %s: %s", name, e)
            print(f"  FAILED {name}: {e}")

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        out = "data/raw/fbref_all.csv"
        merged.to_csv(out, index=False)
        print(f"\nSaved {len(merged)} total players to {out}")


if __name__ == "__main__":
    main()
