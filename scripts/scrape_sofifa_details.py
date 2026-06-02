import io
import logging
import os
import re
import sys
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_sofifa_details.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

DELAY = 2.0

CATEGORIES = {"Attacking", "Skill", "Movement", "Power", "Mentality", "Defending", "Goalkeeping"}


def parse_attribute_text(text: str) -> tuple[int, str] | None:
    """Split 'NN AttrName' into (value, name)."""
    m = re.match(r"^(\d+)([A-Za-z].*)$", text)
    if not m:
        return None
    return int(m.group(1)), m.group(2).strip()


def scrape_player(sofifa_url: str) -> Dict:
    """Scrape one Sofifa player page and return all FIFA attributes."""
    time.sleep(DELAY)
    r = requests.get(sofifa_url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    stats: Dict[str, int] = {}
    for h5 in soup.find_all("h5"):
        title = h5.get_text(strip=True)
        if title not in CATEGORIES:
            continue
        for p in h5.parent.find_all("p", recursive=False):
            parsed = parse_attribute_text(p.get_text(strip=True))
            if parsed:
                val, name = parsed
                # store as snake_case
                key = name.lower().replace(" ", "_")
                stats[key] = val

    return stats


def main() -> None:
    df = pd.read_csv("data/processed/players_with_sofifa.csv")
    matched = df[df["sofifa_url"].notna()].copy()
    print(f"Scraping {len(matched)} matched players...")

    all_rows = []
    for i, row in enumerate(matched.itertuples(index=False), 1):
        try:
            stats = scrape_player(row.sofifa_url)
            stats["sofifa_id"] = row.sofifa_id
            all_rows.append(stats)
            if i % 25 == 0:
                print(f"  [{i}/{len(matched)}] {row.player}")
        except Exception as e:
            logging.error("failed %s: %s", row.player, e)
            print(f"  [{i}] {row.player} FAILED: {e}")

    stats_df = pd.DataFrame(all_rows)
    out = "data/raw/sofifa_attributes.csv"
    stats_df.to_csv(out, index=False)
    print(f"\nSaved {len(stats_df)} rows to {out}")
    print(f"Attributes captured: {len(stats_df.columns) - 1}")


if __name__ == "__main__":
    main()
