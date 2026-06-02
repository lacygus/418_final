import logging
import os
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_sofifa_ids.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

DELAY = 2.0
PAGES = 12  # 60 per page * 12 = 720 top players


def scrape_listing_page(offset: int) -> List[Dict]:
    """Scrape one listing page from Sofifa top players."""
    url = f"https://sofifa.com/players?type=all&col=oa&sort=desc&offset={offset}"
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    table = soup.find("table")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all("td")
        if len(cells) < 5:
            continue
        link = cells[1].find("a", href=True)
        if not link:
            continue
        # extract player id from URL like /player/239085/erling-haaland/260032/
        href = link["href"]
        parts = href.strip("/").split("/")
        if len(parts) < 3 or parts[0] != "player":
            continue
        sofifa_id = parts[1]
        # name is in <a> text without position chars
        name = link.get_text(strip=True)
        # listing also includes position abbreviations after name
        rows.append({
            "sofifa_id": sofifa_id,
            "sofifa_name": name,
            "sofifa_url": "https://sofifa.com" + href,
        })
    return rows


def main() -> None:
    all_players = []
    for i in range(PAGES):
        offset = i * 60
        try:
            page = scrape_listing_page(offset)
            all_players.extend(page)
            print(f"  page {i+1}/{PAGES} (offset {offset}): {len(page)} players")
        except Exception as e:
            logging.error("page %d failed: %s", i, e)
            print(f"  page {i+1} FAILED: {e}")

    df = pd.DataFrame(all_players)
    out = "data/raw/sofifa_ids.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} Sofifa IDs to {out}")


if __name__ == "__main__":
    main()
