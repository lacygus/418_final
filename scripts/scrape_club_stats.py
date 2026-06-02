import logging
import os
import re
import time
from typing import Dict, List

import pandas as pd
import requests
from bs4 import BeautifulSoup

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/scrape_club_stats.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.transfermarkt.com/",
}

DELAY = 3.0


def to_int(text: str) -> int | None:
    """Convert a text cell to int, treating '-' and empty as None."""
    text = text.strip().replace(".", "").replace(",", "")
    if not text or text in {"-", "?"}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def to_minutes(text: str) -> int | None:
    """Parse '1.234' minutes string to int."""
    text = text.strip().replace(".", "").replace(",", "").replace("'", "")
    if not text or text == "-":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def club_stats_url(club_url: str) -> str:
    """Convert club startseite URL to detailed leistungsdaten URL (plus/1)."""
    base = club_url.replace("/startseite/", "/leistungsdaten/").split("/saison_id")[0]
    return base + "/plus/1"


def scrape_club(url: str) -> List[Dict]:
    """Scrape squad stats for one club's leistungsdaten page."""
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    table = soup.find("table", class_="items")
    if not table:
        logging.warning("no items table on %s", url)
        return []

    rows = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 15:
            continue

        name_tag = tr.find("td", class_="hauptlink")
        if not name_tag:
            continue
        a = name_tag.find("a", title=True)
        name = a["title"] if a else name_tag.get_text(strip=True)

        rows.append({
            "player": name,
            "appearances": to_int(cells[5].get_text(strip=True)),
            "goals": to_int(cells[6].get_text(strip=True)),
            "assists": to_int(cells[7].get_text(strip=True)),
            "yellow_cards": to_int(cells[8].get_text(strip=True)),
            "red_cards": to_int(cells[10].get_text(strip=True)),
            "minutes": to_minutes(cells[14].get_text(strip=True)),
        })

    logging.info("scraped %d players from %s", len(rows), url)
    return rows


def main() -> None:
    df = pd.read_csv("data/raw/transfermarkt_all.csv")
    club_urls = df["club_url"].dropna().unique()
    print(f"Found {len(club_urls)} unique clubs")

    all_rows = []
    for i, url in enumerate(club_urls, 1):
        stats_url = club_stats_url(url)
        try:
            rows = scrape_club(stats_url)
            all_rows.extend(rows)
            print(f"  [{i}/{len(club_urls)}] {url.split('/')[3]}: {len(rows)} players")
        except Exception as e:
            logging.error("failed %s: %s", url, e)
            print(f"  [{i}/{len(club_urls)}] FAILED: {e}")

    stats_df = pd.DataFrame(all_rows)
    out = "data/raw/club_stats.csv"
    stats_df.to_csv(out, index=False)
    print(f"\nSaved {len(stats_df)} player-stats rows to {out}")


if __name__ == "__main__":
    main()
