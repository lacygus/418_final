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
    filename="logs/scrape_squad_values.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

DELAY = 3.0


def parse_value(text: str) -> float | None:
    """Convert a market value string ('€45.00m', '€500k') to euros."""
    m = re.search(r"€([\d.]+)\s*(m|k)?", text.strip(), re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "m":
        return val * 1_000_000
    if unit == "k":
        return val * 1_000
    return val


def parse_age(text: str) -> int | None:
    """Extract age from a 'DD/MM/YYYY (AA)' birth-date cell."""
    m = re.search(r"\((\d+)\)", text)
    return int(m.group(1)) if m else None


def scrape_squad(url: str, club: str, league: str) -> List[Dict]:
    """Scrape one club's squad page for player name, position, age, and value."""
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    table = soup.find("table", class_="items")
    if not table:
        return []

    rows = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        cells = tr.find_all("td", recursive=False)
        if len(cells) < 5:
            continue

        name_tag = tr.find("td", class_="hauptlink")
        if not name_tag:
            continue
        a = name_tag.find("a", title=True)
        name = a["title"] if a else name_tag.get_text(strip=True)

        pos_table = cells[1].find("table")
        position = ""
        if pos_table:
            trs = pos_table.find_all("tr")
            if len(trs) > 1:
                position = trs[1].get_text(strip=True)

        age = parse_age(cells[2].get_text(strip=True))
        value = parse_value(cells[4].get_text(strip=True))

        rows.append({
            "player": name,
            "position": position,
            "age": age,
            "club": club,
            "league": league,
            "market_value": value,
        })

    logging.info("scraped %d players from %s", len(rows), url)
    return rows


def main() -> None:
    ref = pd.read_csv("data/raw/transfermarkt_all.csv")
    clubs = ref[["club", "league", "club_url"]].drop_duplicates("club_url")
    print(f"Scraping {len(clubs)} club squads...")

    all_rows = []
    for i, row in enumerate(clubs.itertuples(index=False), 1):
        try:
            rows = scrape_squad(row.club_url, row.club, row.league)
            all_rows.extend(rows)
            print(f"  [{i}/{len(clubs)}] {row.club}: {len(rows)} players")
        except Exception as e:
            logging.error("failed %s: %s", row.club, e)
            print(f"  [{i}] {row.club} FAILED: {e}")

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["market_value"])
    out = "data/raw/squad_values.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df)} players with values to {out}")
    print(f"Value range: EUR {df['market_value'].min()/1e6:.2f}M - {df['market_value'].max()/1e6:.0f}M")


if __name__ == "__main__":
    main()
