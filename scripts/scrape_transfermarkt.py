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
    filename="logs/scrape_transfermarkt.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

LEAGUES = {
    "EPL": ("premier-league", "GB1"),
    "LaLiga": ("laliga", "ES1"),
    "Bundesliga": ("bundesliga", "L1"),
    "SerieA": ("serie-a", "IT1"),
    "Ligue1": ("ligue-1", "FR1"),
}

DELAY = 3.0
PAGES_PER_LEAGUE = 4  # 25 players per page -- top 100 per league


def parse_value(text: str) -> float | None:
    """Convert market value string ('€50.00m', '€500k') to euros."""
    text = text.strip()
    m = re.search(r"€([\d.]+)\s*(m|k)?", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "m":
        return val * 1_000_000
    if unit == "k":
        return val * 1_000
    return val


def scrape_page(slug: str, code: str, page: int) -> List[Dict]:
    """Scrape one market-value page for a league."""
    url = f"https://www.transfermarkt.com/{slug}/marktwerte/wettbewerb/{code}/page/{page}"
    time.sleep(DELAY)
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")

    table = soup.find("table", class_="items")
    if not table:
        logging.warning("no table on %s", url)
        return []

    rows = []
    for tr in table.find_all("tr", class_=["odd", "even"]):
        cells = tr.find_all("td")
        if len(cells) < 9:
            continue

        name = cells[3].get_text(strip=True)
        position = cells[4].get_text(strip=True)
        age = cells[6].get_text(strip=True)

        link = cells[3].find("a", href=True)
        player_url = "https://www.transfermarkt.com" + link["href"] if link else ""

        nat_imgs = cells[5].find_all("img")
        nationality = nat_imgs[0].get("title", "") if nat_imgs else ""

        club_img = cells[7].find("img")
        club = club_img.get("alt", "") if club_img else cells[7].get_text(strip=True)

        club_link = cells[7].find("a", href=True)
        club_url = "https://www.transfermarkt.com" + club_link["href"] if club_link else ""

        value = parse_value(cells[8].get_text(strip=True))

        rows.append({
            "player": name,
            "position": position,
            "age": age,
            "nationality": nationality,
            "club": club,
            "market_value": value,
            "player_url": player_url,
            "club_url": club_url,
        })

    logging.info("page %d of %s: %d players", page, code, len(rows))
    return rows


def scrape_league(name: str, slug: str, code: str) -> pd.DataFrame:
    """Scrape multiple pages for one league."""
    all_rows = []
    for page in range(1, PAGES_PER_LEAGUE + 1):
        try:
            rows = scrape_page(slug, code, page)
            if not rows:
                break
            all_rows.extend(rows)
            print(f"  {name} page {page}: {len(rows)} players")
        except Exception as e:
            logging.error("%s page %d failed: %s", name, page, e)
            print(f"  {name} page {page} FAILED: {e}")

    df = pd.DataFrame(all_rows)
    df["league"] = name
    return df


def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    all_dfs = []
    for name, (slug, code) in LEAGUES.items():
        print(f"\nScraping {name}...")
        df = scrape_league(name, slug, code)
        df.to_csv(f"data/raw/tm_{name}.csv", index=False)
        all_dfs.append(df)

    if all_dfs:
        merged = pd.concat(all_dfs, ignore_index=True)
        out = "data/raw/transfermarkt_all.csv"
        merged.to_csv(out, index=False)
        print(f"\nSaved {len(merged)} total players to {out}")


if __name__ == "__main__":
    main()
