import logging
import os

import pandas as pd
import requests

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/add_usd.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def fetch_eur_usd_rate() -> float:
    """Fetch the current EUR/USD exchange rate."""
    r = requests.get("https://open.er-api.com/v6/latest/EUR", timeout=10)
    r.raise_for_status()
    rate = r.json()["rates"]["USD"]
    logging.info("EUR/USD rate: %.4f", rate)
    return rate


def main() -> None:
    rate = fetch_eur_usd_rate()
    print(f"EUR/USD rate: {rate:.4f}")

    df = pd.read_csv("data/raw/transfermarkt_all.csv")
    df["market_value_usd"] = (df["market_value"] * rate).round(2)
    df["eur_usd_rate"] = rate

    out = "data/raw/transfermarkt_all.csv"
    df.to_csv(out, index=False)
    logging.info("added market_value_usd to %d rows", len(df))
    print(f"Updated {len(df)} rows in {out}")
    print(f"Top value: EUR {df['market_value'].max()/1e6:.0f}M = USD {df['market_value_usd'].max()/1e6:.2f}M")


if __name__ == "__main__":
    main()
