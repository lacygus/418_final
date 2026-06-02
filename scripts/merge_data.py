import logging
import os

import pandas as pd

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/merge_data.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    """Merge market value data with squad stats by player name."""
    market = pd.read_csv("data/raw/transfermarkt_all.csv")
    stats = pd.read_csv("data/raw/club_stats.csv")

    # Aggregate stats in case a player appears in multiple clubs (transfers)
    stats_agg = stats.groupby("player", as_index=False).agg({
        "appearances": "sum",
        "goals": "sum",
        "assists": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "minutes": "sum",
    })

    df = market.merge(stats_agg, on="player", how="left")
    matched = df["appearances"].notna().sum()
    logging.info("merged: %d total, %d matched on stats", len(df), matched)

    out = "data/processed/players.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Merged {len(df)} players, {matched} with stats")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
