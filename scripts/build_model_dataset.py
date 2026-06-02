import io
import logging
import os
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/build_model_dataset.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    """Merge full-range squad values with squad performance stats."""
    values = pd.read_csv("data/raw/squad_values.csv")
    stats = pd.read_csv("data/raw/club_stats.csv")

    # club_stats may list a player at multiple clubs after transfers -- aggregate
    stats_agg = stats.groupby("player", as_index=False).agg({
        "appearances": "sum",
        "goals": "sum",
        "assists": "sum",
        "yellow_cards": "sum",
        "red_cards": "sum",
        "minutes": "sum",
    })

    df = values.merge(stats_agg, on="player", how="inner")
    df = df.dropna(subset=["age", "market_value"])
    df = df.drop_duplicates(subset="player")

    out = "data/processed/model_dataset.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Merged dataset: {len(df)} players")
    print(f"Value range: EUR {df['market_value'].min()/1e6:.2f}M - {df['market_value'].max()/1e6:.0f}M")
    print(f"Saved to {out}")
    logging.info("built model dataset: %d rows", len(df))


if __name__ == "__main__":
    main()
