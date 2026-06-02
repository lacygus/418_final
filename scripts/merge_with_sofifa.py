import logging
import os

import pandas as pd

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/merge_with_sofifa.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    """Merge TMK + club stats with Sofifa attributes on sofifa_id."""
    base = pd.read_csv("data/processed/players_with_sofifa.csv")
    attrs = pd.read_csv("data/raw/sofifa_attributes.csv")

    df = base.merge(attrs, on="sofifa_id", how="left")
    df = df.drop_duplicates(subset="player", keep="first")
    n_full = df["finishing"].notna().sum() if "finishing" in df.columns else 0
    print(f"Total players: {len(df)}")
    print(f"With Sofifa attributes: {n_full}")
    print(f"Total columns: {len(df.columns)}")

    out = "data/processed/players_full.csv"
    df.to_csv(out, index=False)
    logging.info("merged %d rows, %d with attrs", len(df), n_full)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
