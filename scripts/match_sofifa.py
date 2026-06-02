import logging
import os
import re

import pandas as pd

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/match_sofifa.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def clean_sofifa_name(name: str) -> str:
    """Strip trailing position abbreviations like 'STLW' from Sofifa names."""
    return re.sub(r"(ST|CF|LW|RW|LM|RM|CAM|CDM|CM|LB|RB|CB|LWB|RWB|GK)+$", "", name).strip()


def make_match_key(name: str) -> str:
    """Build a 'first_initial + surname' lowercase key for fuzzy matching."""
    parts = name.replace(".", "").split()
    if not parts:
        return ""
    first = parts[0][0].lower()
    surname = parts[-1].lower()
    return f"{first}_{surname}"


def main() -> None:
    tm = pd.read_csv("data/processed/players.csv")
    sf = pd.read_csv("data/raw/sofifa_ids.csv")

    sf["name_clean"] = sf["sofifa_name"].apply(clean_sofifa_name)
    sf["match_key"] = sf["name_clean"].apply(make_match_key)
    tm["match_key"] = tm["player"].apply(make_match_key)

    # Drop dup keys in sf (keep first = highest overall)
    sf_unique = sf.drop_duplicates("match_key", keep="first")

    matched = tm.merge(sf_unique[["match_key", "sofifa_id", "sofifa_url"]], on="match_key", how="left")
    n_matched = matched["sofifa_id"].notna().sum()
    print(f"TMK players: {len(tm)}")
    print(f"Matched: {n_matched} ({n_matched/len(tm)*100:.0f}%)")

    out = "data/processed/players_with_sofifa.csv"
    matched.to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
