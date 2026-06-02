"""Add club_id and logo_url columns to the model dataset.

Club IDs come from the Transfermarkt club URL pattern: /verein/<ID>.
Logo URLs follow https://tmssl.akamaized.net/images/wappen/medium/<ID>.png.
"""
import io
import os
import re
import sys

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

LOGO_URL = "https://tmssl.akamaized.net/images/wappen/medium/{cid}.png"


def main() -> None:
    """Add club_id and logo_url to model_dataset.csv."""
    df = pd.read_csv("data/processed/model_dataset.csv")
    tm = pd.read_csv("data/raw/transfermarkt_all.csv")

    # Extract club_id from URL like /verein/281/...
    tm["club_id"] = tm["club_url"].str.extract(r"/verein/(\d+)")
    tm_map = tm[["club", "club_id"]].drop_duplicates("club").dropna()

    # Also derive club_id from squad_values rows if we scraped club_url there
    out = df.merge(tm_map, on="club", how="left")

    # Fall back: for clubs not in tm_map, try squad_values.csv
    if out["club_id"].isnull().any():
        sv = pd.read_csv("data/raw/squad_values.csv")
        if "club_url" in sv.columns:
            sv["club_id"] = sv["club_url"].str.extract(r"/verein/(\d+)")
            sv_map = sv[["club", "club_id"]].drop_duplicates("club").dropna()
            missing = out["club_id"].isnull()
            patch = out.loc[missing, ["club"]].merge(sv_map, on="club", how="left")
            out.loc[missing, "club_id"] = patch["club_id"].values

    out["logo_url"] = out["club_id"].apply(
        lambda cid: LOGO_URL.format(cid=cid) if pd.notna(cid) else ""
    )

    n_with_logo = (out["logo_url"] != "").sum()
    out.to_csv("data/processed/model_dataset.csv", index=False)
    print(f"Added club_id and logo_url to {len(out)} rows ({n_with_logo} with logo).")


if __name__ == "__main__":
    main()
