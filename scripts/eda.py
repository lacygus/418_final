import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

OUT = "data/analysis"
os.makedirs(OUT, exist_ok=True)


def load() -> pd.DataFrame:
    df = pd.read_csv("data/processed/players_full.csv")
    df["mv_eur_m"] = df["market_value"] / 1e6
    return df


def overview(df: pd.DataFrame) -> None:
    print("=" * 50)
    print("OVERVIEW")
    print("=" * 50)
    print(f"Total players: {len(df)}")
    print(f"Leagues: {df['league'].nunique()}, Clubs: {df['club'].nunique()}")
    print(f"Position groups: {df['position'].nunique()}")
    print(f"With Sofifa attrs: {df['finishing'].notna().sum()}")
    print()
    print("Market value (EUR M):")
    print(df["mv_eur_m"].describe().round(2).to_string())
    print()


def plot_value_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["mv_eur_m"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Market Value (EUR M)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Market Value Distribution")

    axes[1].hist(np.log(df["mv_eur_m"]), bins=30, color="coral", edgecolor="white")
    axes[1].set_xlabel("log(Market Value)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Log Market Value (target for regression)")

    plt.tight_layout()
    plt.savefig(f"{OUT}/01_value_distribution.png", dpi=150)
    plt.close()


def plot_by_league(df: pd.DataFrame) -> None:
    league_stats = df.groupby("league")["mv_eur_m"].agg(["mean", "median", "count"]).round(1)
    print("\n=== by league ===")
    print(league_stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    league_order = league_stats.sort_values("mean", ascending=False).index
    df.boxplot(column="mv_eur_m", by="league", ax=ax)
    ax.set_xlabel("League")
    ax.set_ylabel("Market Value (EUR M)")
    ax.set_title("Market Value by League")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_by_league.png", dpi=150)
    plt.close()


def plot_by_position(df: pd.DataFrame) -> None:
    # group fine positions into broader categories
    def group_pos(p):
        p = str(p)
        if "Goalkeeper" in p:
            return "GK"
        if "Back" in p:
            return "DF"
        if "Centre-Back" in p:
            return "DF"
        if "Midfield" in p:
            return "MF"
        if "Winger" in p or "Forward" in p or "Striker" in p:
            return "FW"
        return "FW"

    df["pos_group"] = df["position"].apply(group_pos)
    pos_stats = df.groupby("pos_group")["mv_eur_m"].agg(["mean", "median", "count"]).round(1)
    print("\n=== by position group ===")
    print(pos_stats.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    df.boxplot(column="mv_eur_m", by="pos_group", ax=ax)
    ax.set_xlabel("Position Group")
    ax.set_ylabel("Market Value (EUR M)")
    ax.set_title("Market Value by Position")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_by_position.png", dpi=150)
    plt.close()


def plot_age_curve(df: pd.DataFrame) -> None:
    age_stats = df.groupby("age")["mv_eur_m"].agg(["mean", "median", "count"]).round(2)
    print("\n=== age curve ===")
    print(age_stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    valid = age_stats[age_stats["count"] >= 3]
    ax.plot(valid.index, valid["mean"], marker="o", color="steelblue", label="Mean")
    ax.plot(valid.index, valid["median"], marker="s", color="coral", label="Median")
    ax.set_xlabel("Age")
    ax.set_ylabel("Market Value (EUR M)")
    ax.set_title("Market Value vs Age")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_age_curve.png", dpi=150)
    plt.close()


def plot_stats_correlation(df: pd.DataFrame) -> None:
    cols = [
        "mv_eur_m", "age", "appearances", "goals", "assists", "minutes",
        "yellow_cards", "red_cards",
    ]
    corr = df[cols].corr()["mv_eur_m"].drop("mv_eur_m").sort_values()
    print("\n=== correlation with market value (basic stats) ===")
    print(corr.round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["coral" if v < 0 else "steelblue" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with market value")
    ax.set_title("Basic Stats vs Market Value")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_basic_corr.png", dpi=150)
    plt.close()


def plot_sofifa_correlation(df: pd.DataFrame) -> None:
    fifa_cols = [
        "crossing", "finishing", "heading_accuracy", "short_passing", "volleys",
        "dribbling", "curve", "fk_accuracy", "long_passing", "ball_control",
        "acceleration", "sprint_speed", "agility", "reactions", "balance",
        "shot_power", "jumping", "stamina", "strength", "long_shots",
        "aggression", "interceptions", "vision", "penalties", "composure",
        "defensive_awareness", "standing_tackle", "sliding_tackle",
    ]
    valid = df.dropna(subset=fifa_cols)
    corr = valid[fifa_cols + ["mv_eur_m"]].corr()["mv_eur_m"].drop("mv_eur_m").sort_values()
    print(f"\n=== top FIFA correlates (n={len(valid)}) ===")
    print(corr.tail(8).round(3).to_string())
    print("\n=== bottom FIFA correlates ===")
    print(corr.head(5).round(3).to_string())

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["coral" if v < 0 else "steelblue" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Correlation with market value")
    ax.set_title("FIFA Attributes vs Market Value")
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_sofifa_corr.png", dpi=150)
    plt.close()


def plot_top_features_scatter(df: pd.DataFrame) -> None:
    valid = df.dropna(subset=["reactions", "composure", "ball_control"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col in zip(axes, ["reactions", "ball_control", "composure"]):
        ax.scatter(valid[col], valid["mv_eur_m"], alpha=0.5, color="steelblue")
        ax.set_xlabel(col)
        ax.set_ylabel("Market Value (EUR M)")
        r = valid[col].corr(valid["mv_eur_m"])
        ax.set_title(f"{col} vs MV (r={r:.2f})")

    plt.tight_layout()
    plt.savefig(f"{OUT}/07_top_features_scatter.png", dpi=150)
    plt.close()


def main() -> None:
    df = load()
    overview(df)
    plot_value_distribution(df)
    plot_by_league(df)
    plot_by_position(df)
    plot_age_curve(df)
    plot_stats_correlation(df)
    plot_sofifa_correlation(df)
    plot_top_features_scatter(df)
    print(f"\nSaved 7 plots to {OUT}/")


if __name__ == "__main__":
    main()
