"""Streamlit scouting tool for soccer player market value prediction.

Five tabs:
  1. Scout a Player    - predict + SHAP + similar players (search supported)
  2. Market Movers     - most over- and under-valued players, with filters
  3. Leaderboards      - top players, clubs, positions
  4. Model Performance - predicted vs actual diagnostics
  5. About             - methodology, data sources, limitations
"""
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score

API_URL = os.environ.get("API_URL", "https://market-value-api-348858993647.us-central1.run.app")
# This API key is intentionally bundled — it's a class-project key, already
# disclosed in the course submission. Streamlit Cloud users can override it
# via `st.secrets["API_KEY"]` if they want their own.
DEFAULT_API_KEY = "mjyapuBCPAtnQlU41gx7K8XfIhRTrN9i"
DATA_PATH = "data/processed/model_dataset.csv"
MODEL_PATH = "model.pkl"
DATA_PERIOD = "2025-26 season (Transfermarkt snapshot, scraped late 2025)"

# Fallback FX rate if the live fetch fails. Updated alongside data scrapes.
EUR_USD_FALLBACK = 1.17


def get_api_key() -> str:
    """Read API key from Streamlit secrets if set, otherwise use the bundled default."""
    try:
        return st.secrets.get("API_KEY", DEFAULT_API_KEY)
    except (FileNotFoundError, KeyError):
        return DEFAULT_API_KEY

FEATURE_LABELS = {
    "age": "Age",
    "appearances": "Matches played",
    "goals": "Goals",
    "assists": "Assists",
    "minutes": "Minutes played",
    "yellow_cards": "Yellow Cards",
    "red_cards": "Red Cards",
}

FEATURE_HELP = {
    "age": "Player age in years.",
    "appearances": (
        "Total matches played across all competitions this season "
        "(league + domestic cups + European competitions + national team). "
        "A full league season is ~38 matches (EPL/La Liga/Serie A) or ~34 (Bundesliga/Ligue 1). "
        "Top players who play everywhere reach 50-80 appearances."
    ),
    "goals": "Goals scored across all competitions this season.",
    "assists": "Assists across all competitions this season.",
    "minutes": (
        "Total minutes on the field across all competitions this season. "
        "Reference numbers:\n"
        "• Full league season as a starter: ~3,420 min (38 × 90) for EPL, La Liga, Serie A\n"
        "• Plus a deep cup run + Champions League + internationals: ~5,500-5,800 min\n"
        "• Our dataset max is 5,831 min — basically every minute of every game."
    ),
    "yellow_cards": "Yellow cards received across all competitions.",
    "red_cards": "Red cards received across all competitions.",
}

POSITION_GROUPS = ["GK", "DF", "MF", "FW"]

st.set_page_config(
    page_title="Player Market Value Prediction",
    page_icon="📈",
    layout="wide",
)


# --------------------------- loaders & helpers ---------------------------

@st.cache_resource
def load_model_bundle():
    """Load model bundle and SHAP explainer once and reuse across reruns."""
    bundle = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


@st.cache_data(ttl=3600)
def fetch_eur_usd_rate() -> float:
    """Fetch the current EUR/USD rate, falling back to a fixed rate on failure."""
    try:
        r = requests.get("https://open.er-api.com/v6/latest/EUR", timeout=5)
        r.raise_for_status()
        return float(r.json()["rates"]["USD"])
    except Exception:
        return EUR_USD_FALLBACK


def position_group(pos: str) -> str:
    """Collapse fine-grained positions into GK/DF/MF/FW."""
    p = str(pos)
    if "Goalkeeper" in p:
        return "GK"
    if "Back" in p:
        return "DF"
    if "Midfield" in p:
        return "MF"
    return "FW"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the player dataset and tag a coarse position group."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["age", "market_value"])
    df["pos_group"] = df["position"].apply(position_group)
    return df


@st.cache_data
def compute_all_predictions(_bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Predict market value for every player in the dataset (cached)."""
    features = _bundle["features"]
    X = df[features].values
    log_pred = _bundle["model"].predict(X)
    out = df.copy()
    out["predicted_value"] = np.exp(log_pred)
    out["delta"] = out["predicted_value"] - out["market_value"]
    out["ratio"] = out["predicted_value"] / out["market_value"]
    return out


def predict_with_interval(bundle, features: List[float]) -> Tuple[float, float, float, float]:
    """Return point prediction plus an 80% prediction interval and confidence.

    The interval comes from the spread of the individual trees' predictions
    in log space: the 10th and 90th percentiles, exponentiated back.
    Confidence collapses tree disagreement to one number in (0, 1].
    """
    model = bundle["model"]
    X = np.asarray(features, dtype=float).reshape(1, -1)
    tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
    log_pred = float(tree_preds.mean())
    log_lo = float(np.percentile(tree_preds, 10))
    log_hi = float(np.percentile(tree_preds, 90))
    confidence = float(np.exp(-float(tree_preds.std())))
    return float(np.exp(log_pred)), float(np.exp(log_lo)), float(np.exp(log_hi)), confidence


def predict_api(features: List[float], api_key: str) -> dict | None:
    """Call the deployed Cloud Run API. Returns None on failure."""
    try:
        r = requests.post(
            f"{API_URL}/v1/predict",
            json={"features": features},
            headers={"api-key": api_key},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"{r.status_code} {r.text[:120]}"}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=300)
def predict_api_cached(features_tuple: tuple) -> dict:
    """Cached wrapper around predict_api so identical inputs hit the API once."""
    return predict_api(list(features_tuple), get_api_key())


def shap_contributions(explainer, features: List[float], feature_names: List[str]):
    """Return per-feature contributions in EUR plus base and predicted EUR."""
    X = np.asarray(features, dtype=float).reshape(1, -1)
    sv = explainer.shap_values(X)[0]
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.asarray(base).flatten()[0])

    log_pred = base + sv.sum()
    pred_eur = float(np.exp(log_pred))
    base_eur = float(np.exp(base))

    rows = []
    for name, val, raw in zip(feature_names, sv, features):
        delta_eur = float(np.exp(base + val) - base_eur)
        rows.append({
            "feature": FEATURE_LABELS.get(name, name),
            "value": raw,
            "shap_log": val,
            "delta_eur": delta_eur,
        })
    df = pd.DataFrame(rows)
    df = df.reindex(df["delta_eur"].abs().sort_values(ascending=False).index)
    return df, base_eur, pred_eur


def find_similar(df: pd.DataFrame, features: List[float], feature_names: List[str], k: int = 5) -> pd.DataFrame:
    """Return the k players closest to the input vector by z-score distance."""
    X = df[feature_names].values
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xz = (X - mu) / sd
    qz = (np.asarray(features) - mu) / sd
    d = np.sqrt(((Xz - qz) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    base = ["logo_url"] if "logo_url" in df.columns else []
    cols = base + ["player", "club", "league", "age", "goals", "assists", "minutes", "market_value"]
    return df.iloc[idx][cols].reset_index(drop=True)


_FX_RATE = EUR_USD_FALLBACK  # overwritten in main()


def fmt_money(eur_value: float) -> str:
    """Format a EUR value as USD millions ($X.YM) using the current FX rate."""
    return f"${eur_value * _FX_RATE / 1e6:.1f}M"


def out_of_range_warning(features: List[float], df: pd.DataFrame, feature_names: List[str]) -> List[str]:
    """Flag features that fall outside the trained data range."""
    warnings = []
    for name, val in zip(feature_names, features):
        lo, hi = df[name].min(), df[name].max()
        if val < lo or val > hi:
            warnings.append(f"{FEATURE_LABELS[name]} = {val:.0f} is outside training range [{lo:.0f}, {hi:.0f}]")
    return warnings


# --------------------------- Tab 1: Scout ---------------------------

def tab_scout(df: pd.DataFrame, bundle, explainer):
    """Single-player scouting view with search."""
    features = bundle["features"]

    st.sidebar.markdown("### 1. Choose a player")
    mode = st.sidebar.radio("Mode", ["Pick a player", "Custom stats"], label_visibility="collapsed")

    if mode == "Pick a player":
        search = st.sidebar.text_input("Search by name", placeholder="e.g. Haaland, Mbappe").strip()
        if search:
            matches = df[df["player"].str.contains(search, case=False, na=False)]
            matches = matches.sort_values("market_value", ascending=False)
            st.sidebar.caption(f"{len(matches)} matches across all leagues")
            if matches.empty:
                st.sidebar.error("No players match that search.")
                st.stop()
            league_df = matches
        else:
            leagues = sorted(df["league"].unique())
            league = st.sidebar.selectbox("League", leagues, index=leagues.index("EPL") if "EPL" in leagues else 0)
            league_df = df[df["league"] == league].sort_values("market_value", ascending=False)
            st.sidebar.caption(f"{len(league_df)} players in {league}")

        labels = [f"{r.player} ({r.club}, {r.league}) — {fmt_money(r.market_value)}" for r in league_df.itertuples(index=False)]
        sel = st.sidebar.selectbox("Player", labels)
        row = league_df.iloc[labels.index(sel)]
        input_features = [float(row[f]) for f in features]
        actual_value = float(row["market_value"])
        player_name = row["player"]
        player_meta = f"{row['club']} · {row['position']} · {row['league']}"
    else:
        st.sidebar.markdown("Move the sliders to build a player profile:")
        with st.sidebar.expander("What do these stats mean?", expanded=False):
            st.markdown(
                "All counts are across **every competition** this season — league, "
                "domestic cup, European competition, and national team. Useful "
                "reference points:\n\n"
                "| Profile | Matches | Minutes |\n"
                "|---|---:|---:|\n"
                "| Full league season as a starter | ~38 | ~3,420 |\n"
                "| Adds domestic cup runs | ~45 | ~4,000 |\n"
                "| Adds Champions League run | ~55 | ~5,000 |\n"
                "| Adds national team duty | ~65 | ~5,500 |\n"
                "| Plays absolutely everything (dataset max) | 80 | 5,831 |\n\n"
                "A bench player on a non-European club might sit around "
                "10-15 appearances and 600-1,200 minutes."
            )
        defaults = {"age": 25, "appearances": 30, "goals": 5, "assists": 3, "minutes": 2400, "yellow_cards": 4, "red_cards": 0}
        ranges = {"age": (16, 42, 1), "appearances": (0, 80, 1), "goals": (0, 55, 1), "assists": (0, 30, 1),
                  "minutes": (0, 6000, 50), "yellow_cards": (0, 15, 1), "red_cards": (0, 4, 1)}
        input_features = []
        for f in features:
            lo, hi, step = ranges[f]
            input_features.append(float(st.sidebar.slider(
                FEATURE_LABELS[f], lo, hi, defaults[f], step, help=FEATURE_HELP[f]
            )))
        actual_value = None
        player_name = "Custom player"
        player_meta = "Manual input"

    pred_value, _lo, _hi, _conf = predict_with_interval(bundle, input_features)
    shap_df, base_eur, _ = shap_contributions(explainer, input_features, features)

    # Verify the same input against the deployed Cloud Run API.
    api_result = predict_api_cached(tuple(input_features))

    warnings = out_of_range_warning(input_features, df, features)
    if warnings:
        st.warning("Some inputs are outside the training data range — predictions are extrapolations:\n\n- " + "\n- ".join(warnings))

    st.header(player_name)
    st.caption(player_meta)
    if actual_value is not None:
        c1, c2 = st.columns(2)
        c1.metric("Predicted value", fmt_money(pred_value))
        delta = pred_value - actual_value
        sign = "+" if delta >= 0 else ""
        c2.metric("Actual value", fmt_money(actual_value),
                  delta=f"{sign}{fmt_money(delta)} predicted - actual")
    else:
        median_value = float(df["market_value"].median())
        delta = pred_value - median_value
        sign = "+" if delta >= 0 else ""
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted value", fmt_money(pred_value))
        c2.metric("Dataset median",
                  fmt_money(median_value),
                  delta=f"{sign}{fmt_money(delta)} vs median",
                  delta_color="off",
                  help=f"Median market value across all {len(df):,} players.")
        c3.metric("Dataset mean",
                  fmt_money(float(df["market_value"].mean())),
                  help=f"Mean market value across all {len(df):,} players.")

    # ---- API parity check (auto, in-page) ----
    if api_result and "prediction" in api_result:
        api_val = float(api_result["prediction"])
        match = abs(api_val - pred_value) <= max(pred_value * 0.001, 100.0)
        req_id = str(api_result.get("request_id", ""))[:8]
        if match:
            st.success(
                f"Verified against deployed Cloud Run API — same input returns "
                f"**{fmt_money(api_val)}** "
                f"({API_URL.split('//')[1]} · request `{req_id}`).",
                icon="✅",
            )
        else:
            st.warning(
                f"API returned **{fmt_money(api_val)}** but the in-app model returned "
                f"**{fmt_money(pred_value)}**. They should normally agree.",
                icon="⚠️",
            )
    elif api_result and "error" in api_result:
        st.info(
            f"Deployed API not reachable right now ({api_result['error']}). "
            f"In-app prediction shown above.",
            icon="ℹ️",
        )

    st.subheader("What's driving this value?")
    st.caption("Bars to the right pull the value up; bars to the left pull it down.")
    chart_df = shap_df.copy()
    chart_df["delta_m"] = chart_df["delta_eur"] * _FX_RATE / 1e6
    chart_df["label"] = chart_df["feature"] + " = " + chart_df["value"].astype(str)
    fig = px.bar(
        chart_df.iloc[::-1], x="delta_m", y="label", orientation="h",
        color="delta_m", color_continuous_scale=["#d9534f", "#dddddd", "#1f77b4"],
        color_continuous_midpoint=0, labels={"delta_m": "Contribution ($M)", "label": ""},
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350,
                      margin=dict(l=0, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where does this rank?")
    pct = float((df["market_value"] < pred_value).mean() * 100)
    st.caption(f"Sits at the **{pct:.0f}th percentile** of {len(df):,} players in the dataset.")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(x=df["market_value"] * _FX_RATE / 1e6, nbinsx=60, marker_color="#cbd5e1"))
    hist_fig.add_vline(x=pred_value * _FX_RATE / 1e6, line_color="#1f77b4", line_width=3, annotation_text="Predicted")
    if actual_value is not None:
        hist_fig.add_vline(x=actual_value * _FX_RATE / 1e6, line_color="#d9534f", line_dash="dash", line_width=2, annotation_text="Actual")
    hist_fig.update_layout(xaxis_title="Market value ($M USD)", yaxis_title="Players", height=300,
                           showlegend=False, margin=dict(l=0, r=10, t=10, b=10))
    st.plotly_chart(hist_fig, use_container_width=True)

    st.subheader("Most similar players in the dataset")
    sim = find_similar(df, input_features, features, k=5)
    sim["market_value"] = sim["market_value"].apply(fmt_money)
    has_logo = "logo_url" in sim.columns
    if has_logo:
        sim.columns = ["", "Player", "Club", "League", "Age", "Goals", "Assists", "Minutes", "Market Value"]
        st.dataframe(
            sim, hide_index=True, use_container_width=True,
            column_config={"": st.column_config.ImageColumn("", width="small")},
        )
    else:
        sim.columns = ["Player", "Club", "League", "Age", "Goals", "Assists", "Minutes", "Market Value"]
        st.dataframe(sim, hide_index=True, use_container_width=True)


# --------------------------- Tab 2: Market Movers ---------------------------

def tab_movers(df_pred: pd.DataFrame):
    """Show the players the model thinks are most mispriced."""
    st.subheader("Most mispriced players")
    st.caption(
        "Predicted minus actual market value. Positive deltas suggest the "
        "model thinks the player is undervalued. Use the filters to compare "
        "like with like."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        leagues = ["All"] + sorted(df_pred["league"].unique())
        lg = st.selectbox("League", leagues, key="movers_league")
    with c2:
        pos = st.selectbox("Position group", ["All"] + POSITION_GROUPS, key="movers_pos")
    with c3:
        min_mins = st.slider("Minimum minutes", 0, 4000, 900, step=100, key="movers_mins")

    view = df_pred.copy()
    if lg != "All":
        view = view[view["league"] == lg]
    if pos != "All":
        view = view[view["pos_group"] == pos]
    view = view[view["minutes"] >= min_mins]

    st.caption(f"Filtered set: **{len(view):,} players**.")

    has_logo = "logo_url" in view.columns

    def fmt_table(rows: pd.DataFrame) -> pd.DataFrame:
        base = ["logo_url"] if has_logo else []
        out = rows[base + ["player", "club", "league", "pos_group", "age",
                           "market_value", "predicted_value", "delta"]].copy()
        out["market_value"] = out["market_value"].apply(fmt_money)
        out["predicted_value"] = out["predicted_value"].apply(fmt_money)
        out["delta"] = out["delta"].apply(fmt_money)
        if has_logo:
            out.columns = ["", "Player", "Club", "League", "Pos", "Age", "Actual", "Predicted", "Delta"]
        else:
            out.columns = ["Player", "Club", "League", "Pos", "Age", "Actual", "Predicted", "Delta"]
        return out

    col_cfg = {"": st.column_config.ImageColumn("", width="small")} if has_logo else None
    col_u, col_o = st.columns(2)
    with col_u:
        st.markdown("**Top 10 undervalued** &nbsp;·&nbsp; predicted &gt; actual")
        st.dataframe(fmt_table(view.nlargest(10, "delta")), hide_index=True,
                     use_container_width=True, column_config=col_cfg)
    with col_o:
        st.markdown("**Top 10 overvalued** &nbsp;·&nbsp; predicted &lt; actual")
        st.dataframe(fmt_table(view.nsmallest(10, "delta")), hide_index=True,
                     use_container_width=True, column_config=col_cfg)

    st.subheader("Distribution of pricing gaps")
    fig = px.histogram(view, x=(view["delta"] * _FX_RATE / 1e6), nbins=60,
                       labels={"x": "Predicted - Actual ($M USD)"})
    fig.update_traces(marker_color="#1f77b4")
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=10), yaxis_title="Players")
    st.plotly_chart(fig, use_container_width=True)

    csv = view[["player", "club", "league", "pos_group", "age", "market_value", "predicted_value", "delta"]].to_csv(index=False)
    st.download_button("Download filtered predictions (CSV)", csv,
                       file_name="market_movers.csv", mime="text/csv")


# --------------------------- Tab 3: Leaderboards ---------------------------

def tab_leaderboards(df_pred: pd.DataFrame):
    """League, club and position leaderboards."""
    leagues = sorted(df_pred["league"].unique())
    has_logo = "logo_url" in df_pred.columns

    st.subheader("Top 10 by actual market value")
    cols = st.columns(len(leagues))
    for col, lg in zip(cols, leagues):
        with col:
            st.markdown(f"**{lg}**")
            sub = df_pred[df_pred["league"] == lg].nlargest(10, "market_value")
            display_cols = (["logo_url"] if has_logo else []) + ["player", "club", "market_value"]
            top = sub[display_cols].copy()
            top["market_value"] = top["market_value"].apply(fmt_money)
            if has_logo:
                top.columns = ["", "Player", "Club", "Value"]
                st.dataframe(
                    top, hide_index=True, use_container_width=True,
                    column_config={"": st.column_config.ImageColumn("", width="small")},
                )
            else:
                top.columns = ["Player", "Club", "Value"]
                st.dataframe(top, hide_index=True, use_container_width=True)

    st.subheader("Average market value by league")
    by_league = (
        df_pred.groupby("league")
        .agg(mean=("market_value", "mean"), median=("market_value", "median"), n=("player", "count"))
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    by_league["mean_m"] = by_league["mean"] * _FX_RATE / 1e6
    by_league["median_m"] = by_league["median"] * _FX_RATE / 1e6
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Mean", x=by_league["league"], y=by_league["mean_m"], marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Median", x=by_league["league"], y=by_league["median_m"], marker_color="#9ec5e8"))
    fig.update_layout(
        barmode="group", height=320, margin=dict(l=0, r=10, t=10, b=10),
        yaxis_title="Market value ($M USD)", xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("EPL typically sits well above the other four leagues at every percentile.")

    st.subheader("League × Position heatmap (mean market value)")
    grid = (
        df_pred.groupby(["league", "pos_group"])["market_value"].mean()
        .reset_index()
    )
    grid["value_m"] = grid["market_value"] * _FX_RATE / 1e6
    pivot = grid.pivot(index="pos_group", columns="league", values="value_m")
    pivot = pivot.reindex(["FW", "MF", "DF", "GK"])
    fig = px.imshow(
        pivot, text_auto=".1f", aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="League", y="Position", color="$M USD"),
    )
    fig.update_layout(height=320, margin=dict(l=0, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Read each cell as: mean market value of that position group in that league.")

    st.subheader("Most valuable clubs (sum of squad market value)")
    clubs = (
        df_pred.groupby(["club", "league"])
        .agg(squad_value=("market_value", "sum"), n=("player", "count"),
             logo_url=("logo_url", "first") if has_logo else ("club", "first"))
        .reset_index()
        .nlargest(15, "squad_value")
    )
    clubs["squad_value_m"] = clubs["squad_value"] * _FX_RATE / 1e6
    fig = px.bar(
        clubs.iloc[::-1], x="squad_value_m", y="club", color="league",
        orientation="h", labels={"squad_value_m": "Squad value ($M USD)", "club": ""},
        hover_data={"n": True, "squad_value_m": ":.1f"},
    )
    fig.update_layout(height=450, margin=dict(l=0, r=10, t=10, b=10), legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    if has_logo:
        st.markdown("**Top 15 squads**")
        clubs_display = clubs[["logo_url", "club", "league", "n", "squad_value_m"]].copy()
        clubs_display["squad_value_m"] = clubs_display["squad_value_m"].apply(lambda v: f"${v:.0f}M")
        clubs_display.columns = ["", "Club", "League", "Players", "Squad value"]
        st.dataframe(
            clubs_display, hide_index=True, use_container_width=True,
            column_config={"": st.column_config.ImageColumn("", width="small")},
        )

    csv = df_pred[["player", "club", "league", "pos_group", "age", "market_value", "predicted_value"]].to_csv(index=False)
    st.download_button("Download full predictions (CSV)", csv,
                       file_name="all_predictions.csv", mime="text/csv")


# --------------------------- Tab 4: Model Performance ---------------------------

def tab_model_perf(df_pred: pd.DataFrame, bundle):
    """Predicted vs actual and residual diagnostics."""
    st.subheader("How accurate is the model?")
    metrics = bundle["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Test R² (log target)", f"{metrics['r2']:.2f}")
    c2.metric("Test MAE", fmt_money(metrics["mae_eur"]))
    c3.metric("Training rows", f"{bundle['n_train']:,}")

    st.caption(
        "Metrics above are from a held-out 20% test split. The scatter below "
        "is the trained model's predictions on the full dataset — points near "
        "the diagonal are well predicted; far from it are surprises."
    )

    leagues = sorted(df_pred["league"].unique())
    selected = st.multiselect(
        "Show leagues",
        leagues, default=leagues,
        help="Uncheck a league to focus on the others. Empty = show everything.",
    )
    pool = df_pred[df_pred["league"].isin(selected)] if selected else df_pred
    sample = pool.sample(min(len(pool), 2000), random_state=42)

    fig = px.scatter(
        sample, x=sample["market_value"] * _FX_RATE / 1e6, y=sample["predicted_value"] * _FX_RATE / 1e6,
        color="league", hover_data={"player": True, "club": True, "age": True},
        labels={"x": "Actual market value ($M USD)", "y": "Predicted market value ($M USD)"},
        opacity=0.6,
    )
    diag = max(sample["market_value"].max(), sample["predicted_value"].max()) * _FX_RATE / 1e6
    fig.add_shape(type="line", x0=0, y0=0, x1=diag, y1=diag,
                  line=dict(color="#FFD60A", dash="dash", width=3))
    fig.update_layout(
        height=500, margin=dict(l=0, r=10, t=10, b=10), legend_title="",
        legend=dict(itemclick="toggleothers", itemdoubleclick="toggle"),
    )
    st.caption(
        f"Showing {len(sample):,} of {len(pool):,} filtered players. "
        "Click a league in the legend to isolate it (click again to restore)."
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Residuals (predicted - actual)")
    residuals = (df_pred["predicted_value"] - df_pred["market_value"]) * _FX_RATE / 1e6
    fig = px.histogram(x=residuals, nbins=80, labels={"x": "Residual ($M USD)"})
    fig.update_traces(marker_color="#1f77b4")
    fig.add_vline(x=0, line_color="#FFD60A", line_dash="dash", line_width=3)
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=10), yaxis_title="Players")
    st.plotly_chart(fig, use_container_width=True)

    overall_r2 = r2_score(np.log(df_pred["market_value"]), np.log(df_pred["predicted_value"]))
    overall_mae = mean_absolute_error(df_pred["market_value"], df_pred["predicted_value"])
    st.caption(
        f"On the full {len(df_pred):,} players the trained model achieves "
        f"R² = {overall_r2:.2f} and MAE = {fmt_money(overall_mae)}. The "
        f"held-out test metrics above are the honest measure of out-of-sample "
        f"performance."
    )


# --------------------------- Tab 5: About ---------------------------

def tab_about(df: pd.DataFrame, bundle):
    """Methodology, data sources, limitations."""
    st.subheader("About this project")

    st.markdown(f"""
**Goal.** Predict a soccer player's market value from public performance
stats, and explain what's driving each prediction.

**Data.** {DATA_PERIOD}. Scraped from Transfermarkt squad and market-value
pages. {len(df):,} players across the top 5 European leagues (Premier League,
La Liga, Bundesliga, Serie A, Ligue 1). Market values span roughly $60k to $235M USD.

**Features used by the model.**
""")
    feat_rows = [{"Feature": FEATURE_LABELS[f], "Min": df[f].min(),
                  "Median": df[f].median(), "Max": df[f].max()}
                 for f in bundle["features"]]
    st.dataframe(pd.DataFrame(feat_rows), hide_index=True, use_container_width=True)

    st.markdown(f"""
**Model.** RandomForestRegressor (300 trees, max depth 14) on
log(market value). Held-out test R² = **{bundle['metrics']['r2']:.2f}**,
MAE = **{fmt_money(bundle['metrics']['mae_eur'])}**.

**Explanation.** SHAP values decompose each prediction into per-feature
contributions in USD, relative to the dataset baseline.
""")

    st.subheader("Architecture")
    st.markdown(f"""
- **Streamlit app** (this site) loads `model.pkl` locally for instant
  prediction and SHAP, and can also hit the deployed prediction API for
  parity checks.
- **FastAPI service** deployed to Google Cloud Run for production use:
  [API docs]({API_URL}/docs).
- **Containerization** with Podman + a multi-stage Dockerfile.
""")

    st.subheader("Known limitations")
    st.markdown("""
- **Selection bias.** Trained only on top-5-league squads. Predictions
  for lower-tier leagues, youth players, or unknown profiles are
  extrapolations and may be unreliable.
- **Features.** Market value also depends on contract length,
  reputation, transfer rumors, and injury history — none of those are
  features here. The model captures the part of value that performance
  stats can explain, not the rest.
- **Snapshot.** Market values move with the transfer window. The
  numbers in the app reflect the scrape date and will drift over time.
""")

    st.subheader("Code")
    st.markdown(f"""
- API docs (Swagger): [{API_URL}/docs]({API_URL}/docs)
- Final project repo: see the course PR for the link to the standalone
  GitHub repository.
""")


# --------------------------- main ---------------------------

def main() -> None:
    """Render the scouting app with five tabs."""
    global _FX_RATE
    _FX_RATE = fetch_eur_usd_rate()

    bundle, explainer = load_model_bundle()
    df = load_dataset()
    df_pred = compute_all_predictions(bundle, df)

    st.markdown("# Player Market Value Prediction")
    st.markdown(
        f"A RandomForest model predicts a player's market value from seven per-season "
        f"match stats and attributes each prediction with SHAP. "
        f"Trained on **{len(df):,} players** from the top 5 European leagues. "
        f"Test **R² = {bundle['metrics']['r2']}**, **MAE = {fmt_money(bundle['metrics']['mae_eur'])}**."
    )

    st.info(
        "Search or pick a player in **Scout a Player** to see their predicted value and "
        "the features driving it. Use **Market Movers** to see who the model thinks is "
        "over- or under-valued, or **Leaderboards** to compare leagues and clubs."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Scout a Player", "Market Movers", "Leaderboards", "Model Performance", "About"]
    )

    with tab1:
        tab_scout(df, bundle, explainer)
    with tab2:
        tab_movers(df_pred)
    with tab3:
        tab_leaderboards(df_pred)
    with tab4:
        tab_model_perf(df_pred, bundle)
    with tab5:
        tab_about(df, bundle)

    st.divider()
    cols = st.columns([2, 1])
    with cols[0]:
        st.caption(
            "Model: RandomForestRegressor on log(market_value), 7 features. "
            "Data: Transfermarkt, top 5 European leagues. "
            f"Values in USD at live rate 1 EUR = {_FX_RATE:.4f} USD."
        )
    with cols[1]:
        st.caption(f"[Cloud Run API]({API_URL}/docs)")


if __name__ == "__main__":
    main()
