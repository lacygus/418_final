# Soccer Player Market Value Scout

Predict soccer player market values from performance stats, and explain
what's driving each prediction. STAT 418 final project.

## Live services

- **Web app (Streamlit):** https://418final-uxdflcbiixufvv9rpxytej.streamlit.app/
- **Prediction API (Cloud Run):** https://market-value-api-348858993647.us-central1.run.app
- **API docs (Swagger UI):** https://market-value-api-348858993647.us-central1.run.app/docs

## What it does

- Pick any of 2,013 players from the top 5 European leagues and see the
  model's predicted market value, an 80% prediction interval, and the
  per-feature contributions that explain it.
- Browse the players the model thinks are most over- or undervalued,
  filtered by league, position, and playing time.
- Compare leagues, clubs, and positions on actual market value.
- Inspect how accurate the model is league-by-league.

## Model

- `RandomForestRegressor` on `log(market_value)`, 7 features
  (age, appearances, goals, assists, minutes, yellow_cards, red_cards).
- Trained on 2,013 players. Held-out test R² = 0.59, MAE = ~$9.6M USD.
- Confidence and the 80% interval come from the spread of the forest's
  individual tree predictions.

## Repo layout

```
.
├── app.py                       Streamlit web app
├── model.pkl                    Trained model bundle (~7 MB)
├── requirements.txt
├── data/
│   ├── processed/model_dataset.csv   2,013 players × 12 columns
│   └── analysis/                7 EDA plots
├── scripts/                     scraping, processing, EDA
└── api/                         FastAPI service deployed to Cloud Run
    ├── main.py / models.py / auth.py / config.py
    ├── train_model.py
    ├── Dockerfile + deployment/
    └── tests/test_api.py
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `model.pkl` for instant local prediction and SHAP. It can
also call the deployed Cloud Run API for parity checks.

## Architecture

```
Transfermarkt + Sofifa
        │
        ▼
   scripts/        (scrape + clean + merge)
        │
        ▼
data/processed/model_dataset.csv  ──►  api/train_model.py  ──►  model.pkl
                                                                    │
                                  ┌─────────────────────────────────┤
                                  ▼                                 ▼
                          api/  (FastAPI)                    app.py  (Streamlit)
                                  │                                 │
                          Cloud Run container                Streamlit Cloud
                                  │                                 │
                                  └────────────► API call ◄─────────┘
```

## AI assistant usage

This project was built with heavy use of Claude (Anthropic's Claude Code).
Specifically:

- **Data scraping:** Claude wrote the initial Transfermarkt and Sofifa
  scrapers; I redirected the approach when range-restricted data made
  the model R² go negative, and Claude rewrote the data collection to
  use full club squad pages.
- **API:** Claude scaffolded the FastAPI app, Pydantic models, auth,
  Dockerfile, and tests. I reviewed and adjusted error codes (missing
  vs wrong key → 403/401) to match the assignment rubric.
- **Streamlit app:** Claude built the 5-tab structure. I directed
  feature additions (search box, position filter, league heatmap, club
  logos) and trade-off decisions (live vs cached FX rate).
- **Deployment:** Podman setup on Windows hit WSL issues; Claude
  diagnosed and switched to `gcloud run deploy --source .` (same
  Dockerfile, Cloud Build does the work) so the API still ships.
- **Where I had to step in:** model design decisions, sample selection
  bias diagnosis, and the explicit limitations panel — those came from
  understanding the data, not from AI generation.
