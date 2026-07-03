# World Cup 2026 Predictor

A football match predictor built around the 2026 World Cup. It predicts match
outcomes, breaks down the likely scorelines, and runs a Monte Carlo simulation
of the knockout bracket to estimate each team's chance of winning the trophy.
There's an interactive web app where you can click through the bracket, see the
model's predictions, and enter your own "what-if" results.

Trained on ~49,000 international matches going back to 1872.

## What it does

- **Predicts individual matches** — win/draw/loss probabilities plus the most
  likely scorelines, shown as a heatmap.
- **Simulates the tournament** — plays the knockout bracket thousands of times
  to estimate how often each team reaches the semis, final, or wins it all.
- **Lets you explore** — enter results yourself and watch the odds shift.

## How it works

Different questions need different models, so it uses a few:

- **Elo ratings** — every team gets a rating (like in chess), updated after each
  match, weighted so bigger tournaments and bigger wins count more. Used as a
  baseline and as a feature.
- **A LightGBM classifier** — predicts win/draw/loss from features like Elo
  difference, recent form, goals scored/conceded, rest days, head-to-head, and a
  squad-strength signal built from goalscorer data. This is the most accurate
  piece for outcomes.
- **A Dixon-Coles Poisson model** — predicts actual scorelines by modelling each
  team's goals based on attacking and defensive strength (with the Dixon-Coles
  correction for low-scoring draws). This powers the heatmaps and lets the
  tournament be simulated.
- **Monte Carlo simulation** — plays the remaining bracket thousands of times to
  estimate title odds. Even a strong favourite only wins occasionally, because
  they have to win six or seven games in a row.

## Things worth pointing out

- **No data leakage** — every feature uses only information available before each
  match, and the model is tested on a time-based split (trained on pre-2018,
  tested on 2018+) rather than a random shuffle.
- **Penalty shootouts handled honestly** — a drawn knockout stays a draw for
  rating purposes, but the shootout winner still advances. Historically the
  stronger team only wins shootouts ~52% of the time, so the simulation treats
  them as near coin-flips.

## The web app

Built with Streamlit. The knockout bracket shows as clickable cards — click any
match for the full prediction and scoreline heatmap. You can enter your own
results to explore what-if scenarios; these are private to your browser session
and reset on reload, so multiple people can use it at once without interfering.
Official results come from the data file in the repo.

## Running it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run src/app/streamlit_app.py
```

The first launch fits the models (20-30 seconds), then caches them.

## Project layout

```
data/raw/          source data (results, shootouts, goalscorers)
src/app/           the Streamlit web app
src/features/      Elo, form/rest/H2H features, squad strength
src/models/        Dixon-Coles, LightGBM, training
src/simulation/    standalone tournament simulator
```

## Data

Built on the "International football results from 1872 to present" dataset by
Mart Jürisoo (Kaggle), with its companion shootouts and goalscorers files.
