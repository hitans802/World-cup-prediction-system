"""
Individual match prediction.

Predicts a single match (or all remaining 2026 World Cup fixtures) using the
Dixon-Coles model: win/draw/loss probabilities + the most likely scorelines.

Three modes:
  1. All remaining (unplayed) 2026 World Cup fixtures:
        python src/models/predict_match.py data/raw/results.csv
  2. One custom matchup:
        python src/models/predict_match.py data/raw/results.csv "Brazil" "France"
  3. (neutral flag is on by default for World Cup; host USA games aren't
     special-cased here.)

Output per match:
    P(team1 win) / P(draw) / P(team2 win)
    expected goals for each side
    top 5 most likely scorelines
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src/models")
from dixon_coles import DixonColes


def fit_model(df: pd.DataFrame):
    hist = df.dropna(subset=["home_score", "away_score"])
    hist = hist[hist.date <= pd.Timestamp("2026-06-23")]
    hist = hist[hist.date >= hist.date.max() - pd.Timedelta(days=365 * 8)]
    # competition weighting ON: competitive matches pull harder than friendlies
    model = DixonColes(use_competition_weights=True).fit(hist)

    # apply current 2026 World Cup form on top of the base ratings
    wc2026 = df[(df.tournament == "FIFA World Cup") &
                (df.date >= pd.Timestamp("2026-01-01"))]
    model.apply_tournament_form(wc2026, strength=0.15)

    # fallback strength for unseen teams
    da = np.mean(list(model.params["attack"].values()))
    dd = np.mean(list(model.params["defence"].values()))
    return model, da, dd


def predict(model, da, dd, t1, t2, neutral=True):
    for t in (t1, t2):
        model.params["attack"].setdefault(t, da)
        model.params["defence"].setdefault(t, dd)

    grid = model.score_matrix(t1, t2, neutral=neutral)
    p1 = np.tril(grid, -1).sum()
    pd_ = np.trace(grid)
    p2 = np.triu(grid, 1).sum()

    m = grid.shape[0]
    idx = np.arange(m)
    exp1 = (grid.sum(axis=1) * idx).sum()
    exp2 = (grid.sum(axis=0) * idx).sum()

    flat = grid.ravel()
    top = np.argsort(flat)[::-1][:5]
    scorelines = [(divmod(k, m), flat[k]) for k in top]

    return {
        "t1": t1, "t2": t2,
        "p1": p1, "pd": pd_, "p2": p2,
        "exp1": exp1, "exp2": exp2,
        "scorelines": scorelines,
    }


def print_prediction(r):
    print(f"\n{r['t1']} vs {r['t2']}")
    print(f"  {r['t1']} win: {r['p1']:.1%}   Draw: {r['pd']:.1%}   "
          f"{r['t2']} win: {r['p2']:.1%}")
    print(f"  Expected goals: {r['t1']} {r['exp1']:.2f} - "
          f"{r['exp2']:.2f} {r['t2']}")
    print("  Most likely scorelines:")
    for (hg, ag), p in r["scorelines"]:
        print(f"    {r['t1']} {hg}-{ag} {r['t2']}   {p:.1%}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    model, da, dd = fit_model(df)

    # custom single match
    if len(sys.argv) >= 4:
        r = predict(model, da, dd, sys.argv[2], sys.argv[3])
        print_prediction(r)
        return

    # all remaining 2026 World Cup fixtures
    wc = df[(df.tournament == "FIFA World Cup") &
            (df.date >= pd.Timestamp("2026-01-01"))]
    remaining = wc[wc.home_score.isna()]
    if remaining.empty:
        print("No unplayed 2026 World Cup fixtures found in the data.")
        return

    print(f"Predicting {len(remaining)} remaining 2026 World Cup fixtures:")
    rows = []
    for _, m in remaining.iterrows():
        r = predict(model, da, dd, m.home_team, m.away_team)
        print_prediction(r)
        best = r["scorelines"][0][0]
        rows.append({
            "date": m.date.date(),
            "home": r["t1"], "away": r["t2"],
            "p_home": round(r["p1"], 3),
            "p_draw": round(r["pd"], 3),
            "p_away": round(r["p2"], 3),
            "likely_score": f"{best[0]}-{best[1]}",
        })

    out = pd.DataFrame(rows)
    import os
    os.makedirs("reports", exist_ok=True)
    out.to_csv("reports/match_predictions.csv", index=False)
    print(f"\nSaved table -> reports/match_predictions.csv")


if __name__ == "__main__":
    main()
