"""
Feature engineering for match outcome prediction.

Every feature is computed using ONLY information available BEFORE kickoff
(no leakage). Run elo.py first conceptually — we re-run Elo here so the whole
pipeline is self-contained and produces one model-ready table.

Output columns (the model's inputs), all as home-minus-away differences or
pre-match values:
    elo_home, elo_away, elo_diff
    form_home, form_away, form_diff            (avg points last N matches)
    gf_home, gf_away, gf_diff                  (avg goals scored last N)
    ga_home, ga_away, ga_diff                  (avg goals conceded last N)
    rest_home, rest_away, rest_diff            (days since last match)
    h2h_home                                   (home win-rate in last H2H meetings)
    neutral, is_major                          (context flags)

Target:
    result   -> 0 = away win, 1 = draw, 2 = home win

Usage (from project root):
    python src/features/build_features.py data/raw/results.csv data/processed/features.csv
"""
from __future__ import annotations
import sys
from collections import defaultdict, deque
import pandas as pd

# Import the Elo model we already built.
sys.path.insert(0, "src/features")
from elo import EloModel, load_results

FORM_N = 5          # how many recent matches define "form"
H2H_N = 5           # how many recent head-to-head meetings to look back on
MAJOR_TOURNAMENTS = {
    "FIFA World Cup", "FIFA World Cup qualification",
    "UEFA Euro", "Copa América", "African Cup of Nations",
    "AFC Asian Cup", "CONCACAF Gold Cup",
}


def _points(gf: int, ga: int) -> int:
    """Points from a single match: 3 win / 1 draw / 0 loss."""
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # 1. Elo (adds pre-match elo_home / elo_away / elo_diff, leak-free).
    df = EloModel().run(df)

    # Rolling per-team history. deque keeps only the last N entries.
    form: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))      # points
    gf_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))   # goals for
    ga_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=FORM_N))   # goals against
    last_date: dict[str, pd.Timestamp] = {}
    h2h: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=H2H_N))      # home result vs opp

    rows = []
    for r in df.itertuples(index=False):
        h, a = r.home_team, r.away_team

        def avg(dq, default):
            return sum(dq) / len(dq) if dq else default

        # --- features from history BEFORE this match ---
        form_h, form_a = avg(form[h], 1.0), avg(form[a], 1.0)
        gf_h, gf_a = avg(gf_hist[h], 1.0), avg(gf_hist[a], 1.0)
        ga_h, ga_a = avg(ga_hist[h], 1.0), avg(ga_hist[a], 1.0)

        rest_h = (r.date - last_date[h]).days if h in last_date else 30
        rest_a = (r.date - last_date[a]).days if a in last_date else 30
        rest_h, rest_a = min(rest_h, 365), min(rest_a, 365)  # cap long gaps

        key = tuple(sorted((h, a)))
        meetings = h2h[key]
        # store as 1 if home(of this match) won, 0.5 draw, 0 loss
        h2h_home = (sum(meetings) / len(meetings)) if meetings else 0.5

        is_major = int(getattr(r, "tournament", "") in MAJOR_TOURNAMENTS)

        # --- target ---
        if r.home_score > r.away_score:
            result = 2
        elif r.home_score == r.away_score:
            result = 1
        else:
            result = 0

        rows.append({
            "date": r.date,
            "home_team": h, "away_team": a,
            "elo_home": r.elo_home, "elo_away": r.elo_away, "elo_diff": r.elo_diff,
            "form_home": form_h, "form_away": form_a, "form_diff": form_h - form_a,
            "gf_home": gf_h, "gf_away": gf_a, "gf_diff": gf_h - gf_a,
            "ga_home": ga_h, "ga_away": ga_a, "ga_diff": ga_h - ga_a,
            "rest_home": rest_h, "rest_away": rest_a, "rest_diff": rest_h - rest_a,
            "h2h_home": h2h_home,
            "neutral": int(bool(getattr(r, "neutral", False))),
            "is_major": is_major,
            "result": result,
        })

        # --- update history AFTER recording features (prevents leakage) ---
        form[h].append(_points(r.home_score, r.away_score))
        form[a].append(_points(r.away_score, r.home_score))
        gf_hist[h].append(r.home_score); ga_hist[h].append(r.away_score)
        gf_hist[a].append(r.away_score); ga_hist[a].append(r.home_score)
        last_date[h] = r.date; last_date[a] = r.date
        # h2h stored from the perspective of THIS match's home team
        h2h[key].append(1.0 if result == 2 else 0.5 if result == 1 else 0.0)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/features.csv"

    df = load_results(in_path)
    feats = build_features(df)
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    feats.to_csv(out_path, index=False)

    print(f"Built {len(feats):,} rows -> {out_path}")
    print("\nFeature columns:")
    print([c for c in feats.columns if c not in ("date", "home_team", "away_team")])
    print("\nResult balance (0=away,1=draw,2=home):")
    print(feats["result"].value_counts(normalize=True).round(3).sort_index())
    print("\nSample (recent rows):")
    print(feats.tail(3).to_string())
