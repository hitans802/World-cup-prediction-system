"""
Squad attacking-depth features from goalscorers.csv.

WHAT THIS CAPTURES (and its honest limits):
  goalscorers.csv tells us WHO scored in each international match, but NOT who
  played (no lineups/minutes). So we cannot build a true "this squad's strength"
  rating. What we CAN build is a team-level recent ATTACKING profile:

    scorer_depth   -> # distinct scorers over the recent window (more = deeper,
                      less reliant on one player)
    openplay_rate  -> share of goals from open play (not penalties) -> sustainable
    goal_rate      -> goals per match over the window

These are computed leak-free: for each match, only goals BEFORE that match count.
They are intended as extra features for the LightGBM classifier, and MUST be
validated (does log loss improve?) before being trusted -- attacking depth may
already be captured by Elo / goals-for features.

Usage:
    python src/features/squad_features.py
    -> writes data/processed/squad_features.csv keyed by (date, home, away)
"""
from __future__ import annotations
import sys
from collections import defaultdict, deque
import numpy as np
import pandas as pd

WINDOW_GOALS = 40        # rolling window of recent goals per team
RESULTS = "data/raw/results.csv"
GOALSCORERS = "data/raw/goalscorers.csv"


def build_squad_features(results: pd.DataFrame,
                         goals: pd.DataFrame) -> pd.DataFrame:
    """For each match in results, attach each team's recent attacking profile
    using only goals scored strictly before that match date."""
    goals = goals.dropna(subset=["scorer"]).sort_values("date")
    goals = goals[~goals["own_goal"].fillna(False)]

    # per-team rolling deques of recent (scorer, is_penalty)
    recent = defaultdict(lambda: deque(maxlen=WINDOW_GOALS))
    # we also need goals-per-match -> track recent match count per team
    match_count = defaultdict(lambda: deque(maxlen=WINDOW_GOALS))

    # index goals by date for fast "before this match" updates
    goals_by_date = {d: grp for d, grp in goals.groupby("date")}
    all_goal_dates = sorted(goals_by_date.keys())

    def team_profile(team):
        dq = recent[team]
        if not dq:
            return 0.0, 0.5, 0.0
        scorers = [s for s, _ in dq]
        depth = len(set(scorers))
        openplay = np.mean([0.0 if pen else 1.0 for _, pen in dq])
        # goal_rate approximated as goals / distinct recent match-days for team
        mc = len(set(match_count[team])) or 1
        rate = len(dq) / mc
        return float(depth), float(openplay), float(rate)

    results = results.sort_values("date").reset_index(drop=True)
    gi = 0
    rows = []
    for r in results.itertuples(index=False):
        # roll in all goals strictly before this match's date
        while gi < len(all_goal_dates) and all_goal_dates[gi] < r.date:
            d = all_goal_dates[gi]
            for gr in goals_by_date[d].itertuples(index=False):
                recent[gr.team].append((gr.scorer, bool(gr.penalty)))
                match_count[gr.team].append(d)
            gi += 1

        dh, oh, rh = team_profile(r.home_team)
        da, oa, ra = team_profile(r.away_team)
        rows.append({
            "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
            "depth_home": dh, "depth_away": da, "depth_diff": dh - da,
            "openplay_home": oh, "openplay_away": oa, "openplay_diff": oh - oa,
            "scrate_home": rh, "scrate_away": ra, "scrate_diff": rh - ra,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = pd.read_csv(RESULTS, parse_dates=["date"]).dropna(
        subset=["home_score", "away_score"])
    goals = pd.read_csv(GOALSCORERS, parse_dates=["date"])
    feats = build_squad_features(results, goals)

    import os
    os.makedirs("data/processed", exist_ok=True)
    feats.to_csv("data/processed/squad_features.csv", index=False)
    print(f"Built squad features: {len(feats):,} rows")
    print(feats[["depth_diff", "openplay_diff", "scrate_diff"]].describe().round(3))
    print("\nSample (recent):")
    print(feats.tail(3).to_string())
