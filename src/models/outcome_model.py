"""
Outcome model for the app: LightGBM (win/draw/loss) with the validated
engineered + squad features. This is the ACCURATE classifier the web app uses
for outcome probabilities, while Dixon-Coles handles scorelines & simulation.

Because predicting a NEW fixture needs its features computed, we keep the whole
feature-building state and expose predict_fixture(home, away) that builds the
current pre-match feature vector for any matchup from the latest data.

Trains once and caches to models/outcome_model.pkl (keyed to data size), so the
app doesn't retrain on every launch.

Falls back to sklearn's HistGradientBoosting if lightgbm isn't installed, so the
app always works.
"""
from __future__ import annotations
import os
import pickle
from collections import defaultdict, deque
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _HAVE_LGB = True
except Exception:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAVE_LGB = False

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from elo import EloModel

FORM_N = 5
CACHE = os.path.join("models", "outcome_model.pkl")


def _deque_factory():
    return deque(maxlen=FORM_N)


# ----------------------------------------------------------------------
# Feature state: holds latest per-team history so we can build a feature
# vector for any hypothetical upcoming matchup.
# ----------------------------------------------------------------------
class FeatureState:
    def __init__(self):
        self.elo = {}
        self.form = defaultdict(_deque_factory)
        self.gf = defaultdict(_deque_factory)
        self.ga = defaultdict(_deque_factory)
        self.last_date = {}
        self.depth = {}          # latest squad depth per team
        self.openplay = {}
        self.scrate = {}
        self.max_date = None

    def _avg(self, dq, d=1.0):
        return sum(dq) / len(dq) if dq else d

    def vector(self, home, away):
        """Build the model's feature vector for a home/away matchup NOW."""
        eh = self.elo.get(home, 1500.0)
        ea = self.elo.get(away, 1500.0)
        fh, fa = self._avg(self.form[home]), self._avg(self.form[away])
        gfh, gfa = self._avg(self.gf[home]), self._avg(self.gf[away])
        gah, gaa = self._avg(self.ga[home]), self._avg(self.ga[away])
        rh = 30 if home not in self.last_date else min(
            (self.max_date - self.last_date[home]).days, 365)
        ra = 30 if away not in self.last_date else min(
            (self.max_date - self.last_date[away]).days, 365)
        dh, da = self.depth.get(home, 0.0), self.depth.get(away, 0.0)
        oh, oa = self.openplay.get(home, 0.5), self.openplay.get(away, 0.5)
        sh, sa = self.scrate.get(home, 0.0), self.scrate.get(away, 0.0)
        return {
            "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
            "form_home": fh, "form_away": fa, "form_diff": fh - fa,
            "gf_home": gfh, "gf_away": gfa, "gf_diff": gfh - gfa,
            "ga_home": gah, "ga_away": gaa, "ga_diff": gah - gaa,
            "rest_home": rh, "rest_away": ra, "rest_diff": rh - ra,
            "h2h_home": 0.5, "neutral": 1, "is_major": 1,
            "depth_home": dh, "depth_away": da, "depth_diff": dh - da,
            "openplay_diff": oh - oa, "scrate_diff": sh - sa,
        }


FEATURES = ["elo_home", "elo_away", "elo_diff", "form_home", "form_away",
            "form_diff", "gf_home", "gf_away", "gf_diff", "ga_home", "ga_away",
            "ga_diff", "rest_home", "rest_away", "rest_diff", "h2h_home",
            "neutral", "is_major", "depth_home", "depth_away", "depth_diff",
            "openplay_diff", "scrate_diff"]


def _build_training_table(results, goals):
    """Replicate the leak-free feature build and return (X, y, final_state)."""
    from squad_features import build_squad_features
    from build_features import build_features as build_base

    base = build_base(results)                      # elo/form/rest/h2h + result
    squad = build_squad_features(results, goals)     # depth/openplay/scrate
    df = base.merge(
        squad[["date", "home_team", "away_team", "depth_home", "depth_away",
               "depth_diff", "openplay_diff", "scrate_diff"]],
        on=["date", "home_team", "away_team"], how="left").fillna(0.0)

    # rebuild a FeatureState reflecting the LATEST state (for future fixtures)
    st = FeatureState()
    elo = EloModel()
    r2 = elo.run(results.sort_values("date"))
    st.elo = dict(elo.ratings)
    st.max_date = results["date"].max()
    for r in results.sort_values("date").itertuples(index=False):
        st.form[r.home_team].append(_pts(r.home_score, r.away_score))
        st.form[r.away_team].append(_pts(r.away_score, r.home_score))
        st.gf[r.home_team].append(r.home_score); st.ga[r.home_team].append(r.away_score)
        st.gf[r.away_team].append(r.away_score); st.ga[r.away_team].append(r.home_score)
        st.last_date[r.home_team] = r.date; st.last_date[r.away_team] = r.date
    # latest squad profile per team from the squad feature rows
    for r in squad.itertuples(index=False):
        st.depth[r.home_team] = r.depth_home
        st.depth[r.away_team] = r.depth_away
    return df, st


def _pts(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def train_outcome_model(results, goals):
    df, state = _build_training_table(results, goals)
    X, y = df[FEATURES].fillna(0.0), df["result"].astype(int)
    if _HAVE_LGB:
        model = lgb.LGBMClassifier(
            objective="multiclass", num_class=3, n_estimators=400,
            learning_rate=0.02, num_leaves=31, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_samples=50, reg_lambda=1.0,
            random_state=42, n_jobs=-1, verbose=-1)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    return model, state


def load_or_train(results, goals, cache=CACHE):
    sig = str(results.dropna(subset=["home_score", "away_score"]).shape[0])
    if os.path.exists(cache):
        try:
            with open(cache, "rb") as f:
                blob = pickle.load(f)
            if blob["sig"] == sig:
                return blob["model"], blob["state"]
        except Exception:
            pass
    model, state = train_outcome_model(results, goals)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"sig": sig, "model": model, "state": state}, f)
    return model, state


def predict_outcome(model, state, home, away):
    """Return (p_home, p_draw, p_away) for a matchup from the outcome model."""
    vec = state.vector(home, away)
    X = pd.DataFrame([[vec[f] for f in FEATURES]], columns=FEATURES)
    proba = model.predict_proba(X)[0]     # classes: 0=away,1=draw,2=home
    return float(proba[2]), float(proba[1]), float(proba[0])
