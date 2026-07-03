"""
Dixon-Coles Poisson model for scoreline prediction.

WHY THIS MODEL (vs the LightGBM classifier):
  LightGBM predicts P(home win / draw / away win). That's enough to *rate* a
  match, but to SIMULATE a tournament we need actual scorelines -- to advance
  teams, compute goal difference for group tables, and break ties. This model
  is GENERATIVE: it produces a full probability distribution over every
  scoreline (0-0, 2-1, 3-0, ...), which we can sample from.

THE MODEL:
  Each team has two latent parameters:
      attack[team]   -> how many goals it tends to score
      defence[team]  -> how many goals it tends to concede
  Plus a global home_advantage term.

  Expected goals in a match:
      home_goals ~ Poisson(lambda_home)
      away_goals ~ Poisson(lambda_away)
  where
      log(lambda_home) = attack[home] - defence[away] + home_adv
      log(lambda_away) = attack[away] - defence[home]

  DIXON-COLES CORRECTION (the key refinement over plain Poisson):
  Independent Poissons underestimate low-score draws (0-0, 1-1) and over-
  estimate 1-0 / 0-1. Dixon & Coles (1997) add a correction tau(rho) to the
  joint probability of those four scorelines to fix this.

  TIME WEIGHTING: recent matches matter more, so each match is weighted by
  exp(-xi * age_in_days). Older results count for less.

We fit the parameters by maximum likelihood (scipy.optimize.minimize).

Usage (from project root):
    python src/models/dixon_coles.py data/raw/results.csv
"""
from __future__ import annotations
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10          # truncate scoreline grid at 10 goals per side
XI = 0.0018             # time-decay rate (~ half-life of roughly 1 year)


class DixonColes:
    # Competition importance: how much each match TYPE pulls on team ratings.
    # Mirrors the idea in the Elo model — a World Cup result is more telling
    # of true strength than a friendly. Multiplied into the time-decay weight.
    COMPETITION_WEIGHTS = {
        "FIFA World Cup": 2.0,
        "FIFA World Cup qualification": 1.4,
        "UEFA Euro": 1.8,
        "Copa América": 1.8,
        "UEFA Nations League": 1.4,
        "African Cup of Nations": 1.5,
        "AFC Asian Cup": 1.5,
        "CONCACAF Gold Cup": 1.4,
        "Friendly": 0.7,
    }

    def __init__(self, max_goals: int = MAX_GOALS, xi: float = XI,
                 use_competition_weights: bool = True):
        self.max_goals = max_goals
        self.xi = xi
        self.use_competition_weights = use_competition_weights
        self.teams: list[str] = []
        self.params: dict | None = None

    # ---- the Dixon-Coles low-score correction ----
    @staticmethod
    def _tau(hg, ag, lam, mu, rho):
        if hg == 0 and ag == 0:
            return 1 - lam * mu * rho
        if hg == 0 and ag == 1:
            return 1 + lam * rho
        if hg == 1 and ag == 0:
            return 1 + mu * rho
        if hg == 1 and ag == 1:
            return 1 - rho
        return 1.0

    @staticmethod
    def _tau_vec(hg, ag, lam, mu, rho):
        """Vectorized Dixon-Coles correction over all matches at once."""
        tau = np.ones_like(lam)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        return tau

    def _neg_log_likelihood(self, params, hg, ag, hi, ai, w):
        n = len(self.teams)
        attack = params[:n]
        defence = params[n:2 * n]
        home_adv, rho = params[2 * n], params[2 * n + 1]

        lam = np.exp(attack[hi] - defence[ai] + home_adv)   # home expected goals
        mu = np.exp(attack[ai] - defence[hi])               # away expected goals

        # log P(home=hg) + log P(away=ag) under independent Poisson
        ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)

        # Dixon-Coles correction for the four low-score cells (vectorized)
        tau = np.clip(self._tau_vec(hg, ag, lam, mu, rho), 1e-10, None)
        ll = ll + np.log(tau)

        return -np.sum(w * ll)   # weighted negative log-likelihood

    def fit(self, df: pd.DataFrame, ref_date=None):
        df = df.dropna(subset=["home_score", "away_score"]).copy()
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)

        self.teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = df["home_team"].map(idx).values
        ai = df["away_team"].map(idx).values
        hg = df["home_score"].values
        ag = df["away_score"].values

        # time-decay weights (recent games weighted more)
        ref_date = ref_date or df["date"].max()
        age = (pd.to_datetime(ref_date) - df["date"]).dt.days.values
        w = np.exp(-self.xi * age)

        # competition importance: scale weight by match type so competitive
        # results (World Cup, Euro...) pull harder than friendlies.
        if self.use_competition_weights and "tournament" in df.columns:
            comp = df["tournament"].map(
                lambda t: self.COMPETITION_WEIGHTS.get(t, 1.0)).values
            w = w * comp

        # initial guess: attack~0, defence~0, home_adv=0.25, rho=-0.1
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.1]])

        print(f"Fitting Dixon-Coles on {len(df):,} matches, {n} teams...")
        # L-BFGS-B is fast for this many params. Identifiability is handled by
        # centering the attack ratings after fitting (subtract their mean), which
        # is absorbed into the home_adv / defence terms without changing fit.
        res = minimize(
            self._neg_log_likelihood, x0,
            args=(hg, ag, hi, ai, w),
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-7},
        )

        attack = res.x[:n] - res.x[:n].mean()   # center for identifiability
        defence = res.x[n:2 * n]
        self.params = {
            "attack": dict(zip(self.teams, attack)),
            "defence": dict(zip(self.teams, defence)),
            "home_adv": res.x[2 * n],
            "rho": res.x[2 * n + 1],
        }
        print(f"Done. home_adv={self.params['home_adv']:.3f}  rho={self.params['rho']:.3f}")
        return self

    def score_matrix(self, home: str, away: str, neutral: bool = False) -> np.ndarray:
        """Full probability grid P(home_goals=i, away_goals=j)."""
        p = self.params
        adv = 0.0 if neutral else p["home_adv"]
        lam = np.exp(p["attack"][home] - p["defence"][away] + adv)
        mu = np.exp(p["attack"][away] - p["defence"][home])

        m = self.max_goals + 1
        grid = np.outer(poisson.pmf(np.arange(m), lam),
                        poisson.pmf(np.arange(m), mu))
        # apply low-score correction
        for hg, ag in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            grid[hg, ag] *= self._tau(hg, ag, lam, mu, p["rho"])
        return grid / grid.sum()

    def predict_proba(self, home, away, neutral=False):
        """Return (P_home_win, P_draw, P_away_win) from the score grid."""
        g = self.score_matrix(home, away, neutral)
        p_home = np.tril(g, -1).sum()   # home goals > away goals
        p_draw = np.trace(g)            # diagonal
        p_away = np.triu(g, 1).sum()    # away goals > home goals
        return p_home, p_draw, p_away

    def sample_score(self, home, away, neutral=False, rng=None):
        """Draw one random scoreline -- this is what the simulator calls."""
        rng = rng or np.random.default_rng()
        g = self.score_matrix(home, away, neutral).ravel()
        k = rng.choice(len(g), p=g)
        return divmod(k, self.max_goals + 1)   # (home_goals, away_goals)

    def apply_tournament_form(self, tournament_matches: pd.DataFrame,
                              strength: float = 0.15):
        """Nudge each team's attack/defence by how they're ACTUALLY performing
        in the current tournament vs what the model expected.

        For each played match we compare actual goals to the model's expected
        goals (lambda/mu). A team consistently outscoring expectation gets a
        small attack boost; conceding less than expected gets a defence boost.

        `strength` (0-1) controls how big the nudge is. 0.15 = gentle.
        This is applied AFTER fit(), in-place on self.params, so predictions
        reflect current-tournament form on top of the competition-weighted
        ratings. Call once before predicting/simulating.
        """
        if self.params is None:
            raise RuntimeError("Call fit() before apply_tournament_form().")

        tm = tournament_matches.dropna(subset=["home_score", "away_score"]).copy()
        if tm.empty:
            return self

        attack_delta = defaultdict(list)
        defence_delta = defaultdict(list)

        for r in tm.itertuples(index=False):
            h, a = r.home_team, r.away_team
            if h not in self.params["attack"] or a not in self.params["attack"]:
                continue
            # model's expected goals for this match (neutral venue at WC)
            lam = np.exp(self.params["attack"][h] - self.params["defence"][a])
            mu = np.exp(self.params["attack"][a] - self.params["defence"][h])
            hs, as_ = int(r.home_score), int(r.away_score)
            # log-ratio of actual vs expected (clipped); >0 = overperformed
            attack_delta[h].append(np.log((hs + 0.5) / (lam + 0.5)))
            attack_delta[a].append(np.log((as_ + 0.5) / (mu + 0.5)))
            # defence: conceding FEWER than expected is good -> sign flipped
            defence_delta[h].append(-np.log((as_ + 0.5) / (mu + 0.5)))
            defence_delta[a].append(-np.log((hs + 0.5) / (lam + 0.5)))

        for t, vals in attack_delta.items():
            self.params["attack"][t] += strength * float(np.mean(vals))
        for t, vals in defence_delta.items():
            self.params["defence"][t] += strength * float(np.mean(vals))
        return self

    def team_strengths(self) -> pd.DataFrame:
        p = self.params
        return (pd.DataFrame({"attack": p["attack"], "defence": p["defence"]})
                .assign(net=lambda d: d.attack + d.defence)
                .sort_values("net", ascending=False))


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])

    # fit on the most recent ~8 years for relevance/speed
    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=365 * 8)]

    model = DixonColes().fit(df)

    print("\nTop 15 teams by attack+defence strength:")
    print(model.team_strengths().head(15).round(3).to_string())

    print("\nExample: Brazil (home, neutral) vs Argentina")
    ph, pd_, pa = model.predict_proba("Brazil", "Argentina", neutral=True)
    print(f"  P(Brazil win)={ph:.3f}  P(draw)={pd_:.3f}  P(Argentina win)={pa:.3f}")
    g = model.score_matrix("Brazil", "Argentina", neutral=True)
    top = np.dstack(np.unravel_index(np.argsort(g.ravel())[::-1][:5], g.shape))[0]
    print("  Most likely scorelines:")
    for hg, ag in top:
        print(f"    {hg}-{ag}: {g[hg, ag]:.3f}")
