"""
Elo rating system for international football.

Serves two roles:
  1. A baseline model to beat.
  2. A pre-match feature (elo_home, elo_away, elo_diff) for the ML model.

Usage:
    df = load_results("data/raw/results.csv")   # columns below
    elo = EloModel()
    df = elo.run(df)   # adds pre-match elo cols + updates ratings chronologically

Expected input columns (from the Kaggle international-results dataset):
    date, home_team, away_team, home_score, away_score, tournament, neutral
"""
from __future__ import annotations
import pandas as pd


class EloModel:
    def __init__(self, k: float = 30.0, home_adv: float = 65.0,
                 base_rating: float = 1500.0, tournament_weights: dict | None = None):
        # K controls update speed; home_adv is added to home elo before expectation.
        self.k = k
        self.home_adv = home_adv
        self.base_rating = base_rating
        self.ratings: dict[str, float] = {}
        # Bigger matches move ratings more. Tune these.
        self.tournament_weights = tournament_weights or {
            "FIFA World Cup": 2.0,
            "FIFA World Cup qualification": 1.3,
            "UEFA Euro": 1.7,
            "Copa América": 1.7,
            "Friendly": 0.7,
        }

    def _get(self, team: str) -> float:
        return self.ratings.get(team, self.base_rating)

    @staticmethod
    def _expected(r_a: float, r_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))

    def _weight(self, tournament: str, goal_diff: int) -> float:
        w = self.tournament_weights.get(tournament, 1.0)
        # Margin-of-victory multiplier (caps the blowout effect).
        mov = 1.0 + 0.5 * min(goal_diff - 1, 4) / 4 if goal_diff > 1 else 1.0
        return w * mov

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process matches chronologically. Returns df with PRE-match elo columns
        added (no leakage), then updates ratings using the result."""
        df = df.sort_values("date").reset_index(drop=True)
        elo_home, elo_away = [], []

        for row in df.itertuples(index=False):
            ra, rb = self._get(row.home_team), self._get(row.away_team)
            elo_home.append(ra)
            elo_away.append(rb)

            adv = 0.0 if getattr(row, "neutral", False) else self.home_adv
            exp_home = self._expected(ra + adv, rb)

            if row.home_score > row.away_score:
                s_home = 1.0
            elif row.home_score < row.away_score:
                s_home = 0.0
            else:
                s_home = 0.5

            gd = abs(int(row.home_score) - int(row.away_score))
            k_eff = self.k * self._weight(getattr(row, "tournament", ""), gd)
            change = k_eff * (s_home - exp_home)

            self.ratings[row.home_team] = ra + change
            self.ratings[row.away_team] = rb - change

        df["elo_home"] = elo_home
        df["elo_away"] = elo_away
        df["elo_diff"] = df["elo_home"] - df["elo_away"]
        return df

    def top(self, n: int = 20) -> pd.DataFrame:
        return (pd.Series(self.ratings, name="elo")
                .sort_values(ascending=False).head(n).reset_index()
                .rename(columns={"index": "team"}))


def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "neutral" not in df.columns:
        df["neutral"] = False
    if "tournament" not in df.columns:
        df["tournament"] = ""
    return df.dropna(subset=["home_score", "away_score"])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/results.csv"
    df = load_results(path)
    elo = EloModel()
    df = elo.run(df)
    print(elo.top(20).to_string(index=False))
