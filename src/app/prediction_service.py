"""
prediction_service.py — the "brain" behind the web app.

This module sits between the Streamlit UI (streamlit_app.py) and the underlying
models. The app never talks to the models directly; it calls functions here.

What it provides:
  * Loading the data (results, shootouts, goalscorers).
  * Fitting the models once and caching them (Dixon-Coles for scorelines,
    LightGBM for win/draw/loss) so the app stays fast.
  * predict()      — win/draw/loss probabilities, expected goals, most likely
                     scoreline, and the full scoreline grid for the heatmap.
  * build_bracket_tree() — assembles the live knockout bracket (R32 → Final),
                     filling in later rounds as earlier results come in.
  * simulate_knockouts() — Monte Carlo simulation of the remaining bracket,
                     returning each team's title/final/semi probabilities.
  * apply_session_edits() / session_shootouts() — overlay a single user's private
                     what-if results on top of the official bracket, without
                     touching the shared data.
  * A few helpers for flags, shootout handling, and the empirical shootout
    calibration (drawn knockouts are near coin-flips, per historical data).

Keeping all of this here means the UI file stays purely about layout and clicks.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "features"))
from dixon_coles import DixonColes
import outcome_model as om

DATA_PATH = os.path.join("data", "raw", "results.csv")
SHOOTOUTS_PATH = os.path.join("data", "raw", "shootouts.csv")
GOALSCORERS_PATH = os.path.join("data", "raw", "goalscorers.csv")
HISTORY_PATH = os.path.join("data", "processed", "prediction_history.csv")
MAX_GRID_DISPLAY = 6   # show 0..5 goals in the heatmap (rest is negligible)

# Empirical shootout calibration from 678 historical shootouts:
#   stronger team wins only ~52%, home team ~54% -> nearly a coin flip.
# So simulated shootouts use a heavily-damped strength edge, not the
# overconfident logistic the bracket used before.
SHOOTOUT_HOME_EDGE = 0.02     # small home/familiarity bump
SHOOTOUT_STRENGTH_W = 0.03    # tiny strength tilt: even a big gap stays ~coin-flip


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"])


def load_shootouts(path: str = SHOOTOUTS_PATH) -> pd.DataFrame:
    """Real historical shootout winners. Returns empty df if file missing."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "winner"])
    return pd.read_csv(path, parse_dates=["date"])


def lookup_shootout_winner(shootouts: pd.DataFrame, home, away, date=None):
    """Find the real shootout winner for a drawn tie (teams in either order).
    Returns the winning team name, or None if no shootout record exists."""
    if shootouts.empty:
        return None
    m = (((shootouts["home_team"] == home) & (shootouts["away_team"] == away)) |
         ((shootouts["home_team"] == away) & (shootouts["away_team"] == home)))
    if date is not None:
        m = m & (shootouts["date"] == pd.Timestamp(date))
    hits = shootouts[m]
    if hits.empty:
        return None
    return hits.iloc[-1]["winner"]   # most recent match if several


def fit_model(df: pd.DataFrame):
    """Fit Dixon-Coles with competition weighting + 2026 tournament form."""
    hist = df.dropna(subset=["home_score", "away_score"])
    cutoff = hist["date"].max()
    hist = hist[hist["date"] >= cutoff - pd.Timedelta(days=365 * 8)]
    model = DixonColes(use_competition_weights=True).fit(hist)

    wc = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= pd.Timestamp("2026-01-01"))]
    played = wc[wc["home_score"].notna()]
    if len(played):
        model.apply_tournament_form(played, strength=0.15)

    # fallback strength for unseen teams
    da = float(np.mean(list(model.params["attack"].values())))
    dd = float(np.mean(list(model.params["defence"].values())))
    return model, da, dd


def fit_or_load(df: pd.DataFrame, cache_path: str = "models/dc_params.json"):
    """Load a saved fit if it matches current data, else refit and save.
    Avoids the ~20-30s optimization on app restart when data is unchanged.
    The signature is the count of played matches, so adding a result busts it."""
    import json
    sig = str(df.dropna(subset=["home_score", "away_score"]).shape[0])
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                blob = json.load(f)
            if blob.get("sig") == sig:
                model = DixonColes(use_competition_weights=True)
                model.params = blob["params"]
                model.teams = list(blob["params"]["attack"].keys())
                return model, blob["da"], blob["dd"]
        except Exception:
            pass
    model, da, dd = fit_model(df)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"sig": sig, "params": model.params, "da": da, "dd": dd}, f)
    return model, da, dd


def load_outcome_model(df: pd.DataFrame):
    """Load/train the LightGBM outcome model (accurate W/D/L classifier with
    squad features). Returns (model, state) or None if goalscorers missing."""
    if not os.path.exists(GOALSCORERS_PATH):
        return None
    results = df.dropna(subset=["home_score", "away_score"])
    goals = pd.read_csv(GOALSCORERS_PATH, parse_dates=["date"])
    return om.load_or_train(results, goals)


def _ensure(model, da, dd, *teams):
    for t in teams:
        model.params["attack"].setdefault(t, da)
        model.params["defence"].setdefault(t, dd)


def predict(model, da, dd, home, away, neutral=True, outcome=None):
    """Return a dict with probs, expected goals, likely score, and full grid.

    If `outcome` (the LightGBM model, state) is given, W/D/L probabilities come
    from that more-accurate classifier; Dixon-Coles still provides the scoreline
    grid, expected goals, and most-likely score (which the classifier can't)."""
    _ensure(model, da, dd, home, away)
    grid = model.score_matrix(home, away, neutral=neutral)

    # scoreline probs from Dixon-Coles (always needed for the grid/xG)
    dc_home = float(np.tril(grid, -1).sum())
    dc_draw = float(np.trace(grid))
    dc_away = float(np.triu(grid, 1).sum())

    # outcome probabilities: prefer the accurate LightGBM classifier
    if outcome is not None:
        omodel, ostate = outcome
        try:
            p_home, p_draw, p_away = om.predict_outcome(omodel, ostate, home, away)
            src = "LightGBM (with squad features)"
        except Exception:
            p_home, p_draw, p_away = dc_home, dc_draw, dc_away
            src = "Dixon-Coles"
    else:
        p_home, p_draw, p_away = dc_home, dc_draw, dc_away
        src = "Dixon-Coles"

    m = grid.shape[0]
    idx = np.arange(m)
    exp_home = float((grid.sum(axis=1) * idx).sum())
    exp_away = float((grid.sum(axis=0) * idx).sum())

    # Likely score = the single most probable scoreline (the peak of the grid),
    # so it always matches the heatmap the user sees. Note this can differ from
    # the predicted WINNER (e.g. favourite to win, but 1-1 is the modal score) --
    # that's genuine, not a bug: the win prob is spread across many scorelines.
    hg, ag = np.unravel_index(np.argmax(grid), grid.shape)

    outcome_probs = {"home": p_home, "draw": p_draw, "away": p_away}
    pick = max(outcome_probs, key=outcome_probs.get)

    return {
        "home": home, "away": away,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "prob_source": src, "pick": pick,
        "exp_home": round(exp_home, 2), "exp_away": round(exp_away, 2),
        "likely_score": (int(hg), int(ag)),
        "grid": grid[:MAX_GRID_DISPLAY, :MAX_GRID_DISPLAY],
    }


def unplayed_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Remaining (NA-score) 2026 World Cup matches, sorted by date."""
    wc = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= pd.Timestamp("2026-06-24"))]
    return (wc[wc["home_score"].isna()]
            .sort_values("date")
            .reset_index(drop=True))


def apply_session_edits(official_df: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Layer a user's PRIVATE session edits over the official results, without
    touching the official store. `edits` maps (home, away) -> dict with keys
    home_score, away_score, shootout_winner. Returns a new results DataFrame
    for use in the bracket/sim for THIS session only.

    This is how non-admin users tinker: their scores live only in session_state
    and are overlaid here at read time; nothing is persisted."""
    if not edits:
        return official_df
    df = official_df.copy()
    ko_mask = ((df["tournament"] == "FIFA World Cup") &
               (df["date"] >= pd.Timestamp("2026-06-24")))
    new_rows = []
    for (home, away), e in edits.items():
        pair = (((df["home_team"] == home) & (df["away_team"] == away)) |
                ((df["home_team"] == away) & (df["away_team"] == home)))
        target = df[ko_mask & pair]
        if not target.empty:
            i = target.index[0]
            if df.loc[i, "home_team"] == home:
                df.loc[i, "home_score"] = e["home_score"]
                df.loc[i, "away_score"] = e["away_score"]
            else:
                df.loc[i, "home_score"] = e["away_score"]
                df.loc[i, "away_score"] = e["home_score"]
        else:
            new_rows.append({
                "date": pd.Timestamp("2026-07-03"), "home_team": home,
                "away_team": away, "home_score": e["home_score"],
                "away_score": e["away_score"], "tournament": "FIFA World Cup",
                "neutral": True})
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return df


def session_shootouts(official_sh: pd.DataFrame, edits: dict) -> pd.DataFrame:
    """Overlay session shootout winners (for drawn what-if ties) over official."""
    rows = []
    for (home, away), e in (edits or {}).items():
        if e.get("shootout_winner"):
            rows.append({"date": pd.Timestamp("2026-07-03"), "home_team": home,
                         "away_team": away, "winner": e["shootout_winner"],
                         "first_shooter": ""})
    if not rows:
        return official_sh
    return pd.concat([official_sh, pd.DataFrame(rows)], ignore_index=True)


def knockout_bracket(df: pd.DataFrame, shootouts: pd.DataFrame | None = None):
    """Return the current knockout ties in bracket order.
    Played ties keep their result; unplayed ones are to be simulated.
    Consecutive pairs (0,1),(2,3)... meet in the next round.

    For a played tie that ENDED LEVEL, we look up the real shootout winner
    (from shootouts.csv) so the bracket advances the team that actually went
    through -- without altering the 1-1 scoreline the rating model sees."""
    if shootouts is None:
        shootouts = load_shootouts()
    # Only the Round of 32 ties (28 Jun - 3 Jul). Later rounds are derived by
    # build_bracket_tree from these winners, so we must NOT include R16+ fixtures
    # here even if they exist in the data (that would give >16 ties and break
    # the bracket pairing).
    ko = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= pd.Timestamp("2026-06-28")) &
            (df["date"] <= pd.Timestamp("2026-07-03"))].sort_values("date")
    ties = []
    for r in ko.itertuples(index=False):
        played = pd.notna(r.home_score)
        so_winner = None
        if played and int(r.home_score) == int(r.away_score):
            so_winner = lookup_shootout_winner(
                shootouts, r.home_team, r.away_team, date=r.date)
        ties.append({
            "home": r.home_team, "away": r.away_team,
            "played": played,
            "hs": int(r.home_score) if played else None,
            "as": int(r.away_score) if played else None,
            "shootout_winner": so_winner,
        })
    return ties


def simulate_knockouts(model, da, dd, ties, n_sims: int = 5000, seed: int = 42):
    """Monte Carlo from the REAL current bracket forward.
    Returns dict team -> {champion, final, semi} probabilities.
    Much faster than the full-tournament sim: only knockout ties remain.

    Speed: we cache each matchup's win-probability (incl. shootout on draws)
    the first time it's needed, so repeated sims are cheap lookups not grid
    rebuilds."""
    rng = np.random.default_rng(seed)

    # apply the real 2026 bracket seeding order so simulated R16 pairings match
    # the official bracket (see build_bracket_tree). Only when we have the full
    # 16-tie R32; otherwise use as-is.
    BRACKET_ORDER = [3, 0, 2, 5, 1, 4, 6, 7, 9, 8, 11, 10, 14, 13, 12, 15]
    if len(ties) == 16:
        ties = [ties[i] for i in BRACKET_ORDER]

    teams = set()
    for t in ties:
        teams.add(t["home"]); teams.add(t["away"])
        _ensure(model, da, dd, t["home"], t["away"])

    win_cache: dict[tuple, float] = {}

    def p_home_advances(h, a):
        """P(h beats a) in a knockout tie (regulation result, draws -> shootout).
        Shootout is calibrated to the data: nearly a coin flip with only a small
        strength/home tilt, NOT the overconfident logistic used before."""
        key = (h, a)
        if key in win_cache:
            return win_cache[key]
        grid = model.score_matrix(h, a, neutral=True)
        p_h = float(np.tril(grid, -1).sum())
        p_d = float(np.trace(grid))
        sh = model.params["attack"][h] + model.params["defence"][h]
        sa = model.params["attack"][a] + model.params["defence"][a]
        # empirically-grounded shootout: base 0.5, tiny strength tilt + home edge
        p_so = 0.5 + SHOOTOUT_STRENGTH_W * (sh - sa) + SHOOTOUT_HOME_EDGE
        p_so = float(np.clip(p_so, 0.30, 0.70))
        p = p_h + p_d * p_so      # win in regulation OR draw then win shootout
        win_cache[key] = p
        return p

    def play(h, a):
        return h if rng.random() < p_home_advances(h, a) else a

    def resolve_played(t):
        """Winner of an already-played tie. If it was a draw, prefer the REAL
        shootout winner; only fall back to a simulated shootout if unknown."""
        if t["hs"] > t["as"]:
            return t["home"]
        if t["as"] > t["hs"]:
            return t["away"]
        if t.get("shootout_winner"):
            return t["shootout_winner"]
        return play(t["home"], t["away"])

    from collections import Counter
    champ, fin, semi = Counter(), Counter(), Counter()

    for _ in range(n_sims):
        round_teams = []
        for t in ties:
            w = resolve_played(t) if t["played"] else play(t["home"], t["away"])
            round_teams.append(w)

        last4 = last2 = None
        while len(round_teams) > 1:
            nxt = [play(round_teams[i], round_teams[i + 1])
                   for i in range(0, len(round_teams), 2)]
            if len(round_teams) == 4:
                last4 = list(round_teams)
            if len(round_teams) == 2:
                last2 = list(round_teams)
            round_teams = nxt

        champ[round_teams[0]] += 1
        for t in (last2 or []):
            fin[t] += 1
        for t in (last4 or []):
            semi[t] += 1

    return {t: {"champion": champ[t] / n_sims,
                "final": fin[t] / n_sims,
                "semi": semi[t] / n_sims} for t in teams}


def _winner_of(tie, model=None, da=None, dd=None, outcome=None):
    """Winner of a resolved tie, or None if not yet decided.
    Uses the real score; a drawn knockout uses the shootout winner if known."""
    if not tie.get("played"):
        return None
    if tie["hs"] > tie["as"]:
        return tie["home"]
    if tie["as"] > tie["hs"]:
        return tie["away"]
    return tie.get("shootout_winner")   # draw -> needs shootout winner


ROUND_NAMES = {32: "Round of 32", 16: "Round of 16", 8: "Quarter-finals",
               4: "Semi-finals", 2: "Final"}

# ISO2 codes for flag emojis (covers the 2026 knockout teams; extend as needed)
_ISO2 = {
    "South Africa": "ZA", "Canada": "CA", "Brazil": "BR", "Japan": "JP",
    "Germany": "DE", "Paraguay": "PY", "Netherlands": "NL", "Morocco": "MA",
    "Ivory Coast": "CI", "Norway": "NO", "France": "FR", "Sweden": "SE",
    "Mexico": "MX", "Ecuador": "EC", "England": "GB-ENG", "DR Congo": "CD",
    "Belgium": "BE", "Senegal": "SN", "United States": "US",
    "Bosnia and Herzegovina": "BA", "Spain": "ES", "Austria": "AT",
    "Portugal": "PT", "Croatia": "HR", "Switzerland": "CH", "Algeria": "DZ",
    "Australia": "AU", "Egypt": "EG", "Argentina": "AR", "Cape Verde": "CV",
    "Colombia": "CO", "Ghana": "GH", "Uruguay": "UY", "Italy": "IT",
    "Denmark": "DK", "Czech Republic": "CZ", "Poland": "PL", "Nigeria": "NG",
    "Cameroon": "CM", "South Korea": "KR", "Iran": "IR", "Saudi Arabia": "SA",
    "Qatar": "QA", "Costa Rica": "CR", "Peru": "PE", "Chile": "CL",
    "Serbia": "RS", "Wales": "GB-WLS", "Scotland": "GB-SCT", "Tunisia": "TN",
}


def flag(team) -> str:
    """Return a flag emoji for a national team, or a neutral marker."""
    if not team:
        return "⚪"
    code = _ISO2.get(team)
    if not code:
        return "🏳️"
    if code == "GB-ENG":
        return "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    if code == "GB-WLS":
        return "🏴\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f"
    if code == "GB-SCT":
        return "🏴\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f"
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code)


def build_bracket_tree(df, shootouts=None):
    """Build the full knockout tree (R32 -> Final).

    Round 1 (R32) comes straight from the fixtures. Each later round's matches
    are formed from the WINNERS of consecutive ties in the previous round;
    slots with an undecided feeder show team=None ('Winner of ...').

    Returns a list of rounds; each round is a list of match dicts:
        {round_size, idx, home, away, played, hs, as, shootout_winner,
         home_src, away_src}   (src = label when team not yet known)
    """
    if shootouts is None:
        shootouts = load_shootouts()
    r32 = knockout_bracket(df, shootouts=shootouts)   # existing R32 list

    # Real 2026 World Cup bracket order. The official bracket does NOT pair R32
    # winners consecutively (0v1, 2v3...); it uses a fixed seeding path. These
    # index pairs are the actual R16 matchups (verified against FIFA results):
    #   Morocco/Canada, Paraguay/France, Brazil/Norway, Mexico/England,
    #   USA/Belgium, Portugal/Spain, Argentina/Egypt, Switzerland/Colombia.
    BRACKET_ORDER = [3, 0, 2, 5, 1, 4, 6, 7, 9, 8, 11, 10, 14, 13, 12, 15]
    if len(r32) == 16:
        r32 = [r32[i] for i in BRACKET_ORDER]

    rounds = [[]]
    for i, t in enumerate(r32):
        rounds[0].append({
            "round_size": 32, "idx": i,
            "home": t["home"], "away": t["away"],
            "played": t["played"], "hs": t["hs"], "as": t["as"],
            "shootout_winner": t.get("shootout_winner"),
            "home_src": None, "away_src": None,
        })

    # build subsequent rounds from winners of the previous
    prev = rounds[0]
    size = 16
    while len(prev) > 1:
        cur = []
        for j in range(0, len(prev) - 1, 2):     # -1 guards against odd counts
            m1, m2 = prev[j], prev[j + 1]
            w1 = _winner_of(m1)
            w2 = _winner_of(m2)
            # look up if this later-round match has itself been played
            played, hs, as_, sw = _lookup_played_match(df, shootouts, w1, w2)
            cur.append({
                "round_size": size, "idx": j // 2,
                "home": w1, "away": w2,
                "played": played, "hs": hs, "as": as_, "shootout_winner": sw,
                "home_src": f"Winner of {_tie_label(m1)}" if w1 is None else None,
                "away_src": f"Winner of {_tie_label(m2)}" if w2 is None else None,
            })
        rounds.append(cur)
        prev = cur
        size //= 2
    return rounds


def _tie_label(m):
    if m["home"] and m["away"]:
        return f"{m['home']} v {m['away']}"
    return f"{ROUND_NAMES.get(m['round_size'],'')} #{m['idx']+1}"


def _lookup_played_match(df, shootouts, home, away):
    """Has a later-round match between these two teams been entered already?"""
    if home is None or away is None:
        return False, None, None, None
    ko = df[(df["tournament"] == "FIFA World Cup") &
            (df["date"] >= pd.Timestamp("2026-06-28"))]
    m = (((ko["home_team"] == home) & (ko["away_team"] == away)) |
         ((ko["home_team"] == away) & (ko["away_team"] == home)))
    hit = ko[m & ko["home_score"].notna()]
    if hit.empty:
        return False, None, None, None
    r = hit.iloc[-1]
    sw = None
    if int(r["home_score"]) == int(r["away_score"]):
        sw = lookup_shootout_winner(shootouts, r["home_team"], r["away_team"],
                                    date=r["date"])
    # normalize orientation to (home, away) as requested
    if r["home_team"] == home:
        return True, int(r["home_score"]), int(r["away_score"]), sw
    return True, int(r["away_score"]), int(r["home_score"]), sw


def save_result(home, away, home_score, away_score, path: str = DATA_PATH,
                allow_edit: bool = True):
    """Fill in (or edit) the score for a knockout match. Matches an existing
    fixture between the two teams (either orientation); if none exists, appends
    a new World Cup row. Returns the match date so a shootout can be keyed to it.

    allow_edit=True lets us overwrite an already-entered score (for corrections).
    """
    df = pd.read_csv(path, parse_dates=["date"])
    ko_mask = ((df["tournament"] == "FIFA World Cup") &
               (df["date"] >= pd.Timestamp("2026-06-24")))
    pair = (((df["home_team"] == home) & (df["away_team"] == away)) |
            ((df["home_team"] == away) & (df["away_team"] == home)))
    # prefer an unplayed matching fixture; else (if editing) any matching one
    unplayed = df[ko_mask & pair & df["home_score"].isna()]
    target = unplayed
    if target.empty and allow_edit:
        target = df[ko_mask & pair]
    if not target.empty:
        i = target.index[0]
        # write in the stored orientation
        if df.loc[i, "home_team"] == home:
            df.loc[i, "home_score"] = home_score
            df.loc[i, "away_score"] = away_score
        else:
            df.loc[i, "home_score"] = away_score
            df.loc[i, "away_score"] = home_score
        match_date = df.loc[i, "date"]
    else:
        match_date = pd.Timestamp("2026-07-03")
        new = {c: np.nan for c in df.columns}
        new.update({
            "date": match_date, "home_team": home, "away_team": away,
            "home_score": home_score, "away_score": away_score,
            "tournament": "FIFA World Cup", "neutral": True,
        })
        df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(path, index=False)
    return match_date


def clear_result(home, away, path: str = DATA_PATH,
                 shootouts_path: str = SHOOTOUTS_PATH):
    """Reset a knockout match back to unplayed (removes its score), and remove
    any linked shootout record. Later bracket rounds that depended on it will
    revert to 'undecided' automatically. Returns True if something was cleared."""
    df = pd.read_csv(path, parse_dates=["date"])
    ko_mask = ((df["tournament"] == "FIFA World Cup") &
               (df["date"] >= pd.Timestamp("2026-06-24")))
    pair = (((df["home_team"] == home) & (df["away_team"] == away)) |
            ((df["home_team"] == away) & (df["away_team"] == home)))
    hit = df[ko_mask & pair & df["home_score"].notna()]
    if hit.empty:
        return False
    i = hit.index[-1]
    match_date = df.loc[i, "date"]
    df.loc[i, "home_score"] = np.nan
    df.loc[i, "away_score"] = np.nan
    df.to_csv(path, index=False)
    # drop any shootout record for this tie
    if os.path.exists(shootouts_path):
        sh = pd.read_csv(shootouts_path, parse_dates=["date"])
        keep = ~((sh["date"] == match_date) &
                 (((sh["home_team"] == home) & (sh["away_team"] == away)) |
                  ((sh["home_team"] == away) & (sh["away_team"] == home))))
        sh[keep].to_csv(shootouts_path, index=False)
    return True


def save_shootout(home, away, winner, match_date, path: str = SHOOTOUTS_PATH) -> bool:
    """Record a penalty-shootout winner in shootouts.csv. This is what the
    bracket reads to advance the right team on a drawn knockout tie -- WITHOUT
    altering the drawn scoreline in results.csv (so ratings stay goal-based).
    Won't duplicate an existing record for the same date+teams."""
    if os.path.exists(path):
        sh = pd.read_csv(path, parse_dates=["date"])
    else:
        sh = pd.DataFrame(columns=["date", "home_team", "away_team",
                                   "winner", "first_shooter"])
    dt = pd.Timestamp(match_date)
    dup = ((sh["date"] == dt) &
           (((sh["home_team"] == home) & (sh["away_team"] == away)) |
            ((sh["home_team"] == away) & (sh["away_team"] == home))))
    if dup.any():
        sh.loc[dup, "winner"] = winner          # update existing
    else:
        row = {"date": dt, "home_team": home, "away_team": away,
               "winner": winner, "first_shooter": np.nan}
        sh = pd.concat([sh, pd.DataFrame([row])], ignore_index=True)
    sh.sort_values("date").to_csv(path, index=False)
    return True
