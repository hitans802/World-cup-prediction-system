"""
streamlit_app.py — the web app (the thing users actually see and click).
 
This is the front end: it draws the World Cup knockout bracket, opens a detail
popup when you click a match (predictions + scoreline heatmap), and runs the
tournament simulation when you hit the button. All the modelling logic lives in
prediction_service.py — this file is just layout, styling, and interaction.
 
How results work:
  * The OFFICIAL bracket comes from data/raw/results.csv in the repo. To update
    it (as matches finish), edit that file and push to GitHub; the hosted app
    redeploys with the new results. Visitors cannot change official results.
  * Any visitor can enter their own WHAT-IF results to explore alternative
    brackets. Those edits live only in that person's browser session, are
    private to them, and reset on reload — so lots of people can use the app at
    once without affecting each other or the official bracket.
 
Run locally:
    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "app"))
os.chdir(ROOT)

import prediction_service as ps

st.set_page_config(page_title="World Cup 2026 Bracket", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

# ---------------- theme ----------------
st.markdown("""
<style>
  :root{
    --pitch:#0b2a1a; --pitch2:#0f3625; --line:#1d5a3c;
    --cream:#f2efe4; --muted:#8fa89a; --gold:#f6c945; --gold2:#d99f28;
    --display:'Trebuchet MS','Segoe UI',system-ui,sans-serif;
    --mono:'SF Mono','Consolas','Menlo',monospace;
  }
  .stApp{
    background:
      radial-gradient(1000px 460px at 78% -8%, rgba(46,139,87,.28) 0%, transparent 60%),
      radial-gradient(900px 420px at 12% 4%, rgba(246,201,69,.08) 0%, transparent 55%),
      linear-gradient(180deg,#0b2a1a 0%, #061a10 100%);
    color:var(--cream);
  }
  .block-container{ padding-top:2rem; max-width:1500px; }
  .hero{
    position:relative; padding:26px 30px; border-radius:18px; overflow:hidden;
    background:linear-gradient(120deg,#103826 0%, #0a2417 60%, #0a2417 100%);
    border:1px solid var(--line); box-shadow:0 10px 40px rgba(0,0,0,.45);
    margin-bottom:18px;
  }
  .hero::after{ content:""; position:absolute; inset:0; opacity:.06; pointer-events:none;
    background:repeating-linear-gradient(90deg,#fff 0 2px,transparent 2px 68px); }
  .hero::before{ content:""; position:absolute; top:-70px; right:-40px;
    width:230px; height:230px; border-radius:50%; border:2px solid rgba(246,201,69,.25); }
  .hero h1{ font-family:var(--display); font-weight:800!important; font-size:2.9rem!important;
    line-height:1.02; letter-spacing:.5px; margin:0; text-transform:uppercase; color:var(--cream); }
  .hero h1 .yr{ color:var(--gold); }
  .hero p{ color:var(--muted); margin:.55rem 0 0; font-size:.92rem; max-width:640px; }
  .eyebrow{ display:inline-block; font-family:var(--mono); font-size:.68rem; letter-spacing:3px;
    text-transform:uppercase; color:var(--gold); background:rgba(246,201,69,.1);
    border:1px solid rgba(246,201,69,.3); padding:4px 10px; border-radius:20px; margin-bottom:14px; }
  h2,h3,h4{ color:var(--cream)!important; font-weight:800; }
  .round-head{ font-family:var(--mono); text-transform:uppercase; letter-spacing:2px;
    font-size:.7rem; font-weight:700; color:var(--gold); padding:6px 0 8px; margin-bottom:6px;
    border-bottom:1px solid var(--line); }
  div[data-testid="stVerticalBlockBorderWrapper"]{
    background:linear-gradient(180deg,var(--pitch2) 0%, var(--pitch) 100%)!important;
    border:1px solid var(--line)!important; border-radius:12px!important;
    box-shadow:0 3px 12px rgba(0,0,0,.32);
    transition:transform .14s ease, box-shadow .14s ease, border-color .14s ease; }
  div[data-testid="stVerticalBlockBorderWrapper"]:hover{
    transform:translateY(-3px); border-color:var(--gold)!important;
    box-shadow:0 10px 26px rgba(0,0,0,.55), 0 0 0 1px rgba(246,201,69,.15); }
  .team{ font-size:1rem; line-height:1.75rem; font-weight:500; color:var(--cream); }
  .team-win{ font-weight:800; color:#fff; }
  .team-win::before{ content:"\\25B8 "; color:var(--gold); }
  .team-lose{ color:var(--muted); }
  .team-tbd{ color:var(--muted); font-style:italic; font-size:.9rem; }
  .score-tag{ display:inline-block; margin-top:6px; font-family:var(--mono);
    font-weight:700; color:var(--gold); font-size:1.05rem; }
  .pick-tag{ display:inline-block; margin-top:6px; color:var(--muted); font-size:.78rem;
    font-family:var(--mono); }
  .stButton>button{ border-radius:8px; border:1px solid var(--line)!important;
    background:rgba(255,255,255,.04)!important; color:var(--cream)!important;
    font-weight:600; font-size:.82rem; }
  .stButton>button:hover{ border-color:var(--gold)!important; color:var(--gold)!important;
    background:rgba(246,201,69,.08)!important; }
  .stButton>button[kind="primary"]{
    background:linear-gradient(180deg,var(--gold) 0%,var(--gold2) 100%)!important;
    color:#241a00!important; border:none!important; font-size:.9rem; }
  .dossier-head{ font-family:var(--display); font-size:1.7rem; line-height:1.1; font-weight:800;
    text-transform:uppercase; color:var(--cream); margin-bottom:8px; }
  .dossier-head .vs{ display:block; font-size:.8rem; color:var(--gold); margin:2px 0;
    letter-spacing:2px; font-weight:600; }
  div[data-testid="stMetricValue"]{ color:var(--gold)!important; font-weight:800; font-family:var(--mono); }
  div[data-testid="stMetricLabel"]{ color:var(--muted)!important; }
  hr{ border-color:var(--line)!important; }
</style>
""", unsafe_allow_html=True)


# ---------------- cached model (fit once, shared by all users) ----------------
@st.cache_resource(show_spinner="Fitting the models (first load only)...")
def get_model():
    d = ps.load_data()
    model, da, dd = ps.fit_or_load(d)
    outcome = ps.load_outcome_model(d)
    return model, da, dd, outcome


# per-user private what-if edits: (home,away) -> {home_score,away_score,shootout_winner}
if "user_edits" not in st.session_state:
    st.session_state["user_edits"] = {}

official_df = ps.load_data()
official_sh = ps.load_shootouts()

# effective view = official results + THIS session's private edits
df = ps.apply_session_edits(official_df, st.session_state["user_edits"])
shootouts = ps.session_shootouts(official_sh, st.session_state["user_edits"])

model, da, dd, outcome = get_model()
tree = ps.build_bracket_tree(df, shootouts)


st.markdown(
    "<div class='hero'>"
    "<span class='eyebrow'>Predictive Model · Live</span>"
    "<h1>World Cup <span class='yr'>2026</span> Bracket</h1>"
    "<p>Match outcomes from a LightGBM classifier, scorelines from a "
    "Dixon-Coles Poisson model, and title odds from a Monte Carlo simulation "
    "of the knockout draw. Enter your own results to explore what-if brackets "
    "— your edits are private to you and reset on reload.</p>"
    "</div>",
    unsafe_allow_html=True)

# visitor what-if banner + reset
if st.session_state.get("user_edits"):
    c_note, c_reset = st.columns([4, 1])
    c_note.info(f"Exploring a what-if bracket with "
                f"{len(st.session_state['user_edits'])} of your own result(s). "
                "Private to you · resets on reload.")
    if c_reset.button("↺ Reset mine"):
        st.session_state["user_edits"] = {}
        st.session_state.pop("sim", None)
        st.rerun()


def match_pred(home, away):
    if home is None or away is None:
        return None
    return ps.predict(model, da, dd, home, away, neutral=True, outcome=outcome)


def card_summary(m):
    home_lbl = m["home"] if m["home"] else (m["home_src"] or "TBD")
    away_lbl = m["away"] if m["away"] else (m["away_src"] or "TBD")
    if m["home"] is None or m["away"] is None:
        return home_lbl, away_lbl, ""
    if m["played"]:
        tag = f"{m['hs']}-{m['as']}"
        if m["hs"] == m["as"] and m.get("shootout_winner"):
            tag += f" ({m['shootout_winner']} pens)"
        return home_lbl, away_lbl, tag
    pred = match_pred(m["home"], m["away"])
    pick = {"home": m["home"], "away": m["away"], "draw": "Draw"}[pred["pick"]]
    p = max(pred["p_home"], pred["p_draw"], pred["p_away"])
    return home_lbl, away_lbl, f"{pick} {p:.0%}"


def match_key(m):
    return f"{m['round_size']}_{m['idx']}"


@st.dialog("Match details", width="large")
def render_detail_panel(sel, open_key):
    st.markdown(
        f"<div class='dossier-head'>{sel['home']}"
        f"<span class='vs'>vs</span>{sel['away']}</div>",
        unsafe_allow_html=True)
    pred = match_pred(sel["home"], sel["away"])

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{sel['home']} win", f"{pred['p_home']:.0%}")
    c2.metric("Draw", f"{pred['p_draw']:.0%}")
    c3.metric(f"{sel['away']} win", f"{pred['p_away']:.0%}")
    st.caption(f"Likely score  ·  {sel['home']} {pred['likely_score'][0]}-"
               f"{pred['likely_score'][1]} {sel['away']}")

    grid = pred["grid"]
    gdf = pd.DataFrame(
        grid,
        index=[f"{sel['home']} {i}" for i in range(grid.shape[0])],
        columns=[f"{sel['away']} {j}" for j in range(grid.shape[1])])
    st.dataframe(gdf.style.format("{:.0%}").background_gradient(
        cmap="YlOrRd", axis=None), use_container_width=True)

    st.markdown("**Enter a what-if result** (private to your session)")
    e1, e2 = st.columns(2)
    dh = int(sel["hs"]) if sel["played"] and sel["hs"] is not None else 0
    dai = int(sel["as"]) if sel["played"] and sel["as"] is not None else 0
    hs = e1.number_input(f"{sel['home']} goals", 0, 20, dh, key=f"hs_{open_key}")
    as_ = e2.number_input(f"{sel['away']} goals", 0, 20, dai, key=f"as_{open_key}")

    shootout_winner = None
    if hs == as_:
        st.warning("Level tie — pick the penalty shootout winner (decides who "
                   "advances; the draw itself is kept for the scoreline).")
        shootout_winner = st.radio("Shootout winner", [sel["home"], sel["away"]],
                                   horizontal=True, key=f"so_{open_key}")

    b1, b2, b3 = st.columns(3)
    if b1.button("💾 Apply", type="primary", key=f"save_{open_key}",
                 use_container_width=True):
        st.session_state["user_edits"][(sel["home"], sel["away"])] = {
            "home_score": int(hs), "away_score": int(as_),
            "shootout_winner": shootout_winner}
        st.session_state.pop("sim", None)
        st.session_state.pop("open_match", None)
        st.rerun()

    if (sel["home"], sel["away"]) in st.session_state["user_edits"]:
        if b2.button("🗑️ Clear mine", key=f"clr_{open_key}",
                     use_container_width=True):
            st.session_state["user_edits"].pop((sel["home"], sel["away"]), None)
            st.session_state.pop("sim", None)
            st.session_state.pop("open_match", None)
            st.rerun()

    if b3.button("Close", key=f"close_{open_key}", use_container_width=True):
        st.session_state.pop("open_match", None)
        st.rerun()


# ---------------- win probabilities (manual button) ----------------
st.subheader("🏆 Tournament win probabilities")
sc1, sc2 = st.columns([1, 3])
n_sims = sc2.select_slider("Simulations", [2000, 5000, 10000, 20000],
                           value=10000, key="nsims")
if sc1.button("▶️ Simulate", type="primary"):
    with st.spinner(f"Simulating {n_sims:,} tournaments..."):
        ties = ps.knockout_bracket(df, shootouts)
        st.session_state["sim"] = ps.simulate_knockouts(
            model, da, dd, ties, n_sims=n_sims)

probs = st.session_state.get("sim")
if probs is None:
    st.info("Press **Simulate** to compute win probabilities from the current "
            "bracket (including any of your own what-if results).")
else:
    table = (pd.DataFrame(probs).T.sort_values("champion", ascending=False)
             .rename(columns={"champion": "Champion", "final": "Reach Final",
                              "semi": "Reach Semi"}))
    table = table[["Champion", "Reach Final", "Reach Semi"]]
    st.bar_chart(table["Champion"].head(16))
    st.dataframe(table.style.format("{:.1%}").background_gradient(
        cmap="Greens", subset=["Champion"]), use_container_width=True)


st.divider()
st.subheader("Knockout bracket")
st.caption("Click **Details** on any match to see predictions and enter a "
           "what-if result.")

# resolve open match for the dialog
open_key = st.session_state.get("open_match")
sel = None
if open_key:
    for r in tree:
        for m in r:
            if match_key(m) == open_key and m["home"] and m["away"]:
                sel = m

cols = st.columns(len(tree))
for round_matches, col in zip(tree, cols):
    with col:
        st.markdown(
            f"<div class='round-head'>{ps.ROUND_NAMES[round_matches[0]['round_size']]}</div>",
            unsafe_allow_html=True)
        for m in round_matches:
            home, away, tag = card_summary(m)
            with st.container(border=True):
                tbd = (m["home"] is None or m["away"] is None)
                if m["played"]:
                    w = ps._winner_of(m)
                    hc = "team team-win" if w == m["home"] else "team team-lose"
                    ac = "team team-win" if w == m["away"] else "team team-lose"
                    tag_html = f"<span class='score-tag'>{tag}</span>"
                elif tbd:
                    hc = ac = "team team-tbd"
                    tag_html = ""
                else:
                    hc = ac = "team"
                    tag_html = f"<span class='pick-tag'>{tag}</span>"
                st.markdown(
                    f"<div class='{hc}'>{home}</div>"
                    f"<div class='{ac}'>{away}</div>{tag_html}",
                    unsafe_allow_html=True)
                if m["home"] and m["away"]:
                    if st.button("Details", key=f"btn_{match_key(m)}",
                                 use_container_width=True):
                        st.session_state["open_match"] = match_key(m)
                        st.rerun()

if sel is not None:
    render_detail_panel(sel, open_key)