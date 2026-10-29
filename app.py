"""Streamlit UI for the AI start/sit assistant. Run: streamlit run app.py"""
import pandas as pd
import streamlit as st

import data
import llm

st.set_page_config(page_title="Start/Sit Assistant", page_icon="🏈", layout="wide")
st.title("🏈 AI Start/Sit Assistant")
st.caption("Data-driven analysis from nflverse stats + Gemini. Not a guarantee of results.")

# ---------- sidebar ----------
default_season, default_week = data.current_season_week()
with st.sidebar:
    st.header("Settings")
    season = st.number_input("Season", 2020, 2030, default_season)
    week = st.number_input("Week to start for", 1, 22, default_week)
    ppr = st.toggle("PPR scoring", value=True)
    st.caption("Stats come from games *before* this week.")


# ---------- player pickers ----------
def picker(label, key, default):
    """Text search box + dropdown of matches. Returns a player dict or None."""
    q = st.text_input(f"{label} — search name", default, key=f"q_{key}")
    matches = data.search_players(q, season) if q else []
    if not matches:
        st.warning("No players found.")
        return None
    return st.selectbox(label, matches, key=f"s_{key}",
                        format_func=lambda p: f"{p['name']} ({p['position']}, {p['team']})")


col_a, col_b = st.columns(2)
with col_a:
    player_a = picker("Player A", "a", "Hurts")
with col_b:
    player_b = picker("Player B", "b", "Lamar")


# ---------- helpers ----------
def show_result(res, title):
    """Render one recommendation card."""
    st.subheader(title)
    st.success(f"**Start: {res['recommendation']}**  ·  confidence: **{res['confidence']}**")
    for i, r in enumerate(res["reasons"], 1):
        st.markdown(f"{i}. {r}")
    st.info(f"⚠️ Risk: {res['risk_note']}")


def run_analysis(scn_a=None, scn_b=None, text=None):
    try:
        with st.spinner("Asking Gemini..."):
            return llm.analyze(st.session_state.bundle_a, st.session_state.bundle_b, scn_a, scn_b, text)
    except llm.LLMError as e:
        st.error(str(e))
        return None


# ---------- compare ----------
if st.button("Compare", type="primary", disabled=not (player_a and player_b)):
    if player_a["id"] == player_b["id"]:
        st.error("Pick two different players.")
    else:
        with st.spinner("Loading stats..."):
            st.session_state.bundle_a = data.get_player_bundle(player_a, season, week, ppr)
            st.session_state.bundle_b = data.get_player_bundle(player_b, season, week, ppr)
        st.session_state.result = run_analysis()
        st.session_state.whatif = None  # clear any old scenario result

if st.session_state.get("result"):
    ba, bb = st.session_state.bundle_a, st.session_state.bundle_b
    show_result(st.session_state.result, "Recommendation")

    # data notes (bye week, small sample, missing injury entry...)
    for b in (ba, bb):
        notes = b["recent_form"]["notes"] + b["matchup"].get("notes", []) + b["injury"].get("notes", [])
        for n in notes:
            st.caption(f"ℹ️ {b['name']}: {n}")

    # bar chart of recent fantasy points
    st.subheader("Recent fantasy points")
    chart = {}
    for b in (ba, bb):
        for g in b["recent_form"]["games"]:
            chart.setdefault(f"{g['season']} W{g['week']:02d}", {})[b["name"]] = g["fantasy_points"]
    if chart:
        st.bar_chart(pd.DataFrame(chart).T.sort_index())

    # stat tables + matchup/injury
    tcols = st.columns(2)
    for col, b in zip(tcols, (ba, bb)):
        with col:
            st.markdown(f"**{b['name']}**")
            if b["recent_form"]["games"]:
                st.dataframe(pd.DataFrame(b["recent_form"]["games"]), hide_index=True)
                st.write("Averages:", b["recent_form"]["averages"])
            m, inj = b["matchup"], b["injury"]
            st.write(f"Opponent: **{m.get('opponent') or 'none (bye?)'}** ({m.get('home_away', '-')}) · "
                     f"Defense allows **{m.get('def_pts_allowed_to_position')}** pts/gm to {m.get('position')} "
                     f"(league avg {m.get('league_avg_pts_allowed_to_position')}, rank {m.get('def_rank_vs_position')})")
            st.write(f"Injury report: **{inj.get('status')}**")

    # ---------- What If ----------
    st.divider()
    st.header("🔮 What If?")
    injury_opts = ["Unchanged", "Healthy", "Questionable", "Out"]
    wa, wb = st.columns(2)
    scn, parts = [], []
    for col, b, tag in ((wa, ba, "a"), (wb, bb, "b")):
        with col:
            st.markdown(f"**{b['name']}**")
            inj = st.selectbox("Injury status", injury_opts, key=f"inj_{tag}")
            dd = st.slider("Opposing defense (− tougher / + easier), fantasy pts", -10.0, 10.0, 0.0, 0.5, key=f"dd_{tag}")
            us = st.slider("Reduce expected usage/snaps (%)", 0, 50, 0, 5, key=f"us_{tag}")
            s = {"injury_status": None if inj == "Unchanged" else inj,
                 "defense_delta": dd, "usage_reduction_pct": us}
            scn.append(s)
            if inj != "Unchanged":
                parts.append(f"{b['name']} injury status is {inj}")
            if dd:
                parts.append(f"{b['name']}'s opponent defense is {'easier' if dd > 0 else 'tougher'} by {abs(dd)} pts")
            if us:
                parts.append(f"{b['name']}'s usage reduced {us}%")

    if st.button("Run scenario"):
        if not parts:
            st.warning("Change at least one control first.")
        else:
            text = "; ".join(parts)
            res = run_analysis(scn[0], scn[1], text)
            st.session_state.whatif = {"res": res, "text": text} if res else None

    wi = st.session_state.get("whatif")
    if wi:
        st.caption(f"Scenario applied: {wi['text']}")
        orig, new = st.session_state.result, wi["res"]
        if orig["recommendation"] != new["recommendation"]:
            st.warning(f"🔄 Recommendation FLIPPED: {orig['recommendation']} → {new['recommendation']}")
        else:
            st.info(f"✅ Recommendation unchanged ({new['recommendation']}); confidence {orig['confidence']} → {new['confidence']}")
        c1, c2 = st.columns(2)
        with c1:
            show_result(orig, "Original")
        with c2:
            show_result(new, "With scenario")
