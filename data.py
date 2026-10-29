"""Data layer: pulls nflverse data via nflreadpy and shapes it for the LLM.

Every public function returns plain dicts/lists (JSON-friendly) and puts
human-readable problems (bye week, rookie, no injury entry...) in a "notes"
list instead of raising.
"""
import difflib
import functools
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
POSITIONS = ["QB", "RB", "WR", "TE"]

# Key stats shown/sent per position (column names verified against load_player_stats)
POSITION_STATS = {
    "QB": ["completions", "attempts", "passing_yards", "passing_tds",
           "passing_interceptions", "carries", "rushing_yards", "rushing_tds"],
    "RB": ["carries", "rushing_yards", "rushing_tds", "targets", "receptions",
           "receiving_yards", "receiving_tds"],
    "WR": ["targets", "receptions", "receiving_yards", "receiving_tds", "target_share"],
    "TE": ["targets", "receptions", "receiving_yards", "receiving_tds", "target_share"],
}


# ---------- loading + caching ----------
def _cached(name, season, loader):
    """Load a DataFrame from nflverse; fall back to the last on-disk copy if offline."""
    path = CACHE_DIR / f"{name}_{season}.pkl"
    try:
        df = loader([season]).to_pandas()
        df.to_pickle(path)
        return df
    except Exception:
        if path.exists():
            return pd.read_pickle(path)
        return pd.DataFrame()


@functools.lru_cache(maxsize=8)
def load_stats(season):
    return _cached("stats", season, nfl.load_player_stats)


@functools.lru_cache(maxsize=8)
def load_schedule(season):
    return _cached("schedule", season, nfl.load_schedules)


@functools.lru_cache(maxsize=8)
def load_injury(season):
    return _cached("injuries", season, nfl.load_injuries)


def current_season_week():
    """(season, week) according to nflreadpy, with a safe fallback."""
    try:
        return int(nfl.get_current_season()), int(nfl.get_current_week())
    except Exception:
        return 2025, 1


def _points_col(ppr):
    return "fantasy_points_ppr" if ppr else "fantasy_points"


def _all_games(season):
    """Regular-season stat rows for this and last season (so early weeks still have history)."""
    frames = [load_stats(s) for s in (season - 1, season)]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if df.empty:
        return df
    df = df[df["season_type"] == "REG"]
    return df[df["position"].isin(POSITIONS)]


# ---------- public functions ----------
def search_players(query, season, limit=15):
    """Partial / fuzzy name search. Returns [{id, name, team, position}] (most recent team)."""
    df = load_stats(season)
    if df.empty:
        df = load_stats(season - 1)
    df = df[df["position"].isin(POSITIONS)]
    if df.empty or not query.strip():
        return []
    # one row per player, latest game = current team
    latest = df.sort_values(["season", "week"]).groupby("player_id").tail(1)
    names = latest["player_display_name"]
    q = query.strip().lower()
    hits = latest[names.str.lower().str.contains(q, regex=False)]
    if hits.empty:  # fall back to fuzzy matching for typos
        # score against full name and last name, so "Mahoms" finds "Patrick Mahomes"
        def score(full):
            parts = [full.lower(), full.lower().split()[-1]]
            return max(difflib.SequenceMatcher(None, q, p).ratio() for p in parts)
        scores = names.map(score)
        hits = latest[scores >= 0.7].assign(_s=scores).sort_values("_s", ascending=False)
    return [
        {"id": r.player_id, "name": r.player_display_name, "team": r.team, "position": r.position}
        for r in hits.head(limit).itertuples()
    ]


def _player_rows(player_id, season, week):
    """All of a player's games strictly before (season, week), oldest first."""
    df = _all_games(season)
    df = df[df["player_id"] == player_id]
    before = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    return df[before].sort_values(["season", "week"])


def get_recent_form(player_id, season, week, n=5, ppr=True):
    """Last n games with fantasy points + position-specific stats and averages."""
    notes = []
    rows = _player_rows(player_id, season, week)
    if rows.empty:
        return {"games": [], "averages": {}, "position": None,
                "notes": ["No games found before this week (rookie, or no data yet)."]}
    pos = rows.iloc[-1]["position"]
    last = rows.tail(n)
    if len(last) < n:
        notes.append(f"Only {len(last)} game(s) of history available (small sample).")
    # flag gaps (bye / missed games) within the window for context
    weeks = last[last["season"] == season]["week"].tolist()
    if weeks and week - max(weeks) > 1:
        notes.append(f"Last game was week {max(weeks)}; missed week(s) since (bye or absence).")

    pts = _points_col(ppr)
    cols = POSITION_STATS[pos]
    games = []
    for r in last.itertuples():
        g = {"season": int(r.season), "week": int(r.week), "opponent": r.opponent_team,
             "fantasy_points": round(float(getattr(r, pts)), 1)}
        for c in cols:
            v = getattr(r, c)
            g[c] = None if pd.isna(v) else round(float(v), 2 if c == "target_share" else 1)
        games.append(g)
    averages = {"fantasy_points": round(float(last[pts].mean()), 1)}
    for c in cols:
        averages[c] = round(float(last[c].mean()), 2 if c == "target_share" else 1)
    return {"games": games, "averages": averages, "position": pos, "notes": notes}


def get_matchup(player_id, season, week, ppr=True, window=5):
    """Next opponent + how that defense has fared against the player's position."""
    notes = []
    rows = _player_rows(player_id, season, week)
    if rows.empty:
        return {"notes": ["Cannot determine team/position: no game history."]}
    pos, team = rows.iloc[-1]["position"], rows.iloc[-1]["team"]

    sched = load_schedule(season)
    if sched.empty:
        return {"position": pos, "team": team, "notes": ["Schedule unavailable."]}
    game = sched[(sched["week"] == week) & ((sched["home_team"] == team) | (sched["away_team"] == team))]
    if game.empty:
        return {"position": pos, "team": team, "opponent": None,
                "notes": [f"{team} has no game in week {week} (bye week)."]}
    g = game.iloc[0]
    home = g["home_team"] == team
    opp = g["away_team"] if home else g["home_team"]

    # Fantasy points allowed to this position: sum all players at the position
    # per (opponent defense, week), then average over the last `window` weeks.
    pts = _points_col(ppr)
    df = _all_games(season)
    df = df[df["position"] == pos]
    before = (df["season"] < season) | ((df["season"] == season) & (df["week"] < week))
    df = df[before]
    per_week = (df.groupby(["opponent_team", "season", "week"])[pts].sum().reset_index()
                  .sort_values(["season", "week"]))
    recent = per_week.groupby("opponent_team").tail(window)
    by_def = recent.groupby("opponent_team")[pts].mean().sort_values(ascending=False)
    if opp not in by_def.index:
        notes.append(f"No defensive history for {opp} vs {pos}.")
        allowed, rank = None, None
    else:
        allowed = round(float(by_def[opp]), 1)
        rank = int(list(by_def.index).index(opp)) + 1  # 1 = allows the most points
    return {
        "position": pos, "team": team, "week": week, "opponent": opp,
        "home_away": "home" if home else "away",
        "def_pts_allowed_to_position": allowed,
        "league_avg_pts_allowed_to_position": round(float(by_def.mean()), 1),
        "def_rank_vs_position": f"{rank} of {len(by_def)} (1 = most points allowed)" if rank else None,
        "games_sampled": f"last {window} games",
        "notes": notes,
    }


def get_injury_status(player_id, season, week):
    """Latest injury report status at or before `week` (this season)."""
    inj = load_injury(season)
    if inj.empty:
        return {"status": None, "notes": ["Injury data unavailable."]}
    df = inj[(inj["gsis_id"] == player_id) & (inj["week"] <= week)].sort_values("week")
    if df.empty:
        return {"status": "No injury report entry", "notes": ["No injury report entry (likely healthy)."]}
    r = df.iloc[-1]
    # An entry from an earlier week is stale (the player may have played since),
    # so only treat it as the current status if it's for the selected week.
    if int(r["week"]) < week:
        return {"status": "No injury report entry for this week",
                "notes": [f"Last injury report entry was week {int(r['week'])} "
                          f"({r['report_status']}, {r['report_primary_injury']}); not current."]}
    status = r["report_status"] if pd.notna(r["report_status"]) else "No game status listed"
    return {
        "status": status, "report_week": int(r["week"]),
        "injury": r["report_primary_injury"] if pd.notna(r["report_primary_injury"]) else None,
        "practice_status": r["practice_status"] if pd.notna(r["practice_status"]) else None,
        "notes": [],
    }


def get_player_bundle(player, season, week, ppr=True):
    """Everything the LLM and UI need for one player. `player` = search_players() dict."""
    pid = player["id"]
    return {
        "name": player["name"], "id": pid,
        "recent_form": get_recent_form(pid, season, week, ppr=ppr),
        "matchup": get_matchup(pid, season, week, ppr=ppr),
        "injury": get_injury_status(pid, season, week),
    }


if __name__ == "__main__":  # terminal smoke test: python data.py
    import json
    season, week = current_season_week()
    print("season/week:", season, week)
    for q in ["Hurts", "Lamar"]:
        p = search_players(q, season)[0]
        print(json.dumps(get_player_bundle(p, season, week), indent=1, default=str))
    print("search typo 'Mahomes':", search_players("Mahoms", season)[:2])
