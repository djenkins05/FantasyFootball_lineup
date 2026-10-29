# AI Start/Sit Assistant

Compare two NFL players and let Gemini recommend who to start, using only real stats passed to it. Includes a "What If?" simulator that re-runs the analysis under a changed scenario and shows the original vs. new pick side by side.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your key in GEMINI_API_KEY
streamlit run app.py
```
Get a key at https://aistudio.google.com/apikey. `.env` is git-ignored; never commit it.
Optional: set `GEMINI_MODEL` in `.env` (default `gemini-3.5-flash`; a fallback model is used if it is overloaded).

## How it works
1. `data.py` loads nflverse data with `nflreadpy` (player stats, schedule, injuries), cached in memory and on disk (`.cache/`), so a network blip won't break a demo.
   - Recent form: last 5 games with PPR/standard fantasy points and position-specific stats.
   - Matchup: next opponent, home/away, and average fantasy points that defense allowed to the player's position over its last 5 games (plus league average and rank).
   - Injury: latest injury report status.
2. `llm.py` builds a JSON payload for both players and sends it to Gemini (temperature 0.2, JSON schema output). The system prompt tells it to use ONLY the supplied numbers. Output is validated (valid player name, 3 reasons, confidence level).
3. `app.py` is the Streamlit UI: sidebar (season/week/PPR), two player pickers, results card, bar chart, stat tables.
4. **What If**: override injury status, shift the opposing defense's points allowed, or reduce usage. The overrides are applied to the data and described to Gemini as "Scenario applied: ...". The UI shows whether the recommendation flipped.

## Known limitations
- Only uses nflverse stats, with no news, weather, Vegas lines, or depth charts. Gemini is told not to use outside knowledge.
- Small samples (5 games, early season) make averages noisy; bye weeks or missed games are flagged but not adjusted for.
- QB/RB/WR/TE only.
- Defense-vs-position is a simple recent average, not adjusted for opponent strength.
- The "reduce usage" scenario is interpreted by the LLM, not modeled numerically.
- Injury data depends on nflverse publishing it for the selected week.
- Analysis only, not a guarantee of results.

## Data attribution
Data from [nflverse](https://github.com/nflverse) via `nflreadpy`, licensed CC-BY 4.0.
