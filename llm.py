"""LLM layer: sends both players' data to Gemini and returns a validated recommendation."""
import copy
import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Override with GEMINI_MODEL in .env. (Verified available via client.models.list().)
DEFAULT_MODEL = "gemini-3.5-flash"  # 3.8-flash was ~15x slower in testing
FALLBACK_MODEL = "gemini-3.7-flash"  # used if the main model is overloaded

SYSTEM_INSTRUCTION = (
    "You are a fantasy football analyst. Use ONLY the numbers provided in the data. "
    "Do not use outside knowledge of stats, injuries, or news. If data is missing, say so. "
    "If a 'scenario' is provided, treat it as true and factor it in explicitly."
)

# Structured-output schema Gemini must follow
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendation": {"type": "string", "description": "Exact name of the player to start"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasons": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 3},
        "risk_note": {"type": "string"},
    },
    "required": ["recommendation", "confidence", "reasons", "risk_note"],
}


class LLMError(Exception):
    """Raised with a user-readable message for any API / output problem."""


def build_payload(bundle, scenario=None):
    """Compact per-player JSON for the prompt. `scenario` = overrides dict for this player."""
    b = copy.deepcopy(bundle)
    matchup, injury = b["matchup"], b["injury"]
    payload = {
        "name": b["name"],
        "position": b["recent_form"].get("position"),
        "last_games": b["recent_form"]["games"],
        "averages_last_games": b["recent_form"]["averages"],
        "opponent": matchup.get("opponent"),
        "home_away": matchup.get("home_away"),
        "defense_vs_position": {
            "avg_fantasy_pts_allowed": matchup.get("def_pts_allowed_to_position"),
            "league_avg_pts_allowed": matchup.get("league_avg_pts_allowed_to_position"),
            "rank": matchup.get("def_rank_vs_position"),
        },
        "injury_report": {k: injury.get(k) for k in ("status", "injury", "practice_status")},
        "data_notes": b["recent_form"]["notes"] + matchup.get("notes", []) + injury.get("notes", []),
    }
    # Apply what-if overrides to the numbers themselves, and record them in plain text
    if scenario:
        applied = []
        if scenario.get("injury_status"):
            payload["injury_report"]["status"] = scenario["injury_status"]
            applied.append(f"injury status set to {scenario['injury_status']}")
        delta = scenario.get("defense_delta", 0)
        if delta and payload["defense_vs_position"]["avg_fantasy_pts_allowed"] is not None:
            payload["defense_vs_position"]["avg_fantasy_pts_allowed"] = round(
                payload["defense_vs_position"]["avg_fantasy_pts_allowed"] + delta, 1)
            applied.append(f"opposing defense allows {delta:+.1f} fantasy pts vs this position "
                           f"({'easier' if delta > 0 else 'tougher'})")
        if scenario.get("usage_reduction_pct"):
            applied.append(f"expected usage/snap share reduced by {scenario['usage_reduction_pct']}% "
                           "relative to recent games")
        payload["scenario_applied"] = applied
    return payload


def analyze(bundle_a, bundle_b, scenario_a=None, scenario_b=None, scenario_text=None):
    """Ask Gemini who to start. Returns dict with recommendation/confidence/reasons/risk_note."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file (see .env.example).")
    model = os.getenv("GEMINI_MODEL") or DEFAULT_MODEL

    players = [build_payload(bundle_a, scenario_a), build_payload(bundle_b, scenario_b)]
    names = [p["name"] for p in players]
    prompt = (
        f"Decide which player to start this week: {names[0]} or {names[1]}.\n"
        "Give exactly 3 reasons, each citing specific numbers from the data.\n"
    )
    if scenario_text:
        prompt += f"Scenario applied: {scenario_text}\n"
    prompt += "\nDATA (JSON):\n" + json.dumps({"players": players}, indent=1, default=str)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.2,
        response_mime_type="application/json",
        response_json_schema=RESPONSE_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_level="low"),  # keep latency down for demos
    )
    # Retry transient overloads (503/429), then fall back to a stable model
    resp, last_err = None, None
    for m in [model, model, FALLBACK_MODEL]:
        try:
            resp = client.models.generate_content(model=m, contents=prompt, config=config)
            break
        except Exception as e:  # network, auth, quota, bad model name...
            last_err = e
            if not any(c in str(e) for c in ("503", "429", "UNAVAILABLE")):
                break
            time.sleep(2)
    if resp is None:
        raise LLMError(f"Gemini API error ({model}): {last_err}") from last_err

    # Validate the output rather than trusting it
    try:
        out = json.loads(resp.text)
        assert out["recommendation"] in names, f"recommendation '{out['recommendation']}' is not one of {names}"
        assert out["confidence"] in ("low", "medium", "high"), "bad confidence value"
        assert isinstance(out["reasons"], list) and len(out["reasons"]) == 3, "expected exactly 3 reasons"
        out.setdefault("risk_note", "")
    except Exception as e:
        raise LLMError(f"Gemini returned malformed output ({e}). Try again. Raw: {str(resp.text)[:300]}") from e
    return out


if __name__ == "__main__":  # terminal test: python llm.py
    import data
    season, week = data.current_season_week()
    a = data.get_player_bundle(data.search_players("Hurts", season)[0], season, week)
    b = data.get_player_bundle(data.search_players("Lamar", season)[0], season, week)
    print(json.dumps(analyze(a, b), indent=1))
