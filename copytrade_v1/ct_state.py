from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_STATE: Dict[str, Any] = {
    "token_map": {},
    "open_orders": {},
    "last_sync_ts": 0,
    "target_last_shares": {},
    "target_last_seen_ts": {},
    "target_missing_streak": {},
    "cooldown_until": {},
    "target_last_event_ts": {},
}


def load_state(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return dict(DEFAULT_STATE)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return dict(DEFAULT_STATE)
    state = dict(DEFAULT_STATE)
    if isinstance(payload, dict):
        state.update(payload)
    if "token_map" not in state or not isinstance(state["token_map"], dict):
        state["token_map"] = {}
    if "open_orders" not in state or not isinstance(state["open_orders"], dict):
        state["open_orders"] = {}
    if "target_last_shares" not in state or not isinstance(state["target_last_shares"], dict):
        state["target_last_shares"] = {}
    if "target_last_seen_ts" not in state or not isinstance(state["target_last_seen_ts"], dict):
        state["target_last_seen_ts"] = {}
    if "target_missing_streak" not in state or not isinstance(state["target_missing_streak"], dict):
        state["target_missing_streak"] = {}
    if "cooldown_until" not in state or not isinstance(state["cooldown_until"], dict):
        state["cooldown_until"] = {}
    if "target_last_event_ts" not in state or not isinstance(state["target_last_event_ts"], dict):
        state["target_last_event_ts"] = {}
    return state


def save_state(path: str, state: Dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
