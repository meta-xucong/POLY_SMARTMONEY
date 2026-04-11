import sys

sys.path.insert(0, ".")

from copytrade_run import (
    _estimate_recovery_shares_from_state,
    _mark_must_exit_token,
    _should_clear_must_exit_without_inventory,
)


def test_mark_must_exit_token_upsert():
    state = {}
    _mark_must_exit_token(state, "t1", 100, "sell_action", target_sell_ms=1000)
    _mark_must_exit_token(state, "t1", 120, "reconcile_loop", target_sell_ms=900)
    meta = state.get("must_exit_tokens", {}).get("t1")
    assert isinstance(meta, dict), meta
    assert int(meta.get("first_ts") or 0) == 100
    assert int(meta.get("last_ts") or 0) == 120
    assert int(meta.get("target_sell_ms") or 0) == 1000


def test_estimate_recovery_shares_prefers_max_source():
    state = {
        "last_nonzero_my_shares": {
            "t1": {"shares": 3.0, "ts": 100},
        },
        "open_orders": {
            "t1": [
                {"side": "BUY", "size": 10.0},
                {"side": "SELL", "size": 5.0},
            ]
        },
    }
    est = _estimate_recovery_shares_from_state(state, "t1")
    assert abs(est - 5.0) < 1e-9, est


def test_should_clear_must_exit_guarded_by_recent_cache():
    cfg = {"must_exit_cache_hold_sec": 600}
    state = {
        "buy_notional_accumulator": {"t1": {"usd": 0.0}},
        "last_nonzero_my_shares": {"t1": {"shares": 2.0, "ts": 1000}},
    }
    should_clear_recent = _should_clear_must_exit_without_inventory(
        state=state,
        token_id="t1",
        now_ts=1200,
        eps=1e-9,
        cfg=cfg,
    )
    assert not should_clear_recent
    should_clear_stale = _should_clear_must_exit_without_inventory(
        state=state,
        token_id="t1",
        now_ts=2000,
        eps=1e-9,
        cfg=cfg,
    )
    assert should_clear_stale


if __name__ == "__main__":
    test_mark_must_exit_token_upsert()
    test_estimate_recovery_shares_prefers_max_source()
    test_should_clear_must_exit_guarded_by_recent_cache()
    print("ALL MUST_EXIT TESTS PASSED")
