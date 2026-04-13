import sys

sys.path.insert(0, ".")

from copytrade_run import (
    _is_must_exit_fresh,
    _estimate_recovery_shares_from_state,
    _mark_must_exit_token,
    _should_clear_stale_must_exit_on_buy,
    _should_clear_must_exit_without_inventory,
)


def test_mark_must_exit_token_upsert():
    state = {}
    _mark_must_exit_token(state, "t1", 100, "target_sell_action", target_sell_ms=1000)
    _mark_must_exit_token(state, "t1", 120, "reconcile_loop", target_sell_ms=0)
    meta = state.get("must_exit_tokens", {}).get("t1")
    assert isinstance(meta, dict), meta
    assert int(meta.get("first_ts") or 0) == 100
    assert int(meta.get("last_ts") or 0) == 100
    assert int(meta.get("target_sell_ms") or 0) == 1000
    assert str(meta.get("source") or "") == "target_sell_action"


def test_must_exit_freshness_and_stale_clear_gate():
    meta = {
        "first_ts": 100,
        "last_ts": 100,
        "source": "reconcile_loop",
        "target_sell_ms": 1000,
    }
    assert _is_must_exit_fresh(
        meta=meta,
        last_target_sell_ms=0,
        now_ms=2000,
        fresh_window_sec=5,
    )
    assert not _is_must_exit_fresh(
        meta=meta,
        last_target_sell_ms=0,
        now_ms=120000,
        fresh_window_sec=5,
    )
    assert _should_clear_stale_must_exit_on_buy(
        must_exit_active=True,
        must_exit_fresh=False,
        t_now_present=True,
        t_now=38.0,
        has_buy=True,
        buy_sum=8.0,
        min_target_buy_shares=1.0,
    )
    assert not _should_clear_stale_must_exit_on_buy(
        must_exit_active=True,
        must_exit_fresh=False,
        t_now_present=True,
        t_now=38.0,
        has_buy=False,
        buy_sum=8.0,
        min_target_buy_shares=1.0,
    )
    assert not _should_clear_stale_must_exit_on_buy(
        must_exit_active=True,
        must_exit_fresh=True,
        t_now_present=True,
        t_now=38.0,
        has_buy=True,
        buy_sum=8.0,
        min_target_buy_shares=1.0,
    )


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
    test_must_exit_freshness_and_stale_clear_gate()
    test_estimate_recovery_shares_prefers_max_source()
    test_should_clear_must_exit_guarded_by_recent_cache()
    print("ALL MUST_EXIT TESTS PASSED")
