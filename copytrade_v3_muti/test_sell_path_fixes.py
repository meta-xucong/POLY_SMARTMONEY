import sys
import logging

sys.path.insert(0, ".")

from ct_exec import reconcile_one


logger = logging.getLogger("test_sell_path_fixes")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)


def _base_cfg():
    return {
        "deadband_shares": 0.0,
        "order_size_mode": "fixed_shares",
        "slice_min": 0.0,
        "slice_max": 9999.0,
        "min_order_usd": 1.0,
        "min_order_shares": 5.0,
        "tick_size": 0.01,
        "taker_enabled": True,
        "taker_spread_threshold": 0.01,
        "exit_full_sell": True,
        "allow_short": False,
        "sell_available_buffer_shares": 0.01,
        "sell_accumulator_ttl_sec": 3600,
        "dust_exit_eps": 0.2,
    }


def test_exit_exact_min_not_swallowed():
    cfg = _base_cfg()
    state = {"topic_state": {"t1": {"phase": "EXITING"}}}
    orderbook = {"best_bid": 0.60, "best_ask": 0.61}
    actions = reconcile_one(
        token_id="t1",
        desired_shares=0.0,
        my_shares=5.0,
        orderbook=orderbook,
        open_orders=[],
        now_ts=1000,
        cfg=cfg,
        state=state,
        planned_token_notional=0.0,
    )
    place_actions = [a for a in actions if a.get("type") == "place"]
    assert place_actions, f"Expected SELL place action, got: {actions}"
    place = place_actions[0]
    assert str(place.get("side")).upper() == "SELL", place
    assert float(place.get("size") or 0.0) >= 4.999, place
    assert "dust_exits" not in state or "t1" not in state.get("dust_exits", {}), state
    print("[PASS] exit_exact_min_not_swallowed")


def test_non_exit_sell_accumulator_not_zero_add():
    cfg = _base_cfg()
    state = {
        "topic_state": {},
        "market_status_cache": {"t1": {"meta": {"orderMinSize": "5"}}},
    }
    orderbook = {"best_bid": 0.55, "best_ask": 0.56}
    actions = reconcile_one(
        token_id="t1",
        desired_shares=6.8,  # delta = -1.2 (small SELL)
        my_shares=8.0,
        orderbook=orderbook,
        open_orders=[],
        now_ts=1001,
        cfg=cfg,
        state=state,
        planned_token_notional=0.0,
    )
    assert actions == [], actions
    acc = state.get("sell_shares_accumulator", {}).get("t1", {})
    shares = float(acc.get("shares") or 0.0)
    assert shares > 1.1, acc
    print("[PASS] non_exit_sell_accumulator_not_zero_add")


def test_exiting_below_min_holds_not_dust_when_above_eps():
    cfg = _base_cfg()
    cfg["taker_enabled"] = False
    state = {
        "topic_state": {"t1": {"phase": "EXITING"}},
        "market_status_cache": {"t1": {"meta": {"orderMinSize": "5"}}},
    }
    orderbook = {"best_bid": 0.50, "best_ask": 0.51}
    actions = reconcile_one(
        token_id="t1",
        desired_shares=0.0,
        my_shares=3.0,
        orderbook=orderbook,
        open_orders=[],
        now_ts=1002,
        cfg=cfg,
        state=state,
        planned_token_notional=0.0,
    )
    assert actions == [], actions
    assert "t1" in state.get("topic_state", {}), state
    assert "t1" not in state.get("dust_exits", {}), state
    print("[PASS] exiting_below_min_holds_not_dust_when_above_eps")


def test_true_dust_can_be_cleared():
    cfg = _base_cfg()
    cfg["taker_enabled"] = False
    state = {
        "topic_state": {"t1": {"phase": "EXITING"}},
        "market_status_cache": {"t1": {"meta": {"orderMinSize": "5"}}},
    }
    orderbook = {"best_bid": 0.40, "best_ask": 0.41}
    actions = reconcile_one(
        token_id="t1",
        desired_shares=0.0,
        my_shares=0.1,
        orderbook=orderbook,
        open_orders=[],
        now_ts=1003,
        cfg=cfg,
        state=state,
        planned_token_notional=0.0,
    )
    assert actions == [], actions
    assert "t1" in state.get("dust_exits", {}), state
    assert "t1" not in state.get("topic_state", {}), state
    print("[PASS] true_dust_can_be_cleared")


if __name__ == "__main__":
    test_exit_exact_min_not_swallowed()
    test_non_exit_sell_accumulator_not_zero_add()
    test_exiting_below_min_holds_not_dust_when_above_eps()
    test_true_dust_can_be_cleared()
    print("\nALL SELL PATH FIX TESTS PASSED")
