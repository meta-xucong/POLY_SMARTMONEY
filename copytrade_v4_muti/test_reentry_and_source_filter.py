import sys

sys.path.insert(0, ".")

from copytrade_run import (
    _normalize_idle_conflicting_actions,
    _prepare_sell_health_monitor_for_run,
    _should_accept_buy_action_source,
    _should_count_sell_health_signal,
    _should_hold_reentry_buy,
)


def test_reentry_hold_inside_window_without_force():
    hold, reason = _should_hold_reentry_buy(
        now_ts=1100,
        my_shares=0.0,
        last_exit_ts=1000,
        reentry_cooldown_sec=150,
        signal_buy_shares=3.0,
        order_buy_shares=3.0,
        order_buy_usd=2.4,
        force_buy_shares=8.0,
        force_buy_usd=6.0,
        eps=1e-9,
    )
    assert hold is True and reason == "cooldown_hold", (hold, reason)


def test_reentry_bypass_on_force_signal():
    hold, reason = _should_hold_reentry_buy(
        now_ts=1080,
        my_shares=0.0,
        last_exit_ts=1000,
        reentry_cooldown_sec=150,
        signal_buy_shares=10.0,
        order_buy_shares=5.0,
        order_buy_usd=4.0,
        force_buy_shares=8.0,
        force_buy_usd=6.0,
        eps=1e-9,
    )
    assert hold is False and reason == "force_override", (hold, reason)


def test_reentry_outside_window_or_has_inventory():
    hold, reason = _should_hold_reentry_buy(
        now_ts=1300,
        my_shares=0.0,
        last_exit_ts=1000,
        reentry_cooldown_sec=150,
        signal_buy_shares=3.0,
        order_buy_shares=3.0,
        order_buy_usd=2.4,
        force_buy_shares=8.0,
        force_buy_usd=6.0,
        eps=1e-9,
    )
    assert hold is False and reason == "outside_window", (hold, reason)
    hold, reason = _should_hold_reentry_buy(
        now_ts=1080,
        my_shares=1.0,
        last_exit_ts=1000,
        reentry_cooldown_sec=150,
        signal_buy_shares=3.0,
        order_buy_shares=3.0,
        order_buy_usd=2.4,
        force_buy_shares=8.0,
        force_buy_usd=6.0,
        eps=1e-9,
    )
    assert hold is False and reason == "has_inventory", (hold, reason)


def test_buy_source_filter_position_source_mode():
    assert _should_accept_buy_action_source(
        "position_source_consistent",
        "0xaaa",
        "0xaaa",
    )
    assert not _should_accept_buy_action_source(
        "position_source_consistent",
        "0xbbb",
        "0xaaa",
    )
    # No preferred source yet -> allow fresh BUY discovery.
    assert _should_accept_buy_action_source(
        "position_source_consistent",
        "0xbbb",
        "",
    )
    # Mode disabled -> always allow.
    assert _should_accept_buy_action_source("all", "0xbbb", "0xaaa")


def test_sell_health_signal_requires_live_inventory_only():
    assert not _should_count_sell_health_signal(
        has_sell=True,
        my_shares=0.0,
        open_sell_orders_count=0,
        eps=1e-9,
    )
    assert _should_count_sell_health_signal(
        has_sell=True,
        my_shares=1.0,
        open_sell_orders_count=0,
        eps=1e-9,
    )
    assert not _should_count_sell_health_signal(
        has_sell=True,
        my_shares=0.0,
        open_sell_orders_count=1,
        eps=1e-9,
    )


def test_sell_health_monitor_resets_on_new_run():
    state = {
        "run_start_ms": 1000,
        "sell_health_monitor": {
            "start_ts": 10,
            "signals": 42,
            "actions": 0,
        },
    }
    _prepare_sell_health_monitor_for_run(state, 2000)
    assert state["run_start_ms"] == 2000
    assert "sell_health_monitor" not in state

    state["sell_health_monitor"] = {"signals": 3}
    _prepare_sell_health_monitor_for_run(state, 2000)
    assert state["sell_health_monitor"] == {"signals": 3}


def test_idle_conflicting_actions_keep_only_net_buy():
    has_buy, buy_sum, has_sell, sell_sum, resolution = _normalize_idle_conflicting_actions(
        phase="IDLE",
        my_shares=0.0,
        open_orders_count=0,
        has_buy=True,
        buy_sum=10.0,
        has_sell=True,
        sell_sum=4.0,
        eps=1e-9,
    )
    assert (has_buy, round(buy_sum, 6), has_sell, sell_sum, resolution) == (
        True,
        6.0,
        False,
        0.0,
        "keep_net_buy",
    )


def test_idle_conflicting_actions_drop_non_actionable_sell_bias():
    has_buy, buy_sum, has_sell, sell_sum, resolution = _normalize_idle_conflicting_actions(
        phase="IDLE",
        my_shares=0.0,
        open_orders_count=0,
        has_buy=True,
        buy_sum=3.0,
        has_sell=True,
        sell_sum=5.0,
        eps=1e-9,
    )
    assert (has_buy, buy_sum, has_sell, sell_sum, resolution) == (
        False,
        0.0,
        False,
        0.0,
        "drop_conflict",
    )


if __name__ == "__main__":
    test_reentry_hold_inside_window_without_force()
    test_reentry_bypass_on_force_signal()
    test_reentry_outside_window_or_has_inventory()
    test_buy_source_filter_position_source_mode()
    test_sell_health_signal_requires_live_inventory_only()
    test_sell_health_monitor_resets_on_new_run()
    test_idle_conflicting_actions_keep_only_net_buy()
    test_idle_conflicting_actions_drop_non_actionable_sell_bias()
    print("ALL REENTRY/SOURCE FILTER TESTS PASSED")
