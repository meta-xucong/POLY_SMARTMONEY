import sys

sys.path.insert(0, ".")

from copytrade_run import (
    _should_accept_buy_action_source,
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


if __name__ == "__main__":
    test_reentry_hold_inside_window_without_force()
    test_reentry_bypass_on_force_signal()
    test_reentry_outside_window_or_has_inventory()
    test_buy_source_filter_position_source_mode()
    print("ALL REENTRY/SOURCE FILTER TESTS PASSED")
