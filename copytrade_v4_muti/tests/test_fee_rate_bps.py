import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, ".")

import ct_exec


@pytest.fixture(autouse=True)
def _clear_fee_rate_cache():
    ct_exec._FEE_RATE_CACHE.clear()
    yield
    ct_exec._FEE_RATE_CACHE.clear()


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"status={self.status_code}")


def test_resolve_order_fee_rate_bps_skips_http_for_fee_free_market(monkeypatch):
    client = SimpleNamespace(host="https://clob.polymarket.com")
    state = {"market_status_cache": {"tid-0": {"meta": {"feesEnabled": False}}}}

    def fail_http(*_args, **_kwargs):
        raise AssertionError("fee-rate endpoint should not be called for fee-free markets")

    monkeypatch.setattr(ct_exec.requests, "get", fail_http)

    assert ct_exec._resolve_order_fee_rate_bps(client, "tid-0", state=state) == 0


def test_resolve_order_fee_rate_bps_fetches_and_caches(monkeypatch):
    client = SimpleNamespace(host="https://clob.polymarket.com")
    state = {"market_status_cache": {"tid-1": {"meta": {"feesEnabled": True}}}}
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        assert url == "https://clob.polymarket.com/fee-rate"
        assert params == {"token_id": "tid-1"}
        return _FakeResponse({"base_fee": 1000})

    monkeypatch.setattr(ct_exec.requests, "get", fake_get)

    assert ct_exec._resolve_order_fee_rate_bps(client, "tid-1", state=state) == 1000
    assert ct_exec._resolve_order_fee_rate_bps(client, "tid-1", state=state) == 1000
    assert calls["n"] == 1


def test_resolve_order_fee_rate_bps_raises_when_enabled_market_cannot_resolve(monkeypatch):
    client = SimpleNamespace(host="https://clob.polymarket.com")
    state = {"market_status_cache": {"tid-2": {"meta": {"feesEnabled": True}}}}

    def fail_http(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ct_exec.requests, "get", fail_http)

    with pytest.raises(RuntimeError, match="failed to resolve fee rate"):
        ct_exec._resolve_order_fee_rate_bps(client, "tid-2", state=state)


def test_place_order_passes_fee_rate_bps(monkeypatch):
    captured = {}

    class FakeOrderArgs:
        def __init__(self, token_id, price, size, side, fee_rate_bps=0):
            captured["kwargs"] = {
                "token_id": token_id,
                "price": price,
                "size": size,
                "side": side,
                "fee_rate_bps": fee_rate_bps,
            }

    def fake_post_order(*_args, **_kwargs):
        return {"orderID": "maker-1"}

    import py_clob_client.clob_types as clob_types

    monkeypatch.setattr(clob_types, "OrderArgs", FakeOrderArgs)
    monkeypatch.setattr(ct_exec, "_post_order_with_retry", fake_post_order)

    client = SimpleNamespace(create_order=lambda order_args: order_args)
    result = ct_exec.place_order(
        client,
        token_id="tid-maker",
        side="BUY",
        price=0.51,
        size=2.0,
        fee_rate_bps=1000,
    )

    assert captured["kwargs"]["fee_rate_bps"] == 1000
    assert result["order_id"] == "maker-1"


def test_place_market_order_passes_fee_rate_bps(monkeypatch):
    captured = {}

    class FakeMarketOrderArgs:
        def __init__(self, token_id, amount, side, price=0, fee_rate_bps=0):
            captured["kwargs"] = {
                "token_id": token_id,
                "amount": amount,
                "side": side,
                "price": price,
                "fee_rate_bps": fee_rate_bps,
            }

    def fake_post_order(*_args, **_kwargs):
        return {"orderID": "taker-1"}

    import py_clob_client.clob_types as clob_types

    monkeypatch.setattr(clob_types, "MarketOrderArgs", FakeMarketOrderArgs)
    monkeypatch.setattr(ct_exec, "_post_order_with_retry", fake_post_order)

    client = SimpleNamespace(create_market_order=lambda order_args: order_args)
    result = ct_exec.place_market_order(
        client,
        token_id="tid-taker",
        side="BUY",
        amount=1.5,
        price=0.5,
        fee_rate_bps=1000,
        order_type="FAK",
    )

    assert captured["kwargs"]["fee_rate_bps"] == 1000
    assert result["order_id"] == "taker-1"


def test_apply_actions_passes_fee_rate_bps_to_limit_orders(monkeypatch):
    captured = {}

    def fake_resolve_fee(*_args, **_kwargs):
        return 1000

    def fake_place_order(*_args, **kwargs):
        captured["fee_rate_bps"] = kwargs.get("fee_rate_bps")
        return {"order_id": "maker-1", "response": {"ok": True}}

    monkeypatch.setattr(ct_exec, "_resolve_order_fee_rate_bps", fake_resolve_fee)
    monkeypatch.setattr(ct_exec, "place_order", fake_place_order)

    updated = ct_exec.apply_actions(
        client=object(),
        actions=[
            {
                "type": "place",
                "token_id": "tid-maker",
                "side": "BUY",
                "price": 0.5,
                "size": 2.0,
            }
        ],
        open_orders=[],
        now_ts=100,
        dry_run=False,
        cfg={"allow_partial": True},
        state={},
    )

    assert captured["fee_rate_bps"] == 1000
    assert updated and updated[0]["order_id"] == "maker-1"


def test_apply_actions_passes_fee_rate_bps_to_taker_orders(monkeypatch):
    captured = {}

    def fake_resolve_fee(*_args, **_kwargs):
        return 1000

    def fake_place_market_order(*_args, **kwargs):
        captured["fee_rate_bps"] = kwargs.get("fee_rate_bps")
        return {"order_id": "taker-1", "response": {"ok": True}}

    monkeypatch.setattr(ct_exec, "_resolve_order_fee_rate_bps", fake_resolve_fee)
    monkeypatch.setattr(ct_exec, "place_market_order", fake_place_market_order)

    ct_exec.apply_actions(
        client=object(),
        actions=[
            {
                "type": "place",
                "token_id": "tid-taker",
                "side": "BUY",
                "price": 0.5,
                "size": 2.0,
                "_taker": True,
            }
        ],
        open_orders=[],
        now_ts=100,
        dry_run=False,
        cfg={"allow_partial": True, "taker_order_type": "FAK"},
        state={},
    )

    assert captured["fee_rate_bps"] == 1000


def test_apply_actions_dry_run_does_not_resolve_fee_rate(monkeypatch):
    def fail_resolve_fee(*_args, **_kwargs):
        raise AssertionError("dry-run should not resolve fee rates")

    monkeypatch.setattr(ct_exec, "_resolve_order_fee_rate_bps", fail_resolve_fee)

    updated = ct_exec.apply_actions(
        client=object(),
        actions=[
            {
                "type": "place",
                "token_id": "tid-dry",
                "side": "BUY",
                "price": 0.5,
                "size": 2.0,
            }
        ],
        open_orders=[],
        now_ts=100,
        dry_run=True,
        cfg={"allow_partial": True},
        state={},
    )

    assert updated and updated[0]["order_id"] == "dry_run"
