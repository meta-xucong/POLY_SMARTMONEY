import sys

sys.path.insert(0, ".")

import ct_exec


def test_apply_actions_taker_fak_retry_then_fallback_to_maker(monkeypatch):
    call_count = {"taker": 0, "maker": 0}

    def fake_place_market_order(*_args, **_kwargs):
        call_count["taker"] += 1
        raise RuntimeError("no orders found to match with FAK")

    def fake_place_order(*_args, **_kwargs):
        call_count["maker"] += 1
        return {"order_id": "maker-1", "response": {"ok": True}}

    monkeypatch.setattr(ct_exec, "place_market_order", fake_place_market_order)
    monkeypatch.setattr(ct_exec, "place_order", fake_place_order)
    monkeypatch.setattr(ct_exec.time, "sleep", lambda *_args, **_kwargs: None)

    state = {}
    cfg = {
        "allow_partial": True,
        "taker_order_type": "FAK",
        "taker_fak_retry_max": 1,
        "taker_fak_retry_delay_sec": 0.0,
        "taker_fak_fallback_to_maker": True,
    }
    actions = [
        {
            "type": "place",
            "token_id": "tid-1",
            "side": "BUY",
            "price": 0.5,
            "size": 2.0,
            "_taker": True,
        }
    ]

    updated = ct_exec.apply_actions(
        client=object(),
        actions=actions,
        open_orders=[],
        now_ts=100,
        dry_run=False,
        cfg=cfg,
        state=state,
    )

    assert call_count["taker"] == 2
    assert call_count["maker"] == 1
    assert updated and updated[0].get("order_id") == "maker-1"
    assert state.get("taker_buy_orders") in (None, [])
