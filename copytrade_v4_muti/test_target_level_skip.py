import sys

sys.path.insert(0, ".")

from copytrade_run import (
    _calc_skip_count,
    _pick_target_level_skipped_accounts,
    _resolve_target_level_map,
    _resolve_target_level_skip_ratios,
)


assert _calc_skip_count(10, 0.0) == 0
assert _calc_skip_count(10, 0.4) == 4
assert _calc_skip_count(3, 0.4) == 1
assert _calc_skip_count(2, 0.4) == 1
assert _calc_skip_count(1, 0.8) == 1
print("[PASS] _calc_skip_count rounding")

cfg = {
    "default_target_level": "A",
    "target_addresses": [
        {"address": "0x1111111111111111111111111111111111111111", "level": "A"},
        {"address": "0x2222222222222222222222222222222222222222", "level": "b"},
        {"address": "0x3333333333333333333333333333333333333333", "level": "C"},
    ],
    "target_level_skip_ratios": {"A": 0, "B": 0.4, "C": 0.8},
}
targets = [item["address"] for item in cfg["target_addresses"]]
level_map = _resolve_target_level_map(cfg, targets)
ratios = _resolve_target_level_skip_ratios(cfg)
assert level_map["0x1111111111111111111111111111111111111111"] == "A"
assert level_map["0x2222222222222222222222222222222222222222"] == "B"
assert level_map["0x3333333333333333333333333333333333333333"] == "C"
assert ratios == {"A": 0.0, "B": 0.4, "C": 0.8}
print("[PASS] target level policy parse")

decision_cache = {}
all_accounts = ["a1", "a2", "a3", "a4", "a5"]
signal_id = "target|token|act:123"
first = _pick_target_level_skipped_accounts(signal_id, all_accounts, 0.4, 100, decision_cache)
second = _pick_target_level_skipped_accounts(signal_id, all_accounts, 0.4, 101, decision_cache)
assert len(first) == 2, first
assert sorted(first) == sorted(second), (first, second)
assert set(first).issubset(set(all_accounts))
print("[PASS] signal cache stable skip subset")

print("\nTARGET LEVEL SKIP TESTS PASSED")
