#!/usr/bin/env python3
"""
Usage:
  python3 check_key_address.py --key <hex_private_key> --address <0x_address>

Examples:
  python3 check_key_address.py --key 0xabc... --address 0x123...
  python3 check_key_address.py --key "$POLY_KEY" --address "$POLY_FUNDER"

This script uses eth-account to derive the address from the private key.
Install dependency if needed:
  pip install eth-account
"""
from __future__ import annotations

import argparse
import sys


def _normalize_key(raw: str) -> str:
    key = raw.strip()
    if key.startswith(("0x", "0X")):
        key = key[2:]
    return key


def _normalize_addr(raw: str) -> str:
    addr = raw.strip()
    return addr.lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check if private key matches address.")
    parser.add_argument("--key", required=True, help="hex private key (with or without 0x)")
    parser.add_argument("--address", required=True, help="0x EVM address")
    args = parser.parse_args()

    try:
        from eth_account import Account
    except Exception as exc:
        print("Missing dependency: eth-account. Install with: pip install eth-account", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    key = _normalize_key(args.key)
    addr = _normalize_addr(args.address)

    try:
        acct = Account.from_key(bytes.fromhex(key))
    except ValueError as exc:
        print(f"Invalid private key: {exc}", file=sys.stderr)
        return 3

    derived = _normalize_addr(acct.address)
    if derived == addr:
        print(f"✅ Match: {acct.address} == {args.address}")
        return 0

    print(f"❌ Mismatch: derived {acct.address} != provided {args.address}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
