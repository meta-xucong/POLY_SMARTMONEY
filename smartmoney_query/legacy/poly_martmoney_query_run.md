# Legacy Global Pull

Old style workflow:

```bash
cd /home/trader/polymarket_api/POLY_SMARTMONEY/smartmoney_query
python3 legacy/poly_martmoney_query_run.py --top 5000 --period ALL --order-by vol --days 365 --lifetime-mode all
```

This workflow starts from the global leaderboard rather than topic-specific market discovery.

It is archived because it under-samples topic specialists like NBA/NCAAB traders.
