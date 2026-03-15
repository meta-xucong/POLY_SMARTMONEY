# NBA/NCAAB Topic Workflow

This folder is the dedicated workflow entrypoint for NBA/NCAAB account discovery and screening.

Files:
- `screen_users_config_human_nba_ncaab.json`: stricter NBA/NCAAB screen
- `screen_users_config_human_nba_ncaab_relaxed.json`: wider NBA/NCAAB screen
- `run_topic_pipeline.sh`: discover topic users, deep-fetch them, then run screening
- `run_screen_only.sh`: rerun screening from already-fetched local data

Default output lives under `topics/nba_ncaab/data/`.

Typical usage:

```bash
cd /home/trader/polymarket_api/POLY_SMARTMONEY/smartmoney_query/topics/nba_ncaab
bash run_topic_pipeline.sh
```

For a wider pass:

```bash
TOPIC_SCREEN_CONFIG=screen_users_config_human_nba_ncaab_relaxed.json bash run_topic_pipeline.sh
```
