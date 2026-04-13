# Politics Event Workflow

This folder is the dedicated workflow entrypoint for politics/event-specialist discovery and screening.

Files:
- `screen_users_config_politics_event_specialist.json`
- `screen_users_config_politics_event_specialist_refined.json`
- `screen_users_config_politics_event_specialist_strict_human.json`
- `run_topic_pipeline.sh`
- `run_screen_only.sh`

Default output lives under `topics/politics_event/data/`.

Typical usage:

```bash
cd /home/trader/polymarket_api/POLY_SMARTMONEY/smartmoney_query/topics/politics_event
bash run_topic_pipeline.sh
```

For a stricter human-biased pass:

```bash
TOPIC_SCREEN_CONFIG=topics/politics_event/screen_users_config_politics_event_specialist_strict_human.json bash run_topic_pipeline.sh
```
