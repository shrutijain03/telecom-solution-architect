"""
scheduler.py — runs ingestion on a timer.
Interval is set in config.py → REFRESH_INTERVAL_HOURS
"""

import time
from ingest import run_ingestion
from config import REFRESH_INTERVAL_HOURS

SLEEP_SECS = REFRESH_INTERVAL_HOURS * 3600

print(f"🚀 Scheduler started — re-ingesting every {REFRESH_INTERVAL_HOURS}h")

while True:
    try:
        run_ingestion()
        print("✅ Ingestion completed successfully")
    except Exception as e:
        print(f"❌ Ingestion error: {e}")

    print(f"⏳ Next run in {REFRESH_INTERVAL_HOURS} hours...\n")
    time.sleep(SLEEP_SECS)