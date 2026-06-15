"""
scheduler.py
────────────
Run this locally to keep the ChromaDB fresh.
Do NOT run this on Streamlit Cloud — run it on your local machine,
then commit the updated ./tmforum_db folder to GitHub.
"""

import time
from ingest import run_ingestion

print("🚀 Scheduler started — re-ingests every 24 hours")
print("   After each run, commit ./tmforum_db to GitHub to update the deployed app.\n")

while True:
    try:
        run_ingestion()
        print("✅ Ingestion completed — commit tmforum_db to GitHub now\n")
    except Exception as e:
        print(f"❌ Ingestion error: {e}\n")

    print("⏳ Next run in 24 hours...\n")
    time.sleep(86400)