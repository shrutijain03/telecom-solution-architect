import time
from ingest import run_ingestion

print("🚀 Scheduler started...")

while True:
    try:
        run_ingestion()
        print("✅ Ingestion completed successfully")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")

    print("⏳ Waiting 24 hours for next update...\n")
 
    time.sleep(86400)  # 24 hours


