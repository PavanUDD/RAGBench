import sqlite3, json
from pathlib import Path

p = Path("runs/ragbench.db")
conn = sqlite3.connect(p)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT run_id, created_at, config_json FROM runs ORDER BY created_at DESC LIMIT 20"
).fetchall()

print("Recent TFIDF runs (signature):")
for r in rows:
    cfg = json.loads(r["config_json"]) if r["config_json"] else {}
    if cfg.get("retriever") == "TFIDF":
        print(r["created_at"], r["run_id"], "sig=", cfg.get("benchmark_signature"))

conn.close()
