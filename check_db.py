import sqlite3
conn = sqlite3.connect('ragbench.db')
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT json_extract(r.config_json, '$.retriever') as retriever,
           m.metric_name, round(m.metric_value, 4) as val
    FROM runs r
    JOIN metrics m ON r.run_id = m.run_id
    ORDER BY r.created_at DESC LIMIT 30
""").fetchall()
for r in rows:
    print(f"{r['retriever']:10} | {r['metric_name']:15} | {r['val']}")
conn.close()
