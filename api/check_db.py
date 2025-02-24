import sqlite3

conn = sqlite3.connect("boulder-ai.db")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master;")
print(cursor.fetchall())
conn.close()
