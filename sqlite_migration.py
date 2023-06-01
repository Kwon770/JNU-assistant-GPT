import sqlite3
import redis
import pandas as pd

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Connect to the database file
conn = sqlite3.connect('board_data.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a SELECT query
cursor.execute('SELECT * FROM board')

# Fetch all rows returned by the query
df_json = cursor.fetchall()
# Iterate over the rows and print the data

for i in range(len(df_json)):
    s = '\n\n\n\n'.join(map(str, df_json[i][1:]))
    r.set(df_json[i][1], s)

# # Retrieve the serialized DataFrame from Redis
# df_json = r.get('your_key')
#
# # Convert the serialized string back to a DataFrame
# df = pd.read_json(df_json)
#
# # Close the cursor and the connection
cursor.close()
conn.close()