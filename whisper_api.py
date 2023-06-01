import sqlite3

import redis
from flask import Flask, request
from dotenv import load_dotenv
import os
import openai

load_dotenv()

app = Flask(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"]
db_connection = sqlite3.connect('board_data.db')

# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

# Connect to the database file
conn = sqlite3.connect('board_data.db')

# Create a cursor object to execute SQL queries
cursor = conn.cursor()

# Execute a SELECT query
cursor.execute('SELECT * FROM board')

@app.route("/")
def hello_world():  # put application's code here
    return "Hello World!"


@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    # Check if the request contains a file
    if "file" not in request.files:
        return "No file found in the request"

    audio_file = request.files["file"]
    audio_data = audio_file.read()

    # Make a request to the Whisper API
    response = openai.Transcriber.transcribe(audio_data)

    # Get the transcribed text from the response
    transcription = response["text"]

    # Return the transcribed text as the API response
    return transcription


@app.route('/data', methods=['POST'])
def insert_sqlite():
    data = request.get_json()  # JSON 데이터 가져오기
    if data:
        # SQLite에 데이터 삽입
        cursor = db_connection.cursor()
        cursor.execute("INSERT INTO board (board_type, url_string, title, content, author, create_at) VALUES (?, ?, ?, ?, ?, ?)",
                       (data['board_type'], data['url_string'], data['title'], data['content'], data['author'], data['create_at']))
        db_connection.commit()
        return 'Data inserted into SQLite.'
    else:
        return 'No data received.'
        # insert_redis
    s = '\n\n\n\n'.join(map(str, data[i][0:]))
    r.set(df_json[i][1], s)


if __name__ == '__main__':
    # print("OPENAI_API_KEY : ", end = ' ' )
    # print(os.environ["OPENAI_API_KEY"])
    app.run()
