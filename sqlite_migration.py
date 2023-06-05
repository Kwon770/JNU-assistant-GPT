import sqlite3
import redis
import struct
import numpy as np
import pandas as pd
import os

# for Embeddings
import ast  # for converting embeddings saved as strings back to arrays
import openai  # for calling the OpenAI API
import json
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search

# openai.api_key = os.environ["OPENAI_API_KEY"]
from dotenv import load_dotenv
load_dotenv()

openai.api_key = "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"
# print(os.environ["OPENAI_API_KEY"])
# openai.api_key = os.getenv('OPENAI_API_KEY')

# model
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

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

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_embedding(text, model="text-embedding-ada-002"):
   # text = text.replace("\n", " ")
   a = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   return a

# df['ada_embedding'] = df.combined.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))
# df.to_csv('output/embedded_1k_reviews.csv', index=False)


# for i in range(1):
    # s = '\n\n'.join(map(str, df_json[i][1:]))
for i in range(len(df_json)):
    board_type = df_json[i][0]
    url_string = df_json[i][1]
    title = df_json[i][2]
    content = df_json[i][3]
    author = df_json[i][4]
    date = df_json[i][5]

    s = "제목: " + title + "\n내용: " + content + "\n게시판: " + board_type + "\n업로드날짜: " + date
    print(s)
    # s를 임베딩한다
    embedding = get_embedding(s)
    #  'value' hash로 저장한다
    # 리스트를 직렬화합니다.
    # serialized_list = json.dumps(embedding)
    embedding_byte = struct.pack('f' * len(embedding), *embedding)
    print(type(embedding_byte)," : len : ", len(embedding_byte))
    # if num_tokens_from_string(embedding_byte, "p50k_base") <= 8100:
    if len(embedding_byte) <= 8100:
        print(df_json[i][0])
        r.hset(i,'board_type',board_type)
        r.hset(i,'text',s)
        r.hset(i,'embedding',embedding_byte)
    else:
        print("저장 실패 : embedding")


data = []
board_types = []
text_data = []
# redis deSerialized
for i in range(len(df_json)):
    board_type = r.hget(i,'board_type')
    bytes_of_values = r.hget(i,'embedding')
    bytes_of_file = r.hget(i, 'text')

    embedding_vector = struct.unpack('f' * (len(bytes_of_values)//4), bytes_of_values)
    board_types.append(board_type)
    text_data.append(bytes_of_file)
    data.append(embedding_vector)
print("---------------------------------------------")

df = pd.DataFrame({'board_type': board_type,
                    'embedding': data,
                   'text': bytes_of_file})

# df = pd.DataFrame({'embedding' : data})

print(df)
cursor.close()
conn.close()