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

openai.api_key = os.environ["OPENAI_API_KEY"]
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




# for i in range(len(df_json)):
#     s = '\n\n'.join(map(str, df_json[i][1:]))
#     print("len : " , len(s))
#     # s를 임베딩한다
#     embedding = get_embedding(s)
#     #  'value' hash로 저장한다
#     # 리스트를 직렬화합니다.
#     # serialized_list = json.dumps(embedding)
#     embedding_byte = struct.pack('f' * len(embedding), *embedding)
#     print(type(embedding_byte)," : len : ", len(embedding_byte))
#     # if num_tokens_from_string(embedding_byte, "p50k_base") <= 8100:
#     if len(embedding_byte) <= 8100:
#         r.hset(i,'data',s)
#         r.hset(i,'embedding',embedding_byte)
#     else:
#         print("저장 실패 : embedding")
    # 'embeddings' hash로 저장한다


data = []
# redis deSerialized
for i in range(len(df_json)):
    bytes_of_values = r.hget(i,'embedding')
    embedding_vector = struct.unpack('f' * (len(bytes_of_values)//4), bytes_of_values)
    data.append(embedding_vector)
print("---------------------------------------------")

# array = np.array(data)

df = pd.DataFrame({'embedding' : data})

print(df)
cursor.close()
conn.close()