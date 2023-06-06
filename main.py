from flask import Flask, request
# from flask_cors import CORS, cross_origin

app = Flask(__name__)
# CORS(app, resources={r'*': {'origins': ['http://localhost:5173']}})
import struct
import redis
import ast
import openai
import pandas as pd
from scipy import spatial
import os

from dotenv import load_dotenv

from NER_prompt_engineering import ner_prompt

load_dotenv()

# openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = os.getenv('OPENAI_API_KEY')
# model
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"


def get_redis_by_board_type(board_type):
    # Redis 클라이언트 생성
    r = redis.Redis(host='localhost', port=6379)

    # Redis에서 모든 해시 조회
    all_keys = r.keys("*")

    # board_type 값이 일치하는 해시 필터링
    filtered_keys = [key for key in all_keys if r.hget(key, "board_type").decode("utf-8") == board_type]

    # 데이터프레임 생성
    data = []
    for key in filtered_keys:
        hash_data = r.hgetall(key)
        hash_data = {field.decode("utf-8", errors="ignore"): value.decode("utf-8", errors="ignore") for field, value in
                     hash_data.items()}
        data.append(hash_data)
    return pd.DataFrame(data)


def retrieve_posts_df(
        board_type: str
):
    # load a df
    # (( example csv ))

    embeddings_path = "https://cdn.openai.com/API/examples/data/winter_olympics_2022.csv"
    df = pd.read_csv(embeddings_path)
    # (( TODO : Load the df from Redis ))

    # redis deSerialized
    data = []
    r = redis.Redis(host='localhost', port=6379, db=0)
    for i in range(1489):
        bytes_of_values = r.hget(i, 'embedding')
        bytes_of_file = r.hget(i,'data')

        embedding_vector = struct.unpack('f' * (len(bytes_of_values) // 4), bytes_of_values)

        data.append(embedding_vector)

    # array = np.array(data)

    df = pd.DataFrame({'embedding': data,
                       'text': bytes_of_file})

    # convert embeddings from CSV str type back to list type
    # the dataframe has two columns: "text" and "embedding"
    # df['embedding'] = df['embedding'].apply(ast.literal_eval)

    return df


# search function
# Returns a list of posts and relatednesses, sorted from most related to least
def search_posts_ranked_by_relatedness(
        question: str,
        board_type: str,
        top_n: int,
        relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
) -> tuple[list[str], list[float]]:
    df = retrieve_posts_df(board_type)

    question_embedding_response = openai.Embedding.create(
        model=EMBEDDING_MODEL,
        input=question,
    )
    question_embedding = question_embedding_response["data"][0]["embedding"]

    posts_and_relatednesses = [
        (row["text"], relatedness_fn(question_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    posts_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    posts, relatednesses = zip(*posts_and_relatednesses)

    return posts[:top_n], relatednesses[:top_n]


def ask_based_on_posts(
        question: str,
        related_posts: list[str]
):
    nl = "\n"
    nnl = "\n\n"
    query = f"""아래는 2018년부터 2023년에 업로드된 전남대학교 게시글들을 질문의 답변에 사용해라. 만약 답변을 찾을 수 없다면, "업로드된 공지글이 없습니다"라고 써라.
    
    {nnl.join([f"게시글{index}: {nl}{post} " for index, post in enumerate(related_posts)])}    
    
    질문: {question}"""

    time_query = ner_prompt(question)
    print(query + "\n 업로드날짜: " + time_query)
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': '2018년부터 2023년에 업로드된 전남대학교 게시글들에 대한 질문에 답변해라.'},
            {'role': 'user', 'content': query + "\n 업로드날짜: " + time_query},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    # debug
    print(response)

    return response['choices'][0]['message']['content']


@app.route('/ask', methods=['GET'])
def search_and_ask():
    question = request.args.get('question', type = str)
    board_type = request.args.get('board_type', type=str)
    top_n = 10

    posts, relatednesses = search_posts_ranked_by_relatedness(
        question=question,
        board_type=board_type,
        top_n=top_n
    )
    return ask_based_on_posts(
        question=question,
        related_posts=posts
    )

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    # Check if the request contains a file
    if "file" not in request.files:
        return "No file found in the request"

    audio_file = request.files["file"]
    audio_data = audio_file.read()

    # Make a request to the Whisper API
    response = openai.Audio.transcribe("whisper-1", audio_data)

    # Get the transcribed text from the response
    transcription = response["text"]

    # Return the transcribed text as the API response
    return transcription

if __name__ == '__main__':
    from waitress import serve
    print('server open: 8080')
    serve(app, host="0.0.0.0", port=8080)

#     # examples
#     posts, relatednesses = posts_ranked_by_relatedness("curling gold medal", top_n=5)
#     for string, relatedness in zip(posts, relatednesses):
#         print(f"{relatedness=:.3f}")
