import os
import openai
from datetime import datetime
import pandas as pd

openai.api_key = os.environ["OPENAI_API_KEY"]
GPT_MODEL = "gpt-3.5-turbo"

if __name__ == '__main__':
    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'assistant', 'content': '예를들어, "작년 여름에 어떤 공지사항이 올라왔어?"라는 문장에서 시간 표현은 "작년 여름" 이다'},
            {'role': 'user', 'content': '다음 문장에 시간 표현 하나만 추출해줘. "작년 겨울에 어떤 공지사항이 올라왔어?" '},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    # print(response)
    time_expression = response['choices'][0]['message']['content'].split('"')[1]
    # print(time_expression)

    today_str = datetime.today().strftime('%Y년 %m월 %d일')
    # time_expression = "작년 겨울"

    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': '주어진 오늘 날짜를 기준으로 시간 변화량만큼 바뀐 날짜를 계산해라.'},
            {'role': 'assistant', 'content': '예를들어, 오늘이 2023년 6월 6월일 때, 어제의 datetime을 "YYYY-mm-dd"형식으로 바꾸면, "2023-06-05"다 '},
            {'role': 'assistant', 'content': '예를들어, 오늘이 2023년 6월 6월일 때, 작년 여름의 datetime을 "YYYY-mm-dd~YYYY-mm-dd"형식으로 바꾸면, "2022-06-01~2022-08-31"다 '},
            {'role': 'user', 'content': f'오늘은 {today_str}이야. {time_expression}의 datetime은 뭐니? 오직 "YYYY-mm-dd", "YYYY-mm-dd~YYYY-mm-dd"으로만 대답해줘.'},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    time_format = response['choices'][0]['message']['content'].split('"')[1]
    # print(time_format)
    period = time_format.split('~')
    time_query = ""
    if len(period) == 1:
        time_query = "업로드날짜: " + time_format.replace('-', '.')
    else:
        dates = pd.date_range(period[0], period[1], freq='D')
        for date in dates:
            time_query += "업로드날짜: " + date.strftime('%Y-%m-%d') + "\n"
    print(time_query)


