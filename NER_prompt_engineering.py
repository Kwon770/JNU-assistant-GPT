import os
import openai
from datetime import datetime
import pandas as pd
from time import time
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]
GPT_MODEL = "gpt-3.5-turbo"

# if __name__ == '__main__':
def ner_prompt(s):
    start = time()
    print("-----> ner_prompt() 시작")

    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'assistant', 'content': '예를들어, "작년 카카오테크캠퍼스에 대해서 알려줘"라는 문장에서 시간 표현은 "작년" 이다'},
            {'role': 'assistant', 'content': '예를들어, "작년 여름에 어떤 공지사항이 올라왔어?"라는 문장에서 시간 표현은 "작년 여름" 이다'},
            {'role': 'user', 'content': f'다음 문장에 시간 표현 하나만 추출해줘. "{s}" '},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    # print("ner_prompt : ", response)
    end = time()
    print(f"-> 시간 표현 추출 GPT : {end - start} ms")
    stamp = end

    time_expression = response['choices'][0]['message']['content'].split('"')
    # print(s, time_expression)
    if len(time_expression) <= 1:
        print("-> 시간 표현이 존재하지 않음.")
        return ""

    time_expression = time_expression[1]
    today_str = datetime.today().strftime('%Y년 %m월 %d일')

    response = openai.ChatCompletion.create(
        messages=[
            {'role': 'system', 'content': '주어진 오늘 날짜를 기준으로 시간 변화량만큼 바뀐 날짜를 계산해라.'},
            {'role': 'assistant', 'content': '예를들어, 오늘이 2023년 6월 6월일 때, 어제의 datetime은 "2023-06-05"다 '},
            {'role': 'assistant', 'content': '예를들어, 오늘이 2023년 6월 6월일 때, 작년의 datetime은 "2022-01-01~2022-12-31"다 '},
            {'role': 'assistant', 'content': '예를들어, 오늘이 2023년 6월 6월일 때, 작년 여름의 datetime은 "2022-06-01~2022-08-31"다 '},
            {'role': 'user', 'content': f'오늘은 {today_str}이야. {time_expression}의 datetime은 뭐니? 오직 "YYYY-mm-dd", "YYYY-mm-dd~YYYY-mm-dd"으로만 대답해줘.'},
        ],
        model=GPT_MODEL,
        temperature=0,
    )

    end = time()
    print(f"-> 시간 계산 GPT : {end - stamp} ms")
    stamp = end

    time_format = response['choices'][0]['message']['content'].split('"')
    if len(time_format) == 1:
        return ''
    time_format = time_format[1]
    # print("time_format : ", time_format)
    period = time_format.split('~')

    time_query = ""
    if len(period) == 1: # time_format 이 기간이 아니라면
        time_query = "업로드날짜: " + time_format.replace('-', '.')
        # print(f"date {time_query}")

    else: # time_format 이 기간이라면
        period_start = period[0].split('-')
        period_end = period[1].split('-')

        # 연단위 기간에서 쿼리 생성
        if period_start[0] < period_end[0]:
            period_start = pd.to_datetime(period[0])
            period_end = pd.to_datetime(period[1]) + pd.DateOffset(years=1)
            dates = pd.date_range(period_start, period_end, freq='Y')

            for date in dates:
                time_query += "업로드날짜: " + date.strftime('%Y.') + "\n"
            # print(f"years {time_query}")

        # 달단위 기간에서 쿼리 생성
        elif period_start[1] < period_end[1]:
            period_start = pd.to_datetime(period[0])
            period_end = pd.to_datetime(period[1]) + pd.DateOffset(months=1)
            dates = pd.date_range(period_start, period_end, freq='M')

            for date in dates:
                time_query += "업로드날짜: " + date.strftime('%Y.%m') + "\n"
            # print(f"months {time_query}")

        else:
            dates = pd.date_range(period[0], period[1], freq='M')

            for date in dates:
                time_query += "업로드날짜: " + date.strftime('%Y.%m.%d') + "\n"
            # print(f"days {time_query}")


    end = time()
    print(f"-----> ner_prompt() 종료 : {end - start} ms")

    return time_query


# if __name__ == '__main__':
#     print(pd.to_datetime("2022-12-01"))
#     print(pd.to_datetime("2022-12-01") + pd.DateOffset(months=1))
#     print(pd.date_range("2022-12-01", "2023-12-15", freq='Y'));