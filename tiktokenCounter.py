import tiktoken

MAX_TOKEN_LEN = 4000

encoding = tiktoken.get_encoding("cl100k_base")
token_len = 55


def init():
    global token_len

    # Chat API 기본 탬플릿 문자열의 토큰 수
    # "2018년부터 2023년에 업로드된 전남대학교 게시글들에 대한 질문에 답변해라."
    # "\n\n게시글9: \n"
    token_len = 55


def get_token_len(string: str):
    return len(encoding.encode(string))


def is_appendable(new_string_len: int):
    if token_len + new_string_len <= MAX_TOKEN_LEN:
        return True

    return False


def append(new_string_len: int):
    global token_len

    # print("TOKEN APPEND  ", token_len, new_string_len, token_len+new_string_len)
    token_len += new_string_len


# if __name__ == '__main__':
#     print(len(encoding.encode("2018년부터 2023년에 업로드된 전남대학교 게시글들에 대한 질문에 답변해라.")))
#
#     print(len(encoding.encode("\n\n게시글9: \n")))