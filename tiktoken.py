import tiktoken

if __name__ == '__main__':
    encoding = tiktoken.get_encoding("cl100k_base")
    print(len(encoding.encode("tiktoken is great!")))
