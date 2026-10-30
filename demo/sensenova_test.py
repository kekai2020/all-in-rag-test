from openai import OpenAI
from dotenv import load_dotenv

import os

load_dotenv()


client = OpenAI(
    base_url="https://token.sensenova.cn/v1",
    api_key=os.getenv("sensenova_API_KEY"),
)

# resp = client.chat.completions.create(
#     model="sensenova-6.8-flash-lite",
#     messages=[{"role": "user", "content": "Hello!"}],
# )

# 流式输出
# resp = client.chat.completions.create(
#     model="deepseek-flash",
#     messages=[{"role": "user", "content": "Hello! 1+1 等于多少"}],
#     stream=True,
# )
# for chunk in resp:
#     if chunk.choices:
#         delta = chunk.choices[0].delta
#         content = delta.content
#         if content:
#             print(content, end="", flush=True)
# print()

resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": "Hello! 1+1 等于多少"}],
)
print(resp.choices[0].message.content)