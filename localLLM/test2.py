# APIキーの設定
from openai import OpenAI
client = OpenAI(api_key="hoge")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "面白い話をして、どこが面白いのか解説してください。"}],
    stream=True,
)
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

