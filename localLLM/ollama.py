import ollama

stream = ollama.chat(
    model="gemma2:2b",
        messages=[{"role": "user", "content": "趣味が筋トレの人の1週間の過ごし方を1文で教えて？"}],
    stream=True
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)


"以下のキーワードから構成されるパーソナリティを考えてください。次にその人物の1週間の生活ストーリをA4一枚程度の文字数で作成してください。"
