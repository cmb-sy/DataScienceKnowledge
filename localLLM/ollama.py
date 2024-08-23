import ollama
stream = ollama.chat(
    model="gemma2:2b",
        messages=[{"role": "user", "content": "趣味が筋トレの人の1週間の過ごし方を1文で教えて？"}],
    stream=True
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
