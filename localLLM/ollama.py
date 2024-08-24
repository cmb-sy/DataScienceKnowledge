from getData import create_dataframe
import polars as pl
# import ollama


# stream = ollama.chat(
#     model="gemma2:2b",
#         messages=[{"role": "user", "content": "趣味が筋トレの人の1週間の過ごし方を1文で教えて？"}],
#     stream=True
# )
# for chunk in stream:
#     print(chunk["message"]["content"], end="", flush=True)


if __name__ == "__main__":
    data = pl.read_csv('./data/test.csv')
    new_df = create_dataframe(data)
    for text in new_df[:,1]:
        print("以下のキーワードから、その人物の1週間の生活ストーリをA4一枚程度の文字数で作成してください。\n",
      text)
