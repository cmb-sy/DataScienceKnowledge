import polars as pl

# CSVファイルからデータを取得
data = pl.read_csv('./data/test.csv')

# 行ごとにデータを取得
for row in data[0].iter_rows():
    print(row)