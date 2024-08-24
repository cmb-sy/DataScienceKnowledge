import polars as pl
import numpy as np

data = pl.read_csv('./data/test.csv')

def getData(data):
    one_dimensional_vector = []
    # 行ごとにデータを取得
    for row in data.iter_rows():
        cleaned_data = [item for item in row if item is not None]
        concatenated_row = ','.join(map(str, cleaned_data))
        one_dimensional_vector.append(concatenated_row)
    # NumPy配列に変換し、(193, 1)の形状にリシェイプ
    one_dimensional_array = np.array(one_dimensional_vector).reshape(-1, 1)
    return one_dimensional_array

if __name__ == "__main__":
    # 名前以外の全ての値がnullの列を削除
    notNullData = data.drop_nulls(subset=data[:,1:].columns)
    result = getData(notNullData[:, 1:])
    new_df = pl.DataFrame({
        "名前": notNullData[:, 0],
        'キーワード': result
    })
    
    print(new_df)
