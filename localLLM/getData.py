import polars as pl
import numpy as np

def create_dataframe(data):
    # 名前以外のnull値を含む行を削除
    not_null_data = data.drop_nulls(subset=data[:, 1:].columns)
    
    one_dimensional_vector = []
    for row in not_null_data.iter_rows():
        # Noneでないデータをクリーンアップ
        cleaned_data = [item for item in row if item is not None]
        # データをカンマで連結
        concatenated_row = ','.join(map(str, cleaned_data))
        one_dimensional_vector.append(concatenated_row)
    
    # 1次元のベクトルをnumpy配列に変換し、1列の2次元配列に変形
    one_dimensional_array = np.array(one_dimensional_vector).reshape(-1, 1)
    
    # 新しいDataFrameを作成
    new_df = pl.DataFrame({
        "名前": not_null_data[:, 0],  # 元のデータの最初の列を名前として使用
        'キーワード': one_dimensional_array  # 連結されたデータをキーワードとして使用
    })
    return new_df

if __name__ == "__main__":
    data = pl.read_csv('./data/test.csv')
    new_df = create_dataframe(data)
    print(new_df)
