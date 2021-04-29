# GAUSSIAN_01

import numpy as np
import matplotlib.pyplot as plt

# ガウス関数を定義
def gauss(x, a, mu, sigma):
    return a * np.exp(-(x - mu)**2 / (2*sigma**2))

# Figureを作成
fig = plt.figure(figsize=(8, 6))

# FigureにAxesを追加
ax = fig.add_subplot(111)

# -4～8まで0.1刻みの数値の配列
x = np.arange(-10, 10, 0.1)

# グラフに描く関数
f = gauss(x, a=1, mu=0, sigma=3.0)

ax.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
# Axesにガウス関数を描画
ax.plot(x, f, color="blue")

plt.show()
