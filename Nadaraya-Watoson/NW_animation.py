#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("../../")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from NW import NW

np.random.seed(0)
train_x = np.linspace(-np.pi, np.pi, 10)
train_y = np.sin(train_x) + np.random.randn(*train_x.shape) / 8
test_x = np.linspace(-5, 5, 100)  # 新規のデータ
resolution = 100
sigmas = np.linspace(0.001, 5, resolution)[::-1]
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)

def update(i, fig, ax):
    fig.suptitle(f"sigma={sigmas[i]}")
    ax.cla()
    model = NW(sigma=sigmas[i])
    pred_y = model.fit(train_x, train_y, test_x)
    plt.scatter(train_x, train_y, c='b')
    plt.plot(test_x, pred_y, c='r')

if __name__ == '__main__':

    ani = FuncAnimation(fig,
                        update,
                        frames=resolution,
                        repeat=True,
                        interval=100,
                        fargs=(fig, ax))
    # plt.show()
    ani.save(f"sigma_animation.mp4", writer='ffmpeg')