import numpy as np
import matplotlib.pyplot as plt

def create_triangle3d(n_class, n_sample):
    """
    サンプル数が同じ（固定長な）3次元のtriangleデータセットを返す．
    :param n_class: triangleを何個生成するかを指定する
    :param n_sample: 1クラスあたりのサンプル数
    :return: 形状が(n_class, n_sample, dim=3)のnumpy配列を返す．
    """
    datasets = np.zeros((n_class, n_sample, 3))
    theta = np.linspace(-np.pi / 12, np.pi / 12, n_class)
    for n in range(n_class):
        min_X, max_X = 0, 4
        min_Y, max_Y = -1, 1
        X = np.random.uniform(min_X, max_X, n_sample)
        Y = np.zeros(n_sample)
        for s in range(n_sample):
            deltaY = (max_Y * X[s]) / max_X
            Y[s] = np.random.uniform(-deltaY, deltaY)
        rotate_X = X * np.cos(theta[n]) + Y * np.sin(theta[n])
        rotate_Y = X * np.sin(theta[n]) - Y * np.cos(theta[n])
        rotate_X -= np.mean(rotate_X)
        rotate_Y -= np.mean(rotate_Y)
        datasets[n][:, 0] = rotate_X
        datasets[n][:, 1] = rotate_Y
        datasets[n][:, 2] = n - n_class / 2
    return datasets

if __name__ == '__main__':
    X = create_linear_convex(100)
    # X.shape : (2, 100, 2) タスク, データ数, 観測空間の次元

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(X[0, :, 0], X[0, :, 1])
    plt.scatter(X[1, :, 0], X[1, :, 1])
    plt.show()

