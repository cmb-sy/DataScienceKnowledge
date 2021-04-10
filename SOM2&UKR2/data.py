import numpy as np
import matplotlib.pyplot as plt

def create_linear_convex(n_sample):
    """    サンプル数が同じ（固定長な）2次元の直線と凸関数のデータセットを返す．
       :param n_sample: 1クラスあたりのサンプル数．
       :returns: 形状が(class=2, n_sample, dim=2)のnumpy配列を返す．    """
    datasets = np.zeros((2, n_sample, 2))
    z_linear = np.random.rand(n_sample) * 2 - 1
    f_linear = len(z_linear) * [3]
    z_quadratic = np.random.rand(n_sample) * 2 - 1
    f_quadratic = z_quadratic ** 2
    datasets[0, :, 0] = z_linear
    datasets[0, :, 1] = f_linear
    datasets[1, :, 0] = z_quadratic
    datasets[1, :, 1] = f_quadratic
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

