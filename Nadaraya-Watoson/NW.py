import numpy as np
import matplotlib.pyplot as plt

class NW():
    def __init__(self, sigma):
        self.sigma = sigma
    def fit(self, x, y, test_x):
        delta = test_x[:, None] - x[None, :]  #新規データ - 元データ
        Dist = np.square(delta)
        kernel = np.exp((-0.5 / (self.sigma ** 2)) * Dist) #ガウシアンカーネル
        k_denominator = np.sum(kernel, axis=1)
        r = kernel / k_denominator[:, None]
        pred_y = r @ y
        return pred_y

if __name__ == '__main__':
    np.random.seed(0)
    train_x = np.linspace(-5, 5, 10)
    train_y = np.sin(train_x) + np.random.randn(*train_x.shape) / 8
    test_x = np.linspace(-6, 6, 100) #新規のデータ
    nw = NW(σ:=0.1)
    pred_y = nw.fit(train_x, train_y, test_x)
    fig = plt.figure()
    plt.scatter(train_x, train_y, c='b', lw=2, label="train")
    plt.plot(test_x, pred_y, c='r', lw=2, label="pred")
    fig.suptitle(f"sigma={σ}")
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig("img.png")
    plt.show()