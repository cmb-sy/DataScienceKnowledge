import numpy as np
import matplotlib.pyplot as plt

class gpr(object):
    def __init__(self, σ):
        self.σ = σ

    def fit(self, x_train, x_test, y):
        X_X_dist = np.sqrt((x_train[:, None] - x_train[None, :]) ** 2)
        K = np.exp((-0.5 * self.σ * X_X_dist))
        K_inv = np.linalg.inv(K)

        X_test_dist = np.sqrt((x_train[:, None] - x_test[None, :]) ** 2)
        k_star = np.exp((-0.5 * self.σ * X_test_dist))

        test_test_dist = np.sqrt((x_test - x_test) ** 2)
        k_star_star = np.exp((-0.5 * self.σ * test_test_dist))  # (NY)

        y_pred_mean = k_star.T @ K_inv @ y
        y_pred_cov = k_star_star - ((k_star.T @ K_inv) @ k_star)  # 分散共分散行列
        y_pred_std = np.sqrt(np.diag(y_pred_cov))  # 標準偏差
        return y_pred_mean, y_pred_cov, y_pred_std


if __name__ == "__main__":
    np.random.seed(0)
    resolution =  20
    x_train = np.sort(np.random.rand(resolution) * 15)
    y = np.sin(x_train) + np.random.rand(len(x_train))
    x_test = np.linspace(0, 20, resolution)

    model = gpr(σ=1.0)
    mu, var, std = model.fit(x_train, x_test, y)

    # 描画
    fig = plt.figure(figsize=(12, 4))
    plt.cla()
    plt.scatter(x_train, y, c="orange", marker="+", label="learn data")
    plt.plot(x_test, mu, color="red", linewidth=1, label="function mu")
    plt.scatter(x_test, mu, color="blue", s=10, label="test data")
    plt.fill_between(x_test, mu + std, mu - std, facecolor='green', alpha=0.2, label="std")
    plt.legend()
    plt.show()

