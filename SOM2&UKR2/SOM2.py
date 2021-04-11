from data import create_linear_convex
import numpy as np
import matplotlib.pyplot as plt
# V : 親SOMの潜在空間次元
# H : 親SOMの離さんかされた次元
# J : タスク

class SOM2(object):
    def __init__(self, X, latent_dim, epoch, K, σ_max, σ_min):
        self.X = X
        self.J, self.N, self.D = self.X.shape
        self.L = latent_dim
        self.K = K
        self.T = epoch
        self.σ_max = σ_max
        self.σ_min = σ_min
        self.H = 10 #親SOMの離散化された数
        self.V = 1 #親SOMの潜在空間の次元
        self.kid_Z = np.random.rand(self.J,self.N,self.L)
        self.kid_zeta = np.random.rand(self.J, self.K, self.L)
        self.parent_Z = np.random.rand(self.J, self.V)
        self.parent_zeta = np.random.rand(self.H, self.V)
        self.kid_y = np.zeros((self.J, self.K, self.D))

    def fit(self):
        np.random.seed(seed=0)
        J, N, D = self.X.shape
        for T in range(self.T):
            for j in range(J):
                self.kid_som_fit(j, T)
            self.parent_som_fit()
            self.kid_y = self.kid_y.reshape(self.J, self.K, self.D)

    def kid_som_fit(self,j, T):
        self.kid_y[j] = self.kid_NW(self.kid_zeta, self.kid_Z, X, T, j)
        Dist = np.sum((self.X[j, None, :, :] - self.kid_y[j, :, None, :])**2, axis=2)
        k_star = np.argmin(Dist, axis=0)
        self.kid_Z[j] = self.kid_zeta[j, k_star, :]

    def kid_NW(self, zeta, Z, X, T, j):
        kid_σ = ((self.σ_min - self.σ_max) / self.T) * T + self.σ_max
        Dist = np.sum((zeta[j, None, :, :] - Z[j, :, None, :])**2, axis=2)
        K = np.exp(-0.5/(kid_σ**2)*Dist)
        G = np.sum(K, axis=0)[:, None]
        H = K.T @ X[j]
        return H / G

    def parent_som_fit(self):
        self.kid_y = self.kid_y.reshape(self.J, self.K*self.D)  # J : タスク , 20 = 離散化 * 次元(子SOM参照ベクトル)
        for pT in range(self.T):
            self.parent_y = self.parent_NW(self.parent_zeta, self.parent_Z, self.kid_y, pT)
            Dist = np.sum((self.parent_Z[None, :, :] - self.kid_y[:, None, :]) ** 2, axis=2)
            k_star = np.argmin(Dist, axis=0)
            self.parent_Z = self.parent_zeta[k_star, :]

    def parent_NW(self, parent_zeta, parent_Z, kid_y, pT):
        parent_σ = ((self.σ_min - self.σ_max) / self.T) * pT + self.σ_max
        Dist = np.sum((parent_zeta[None, :, :] - parent_Z[:, None, :])**2, axis=2)
        K = np.exp(-0.5/(parent_σ**2)*Dist)
        G = np.sum(K, axis=0)[:, None]
        H = K.T @ kid_y
        return H / G

if __name__ == '__main__':
    X = create_linear_convex(100)
    # X.shape : (2, 100, 2) タスク, データ数, 観測空間の次元
    som2 = SOM2(X, latent_dim=1, epoch=100, K=100, σ_max=2.0, σ_min=0.03)
    som2.fit()

    print(som2.parent_y.shape)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(X[0, :, 0], X[0, :, 1])
    plt.scatter(X[1, :, 0], X[1, :, 1])
    plt.show()




