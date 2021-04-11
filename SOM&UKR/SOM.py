import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SOM(object):
    def __init__(self, latent_dim, epoch, K, σ_max, σ_min):
        self.L = latent_dim
        self.K = K
        self.T = epoch
        self.σ_max = σ_max
        self.σ_min = σ_min

    def fit(self, X):
        np.random.seed(seed=0)
        N, D = X.shape
        self.history_Z =np.zeros((self.T, N, self.L))
        self.history_y = np.zeros((self.T, self.K, D))
        zeta = np.random.rand(self.K, self.L)
        Z = np.random.rand(N, self.L)

        for T in range(self.T):
            σ = ((self.σ_min - self.σ_max) / self.T)*T + self.σ_max
            y = self.NW(zeta, Z, X, σ)
            Dist = np.sum((X[None, :, :] - y[:, None, :])**2,axis=2)
            #次元Dをsumしている。それにより, N*K
            k_star = np.argmin(Dist, axis=0)
            # 最も小さい参照ベクトル番号を求める(X分), そのため、100次元ベクトルが出力
            Z = zeta[k_star, :]
            # 潜在空間の座標に100個のデータ分の勝者ノード番号をいれる [,:]としたのは後の計算のため,(100,1)になる。

            self.history_y[T] = y
            self.history_Z[T] = Z

    def NW(self, zeta, Z, X, σ):
        Dist = np.sum((zeta[None, :, :] - Z[:, None, :])**2, axis=2)
        K = np.exp(-0.5/(σ**2)*Dist)
        G = np.sum(K, axis=0)[:, None]
        H = K.T @ X
        return H / G

if __name__ == '__main__':
    N = 100
    D = 2
    X = np.zeros([N,D])
    X[:,0] = np.random.rand(N)
    X[:,1] = np.sin(X[:,0])
    som = SOM(latent_dim=1, epoch=100, K=100, σ_max=2.0, σ_min=0.03)
    som.fit(X)
    print(som.history_Z.shape)

    fig = plt.figure(figsize=(10, 5))
    ax_observable = fig.add_subplot(122, projection='3d')
    ax_latent = fig.add_subplot(121)
    #
    def update(i, z, f, x):
        plt.cla()
        ax_latent.cla()
        ax_observable.cla()

        fig.suptitle(f"epoch: {i}")
        ax_latent.scatter(z[i, :, 0], z[i, :, 0], s=50, edgecolors="k", c=x[:, 0])
        ax_observable.scatter(x[:, 0], x[:, 1], c=x[:, 0], s=50, marker='x')
        ax_observable.scatter(f[i, :, 0], f[i, :, 1], color='black')

        ax_observable.set_xlim(x[:, 0].min(), x[:, 0].max())
        ax_observable.set_ylim(x[:, 1].min(), x[:, 1].max())

        ax_observable.set_title('observable_space')
        ax_observable.set_xlabel("X_dim")
        ax_observable.set_ylabel("Y_dim")

        ax_latent.set_title('latent_space')
        ax_latent.set_xlabel("X_dim")

    ani = animation.FuncAnimation(fig, update, fargs=(som.history_Z, som.history_y, X), interval=100, frames=100)
    # ani.save("tmp.gif", writer = "pillow")
    plt.show()