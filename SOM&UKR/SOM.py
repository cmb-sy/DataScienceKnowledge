import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SOM(object):
    def __init__(self, latent_dim, epoch, K, σ_max, σ_min, tau):
        self.L = latent_dim
        self.K = K
        self.T = epoch
        self.σ_max = σ_max
        self.σ_min = σ_min
        self.tau = tau
        self.resolution = 10

    def fit(self, X):
        np.random.seed(seed=0)
        N, D = X.shape
        self.history_Z =np.zeros((self.T, N, self.L))
        self.history_y = np.zeros((self.T, self.K, D))

        zetax = np.linspace(-1, 1, self.resolution)
        zetay = np.linspace(-1, 1,self.resolution)
        xx, yy = np.meshgrid(zetax, zetay)
    # meshgrid関数･･･zetaxとzetayの組み合わせを作る関数(xx:zetax:resolution*resolution,yy=resolution*resolution)
        xx = np.ravel(xx)  # ravel関数･･･2次元配列を1次元配列に変換する
        yy = np.ravel(yy)
        self.zeta = np.c_[xx, yy]  # c_･･･xxとyyを統合
        Z = np.random.rand(N, self.L)

        self.σ_list = np.zeros((self.T))

        for T in range(self.T):
            σ = ((self.σ_min - self.σ_max) / self.T) * T + self.σ_max
            # σ = self.σ_min + (self.σ_max - self.σ_min) * np.exp(-T / self.tau)
            self.σ_list[T] = σ 

            y = self.NW(self.zeta, Z, X, σ)
            Dist = np.sum((X[None, :, :] - y[:, None, :])**2,axis=2)
            #次元Dをsumしている。それにより, N*K
            k_star = np.argmin(Dist, axis=0)
            # 最も小さい参照ベクトル番号を求める(X分), そのため、100次元ベクトルが出力
            Z = self.zeta[k_star, :]
            # 潜在空間の座標に100個のデータ分の勝者ノード番号をいれる [,:]としたのは後の計算のため,(100,1)になる。

            self.history_y[T] = y
            self.history_Z[T] = Z
        self.f = self.history_y.reshape(self.T, self.resolution, self.resolution, D)

    def NW(self, zeta, Z, X, σ):
        Dist = np.sum((zeta[None, :, :] - Z[:, None, :])**2, axis=2)
        K = np.exp(-0.5/(σ**2)*Dist)
        G = np.sum(K, axis=0)[:, None]
        H = K.T @ X
        return H / G

if __name__ == '__main__':
    from data import gen_saddle_shape
    X = gen_saddle_shape(100,random_seed=10, noise_scale=0.0)
    epoch = 150
    som = SOM(latent_dim=2, epoch=epoch, K=100, σ_max=3.0, σ_min=0.02, tau=60)
    som.fit(X)
    print("Z : ", som.history_Z.shape)

    fig = plt.figure(figsize=(10, 5))
    ax_observable = fig.add_subplot(122, projection='3d')
    ax_latent = fig.add_subplot(121)
    def update(i, zeta, z, f, x):
        plt.cla()
        ax_latent.cla()
        ax_observable.cla()

        fig.suptitle(f"epoch: {i}")
        ax_latent.scatter(zeta[:, 0], zeta[:, 1], s=30, alpha=0.5)
        ax_latent.scatter(z[i, :, 0], z[i, :, 1], s=50, edgecolors="k", c=x[:, 0])
        ax_observable.scatter(x[:, 0], x[:, 1],x[:, 2], c=x[:, 0], s=50, marker='x')
        ax_observable.plot_wireframe(f[i, :, :, 0], f[i, :, :, 1],f[i, :, :, 2], color='black')

        ax_observable.set_xlim(x[:, 0].min(), x[:, 0].max())
        ax_observable.set_ylim(x[:, 1].min(), x[:, 1].max())

        ax_observable.set_title('observable_space')
        ax_latent.set_title('latent_space')

    ani = animation.FuncAnimation(fig, update, fargs=(som.zeta, som.history_Z, som.f, X), interval=100, frames=epoch)
    ani.save("tmp.gif", writer = "pillow")
    # σ_t = np.linspace(0, epoch, epoch)
    # plt.plot(σ_t, som.σ_list)
    plt.show()
