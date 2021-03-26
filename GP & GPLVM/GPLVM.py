import numpy as np
import sys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from tqdm import tqdm


class GPLVM(object):
    def __init__(self, θ1, θ2, θ3):
        self.θ1 = θ1
        self.θ2 = θ2
        self.θ3 = θ3

    def fit(self, X, latent_dim, epoch, eta):
        resolution = 10
        T = epoch
        N, D = X.shape
        L = latent_dim
        Z = np.random.randn(N, L) / 100

        history = {}
        history['Z'] = np.zeros((T, N, L))
        history['f'] = np.zeros((T, resolution, resolution, D))

        for t in tqdm(range(T)):
            K = self.θ1 * self.kernel(Z, Z, self.θ2) + self.θ3 * np.eye(N)
            inv_K = np.linalg.inv(K)
            dLdK = 0.5 * (inv_K @ (X @ X.T) @ inv_K - D * inv_K)
            dKdX = -2.0 * (((Z[:, None, :] - Z[None, :, :]) * K[:, :, None])) / self.θ2
            dLdX = np.sum(dLdK[:, :, None] * dKdX, axis=1)

            Z = Z + eta * dLdX
            history['Z'][t] = Z

            z_new_x = np.linspace(min(Z[:, 0]), max(Z[:, 0]), resolution)
            z_new_y = np.linspace(min(Z[:, 1]), max(Z[:, 1]), resolution)
            z_new = np.dstack(np.meshgrid(z_new_x, z_new_y)).reshape(resolution ** 2, L)
            k_star = self.θ1 * self.kernel(z_new, Z, self.θ2)
            F = (k_star @ inv_K @ X).reshape(resolution, resolution, D)
            history['f'][t] = F
        return history

    def kernel(self, X1, X2, θ2):
        Dist = np.sum(((X1[:, None, :] - X2[None, :, :]) ** 2), axis=2)
        K = np.exp((-0.5 / θ2) * Dist)
        return K


if __name__ == "__main__":
    np.random.seed(0)
    resolution = 100
    z1 = np.random.rand(resolution) * 2.0 - 1.0
    z2 = np.random.rand(resolution) * 2.0 - 1.0
    X = np.empty((resolution, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = (z1 ** 2 - z2 ** 2)
    X += np.random.normal(loc=0, scale=0.0, size=X.shape)

    model = GPLVM(θ1=1.0, θ2=0.03, θ3=0.05)
    history = model.fit(X, latent_dim=2, epoch=100, eta=0.00001)

    # ---------描写---------------------------------------------------------------

    fig = plt.figure(figsize=(10, 5))
    ax_observable = fig.add_subplot(122, projection='3d')
    ax_latent = fig.add_subplot(121)


    def update(i, z, x, f):
        plt.cla()
        ax_latent.cla()
        ax_observable.cla()

        fig.suptitle(f"epoch: {i}")
        ax_latent.scatter(z[i, :, 0], z[i, :, 1], s=50, edgecolors="k", c=x[:, 0])
        ax_observable.scatter(x[:, 0], x[:, 1], x[:, 2], c=x[:, 0], s=50, marker='x')
        ax_observable.plot_wireframe(f[i, :, :, 0], f[i, :, :, 1], f[i, :, :, 2], color='black')

        ax_observable.set_xlim(x[:, 0].min(), x[:, 0].max())
        ax_observable.set_ylim(x[:, 1].min(), x[:, 1].max())
        ax_observable.set_zlim(x[:, 2].min(), x[:, 2].max())

        ax_observable.set_title('observable_space')
        ax_observable.set_xlabel("X_dim")
        ax_observable.set_ylabel("Y_dim")
        ax_observable.set_zlabel("Z_dim")
        ax_latent.set_title('latent_space')
        ax_latent.set_xlabel("X_dim")
        ax_latent.set_ylabel("Y_dim")


    ani = animation.FuncAnimation(fig, update, fargs=(history['Z'], X, history['f']), interval=100,
                                  frames=100)
    # ani.save("tmp.gif", writer = "pillow")
    print("X: {}, Y: {}, Z:{}".format(X.shape, history['f'][0].shape, history['Z'][0].shape))
    plt.show()






