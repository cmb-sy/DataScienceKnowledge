from data import gen_multi_logistic
import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

class UKR2(object):
    def __init__(self, z_dim, u_dim):
        self.z_dim = z_dim
        self.u_dim = u_dim

    def fit(self, X, u_eta=0.1, z_eta=0.5, num_epoch=1000):
        I, N, D = X.shape
        U = np.random.normal(scale=10e-2, size=(I, self.u_dim))
        Z = np.random.normal(scale=10e-2, size=(I, N, self.z_dim))
        X, Z, U = jnp.array(X), jnp.array(Z), jnp.array(U)
        history = dict(
            error=np.zeros((num_epoch,)),
            Z=np.zeros((num_epoch, I, N, self.z_dim)),
            U=np.zeros((num_epoch, I, self.u_dim)),
            Y=np.zeros((num_epoch, I, N, D))
        )
        error_history = []
        Y_history = []
        Z_history = []
        U_history = []

        for epoch in tqdm(range(num_epoch)):
            E = lambda z, u: jnp.sum((F(z, Z, u, U, X) - X)**2)
            dZ, dU = grad(E, argnums=(0, 1))(Z, U)
            Z -= z_eta * dZ
            U -= u_eta * dU

            history['error'][epoch] = E(Z, U)
            history['Y'][epoch] = F(Z, Z, U, U, X)
            history['Z'][epoch] = np.array(Z)
            history['U'][epoch] = np.array(U)
        return history


def h_ui(U1, U2, sigma=1):
    dists = ((U1[:, np.newaxis, :] - U2[np.newaxis, :, :])**2).sum(axis=2)
    R = jnp.exp(-0.5 * dists / sigma**2)
    R /= R.sum(axis=1, keepdims=True)
    return R

def h_zij(Z1, Z2, sigma=1):
    dists = ((Z1[:, None, :, None, :] - Z2[None, :, None, :, :])**2).sum(axis=4)
    R = jnp.exp(-0.5 * dists / sigma**2)
    R /= R.sum(axis=3, keepdims=True)
    return R

def F(Z1, Z2, U1, U2, X):
    r_u = h_ui(U1, U2)
    r_z = h_zij(Z1, Z2)
    return jnp.einsum('in,injm,nmd->ijd', r_u, r_z, X)


from utils import make_meshgrid
from matplotlib.animation import FuncAnimation

if __name__ == '__main__':
    X = gen_multi_logistic(a_list=[1,2,3], epoch=1, inits=30)
    u_dim, z_dim = 1, 1
    num_epoch = 100
    ukr2 = UKR2(u_dim=1, z_dim=1)
    history = ukr2.fit(X, u_eta=0.5, z_eta=1., num_epoch=num_epoch)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    def draw(X, Z, U):
        u_reso, z_reso = 10, 20
        U_new = np.linspace(U.min(), U.max(), u_reso)[:, None]
        Z_new = np.tile(make_meshgrid(z_reso, (Z.min(), Z.max()), dim=z_dim), (u_reso, 1, 1))

        Y = F(Z_new, Z, U_new, U, X)
        X, Y = np.array(X), np.array(Y)
        for x in X:
            ax.scatter(x[:, 0], x[:, 1])
        for y in Y:
            ax.plot(y[:, 0], y[:, 1])

    # draw(X, history['Z'][-1], history['U'][-1])

    def update(epoch):
        ax.cla()
        fig.suptitle(f"epoch: {epoch}")
        draw(X, history['Z'][epoch], history['U'][epoch])

    ani = FuncAnimation(fig, update, frames=num_epoch, repeat=True, interval=100)
    plt.show()
    ani.save(f"my-first-ukr2.gif", writer='pillow')
