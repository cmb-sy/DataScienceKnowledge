import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from scipy.spatial import distance as dist
from tqdm import tqdm

class UKR:
    def __init__(self,X,epoch,lamda,wire,eta,latent_dim,sigma, seed, cliping=False):
        self.lamda = lamda
        self.wire = wire
        self.eta = eta
        self.N,self.D = X.shape
        self.L = latent_dim
        self.sigma = sigma
        self.seed = seed
        self.T = epoch
        self.x = X
        self.cliping = cliping
        self.history_z =np.zeros((self.T, self.N, self.L))
        self.history_y = np.zeros((self.T, self.N, self.D))
        self.history_y_new = np.zeros((self.T,self.wire, self.wire,self.D))

    def fit(self):
        np.random.seed(self.seed)
        self.z = np.random.normal(scale=0.1, size=(self.N, self.L))
        for t in tqdm(range(self.T)):
            self.nadareya_estimate(self.z,self.z)
            self.estimetate_f()
            self.history_z[t] = self.z
            self.history_y[t] = self.y

            z_new = np.dstack(np.meshgrid(np.linspace(min(self.z[:,0]),max(self.z[:,0]),self.wire), np.linspace(min(self.z[:,1]),max(self.z[:,1]),self.wire)))
            # print(z_new.shape)
            z_new = np.reshape(z_new,(-1,2))
            self.y_new = self.nadareya_estimate(z_new,self.z).reshape(self.wire,self.wire,self.D)
            self.history_y_new[t] = self.y_new

    def estimetate_f(self):
            self.d_ni = self.y[:,None,:]-self.x[None, : ,:]
            self.delta = self.z[:,None,:] - self.z[None,:,:]
            self.A = self.r_ij * np.einsum("nd,nid->ni" , self.y-self.x , self.d_ni , optimize="true")
            self.bibun = (2/self.N) * np.sum((self.A + self.A.T)[:, :, None] * self.delta , axis=1)
            diff = - (self.eta * (self.bibun + self.lamda * self.z))
            self.z = self.z + diff
            if self.cliping:
                self.z = np.clip(self.z,-1.0,1.0)

    def nadareya_estimate(self, z1, z2):
        self.k_numerator = np.exp((-1 / 2 * ((self.sigma) ** 2)) * dist.cdist(z1, z2) ** 2)
        self.k_denominator = np.sum(self.k_numerator, axis=1, keepdims=True)
        self.r_ij = self.k_numerator / self.k_denominator
        self.y = self.r_ij @ self.x
        return self.y

if __name__ == '__main__':
    from data import gen_saddle_shape
    X = gen_saddle_shape(100, noise_scale=0.0)
    ukr = UKR(X ,epoch=100, lamda=0.0001, wire=10, eta=2, latent_dim=2, sigma=1, seed=0)
    ukr.fit()

# ---------描写---------------------------------------------------------------
    fig = plt.figure(figsize=(10, 5))
    ax_observable = fig.add_subplot(122, projection='3d')
    ax_latent = fig.add_subplot(121)

    def update(i, z, y, x , y_new):
        plt.cla()
        ax_latent.cla()
        ax_observable.cla()
        #
        ax_latent.scatter(z[i,:, 0], z[i,:, 1], s=25, alpha=0.5 , c=x[:,0])
        ax_observable.scatter(x[:, 0], x[:, 1], x[:, 2], s=8, c=x[:,0])
        ax_observable.scatter(y[i,:, 0], y[i,:, 1], y[i,:, 2], s=4,c="r")
        ax_observable.plot_wireframe(y_new[i ,:, :, 0], y_new[i, :, :, 1], y_new[i, :, :, 2], color='red')

        ax_observable.set_xlabel("X_dim")
        ax_observable.set_ylabel("Y_dim")
        ax_observable.set_zlabel("Z_dim")
        ax_latent.set_title('latentspace')

    ani = animation.FuncAnimation(fig, update, fargs=(ukr.history_z, ukr.history_y, X, ukr.history_y_new), interval=100, frames=ukr.T)
    # ani.save("tmp.gif", writer = "pillow")
    plt.show()
