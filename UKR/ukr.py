import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.animation as animation

class UKR:
    def __init__(self,X,epoch,lamda,wire,eta,latent_dim,sigma):

        self.lamda = lamda
        self.wire = wire
        self.eta = eta
        self.N,self.D = X.shape
        self.L = latent_dim
        self.sigma = sigma
        self.T = epoch
        self.x = torch.from_numpy(X).float()
        self.history_z =np.zeros((self.T, self.N, self.L))
        self.history_y = np.zeros((self.T, self.N, self.D))
        self.history_y_new = np.zeros((self.T, self.wire, self.wire, self.D))


    def fit(self):
        self.K = lambda z1, z2: torch.exp(-torch.cdist(z1, z2)**2 / (2 * self.sigma**2))
        self.z = torch.normal(mean=0, std=0.1, size=(self.N, self.L))
        # print(self.z)
        self.z.requires_grad = True
        for t in range(self.T):
            self.estimate_f(self.z,self.z)
            self.estimetate_z()
            self.history_z[t] = self.z.detach().numpy()
            self.history_y[t] = self.y.detach().numpy()
            z_new = np.dstack(np.meshgrid(np.linspace(min(self.z.detach().numpy()[:, 0]), max(self.z.detach().numpy()[:, 0]), self.wire), np.linspace(min(self.z.detach().numpy()[:,1]),max(self.z.detach().numpy()[:,1]),self.wire)))
            print(z_new.shape)
            z_new = np.reshape(z_new,(-1,2))
            print(z_new.shape)
            z_new = torch.from_numpy(z_new).float()
            print("z",z_new.shape)
            self.y_new = self.estimate_f(self.z,z_new).reshape(self.wire,self.wire,self.D)
            self.history_y_new[t] = self.y_new.detach().numpy()

    def estimetate_z(self):
        E = torch.sum((self.y - self.x)**2)
        E.backward()
        self.z.requires_grad = False
        self.z = self.z - self.eta * self.z.grad
        self.z.requires_grad = True
        return self.z

    def estimate_f(self, z1, z2): #z1は新規
        kernel = self.K(z2,z1)
        print(kernel.shape)
        z1 = torch.cat((z1, torch.ones((z1.shape[0], 1))),dim=1).clone()
        z2 = torch.cat((z2, torch.ones((z2.shape[0], 1)),),dim=1).clone()
        A = torch.einsum("nl,kn,nm->klm", z2, kernel, z2)
        inv = torch.inverse(A)
        r = torch.einsum("kl,klm,nm,kn->kn", z1, inv, z2, kernel)
        self.y = r @ self.x
        print(self.y.shape)

if __name__ == '__main__':
    from data import gen_saddle_shape
    X = gen_saddle_shape(100, noise_scale=0.0)
    ukr = UKR(X ,epoch=100, lamda=0.0001, wire=20, eta=2, latent_dim=2, sigma=1)
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
        ax_latent.set_xlabel("X_dim")
        ax_latent.set_ylabel("Y_dim")

    ani = animation.FuncAnimation(fig, update, fargs=(ukr.history_z, ukr.history_y, X, ukr.history_y_new), interval=100, frames=ukr.T)
    # ani.save("tmp.gif", writer = "pillow")
    plt.show()
