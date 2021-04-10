import numpy as np
import matplotlib.pyplot as plt
class SOM(object):
    def __init__(self, latent_dim, epoch, K, σ_max, σ_min):
        self.L = latent_dim
        self.K = K
        self.T = epoch
        self.σ_max = σ_max
        self.σ_min = σ_min
        self.y = np.random.rand(self.K, D)

    def fit(self, X):
        N, D = X.shape
        zeta = np.random.rand(N, self.L)
        for T in range(self.T):
            σ = ((self.σ_min - self.σ_max) / self.T)*T + self.σ_max
            Dist = np.sum((X[None, :, :] - self.y[:, None, :])**2,axis=2)
            k_star = np.argmin(Dist, axis=0)
            print(k_star.shape)
            Z = zeta[k_star, :]
            self.y = self.NW(zeta, Z, X, σ)

    def NW(self, zeta, Z, X, σ):
        Dist = np.sum((zeta[None, :, :] - Z[:, None, :])**2, axis=2)
        K = np.exp(-0.5/(σ**2)*Dist)
        # G = np.sum(K, axis=1).reshape(-1,1)
        G = np.sum(K, axis=0)[:, None]
        H = K.T @ X
        return H / G

if __name__ == '__main__':
    N = 100
    D = 2
    X = np.zeros([N,D])
    X[:,0] = np.random.rand(N)
    X[:,1] = np.sin(X[:,0])
    som = SOM(latent_dim=1, epoch=200, K=10, σ_max=2.0, σ_min=0.03)
    som.fit(X)

    plt.scatter(X[:,0], X[:,1])
    plt.scatter(som.y[:,0],som.y[:,1])
    plt.show()



