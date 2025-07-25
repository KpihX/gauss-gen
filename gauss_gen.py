# %%
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as sps

# %%
def simulation_X(K, theta, n):
    """Simulate ``n`` observations from a ``K``-component Gaussian mixture."""
    simul_Z = npr.choice(range(1, K + 1), p=theta[0], size=n)
    return [sps.multivariate_normal(theta[1][z - 1], theta[2][z - 1]).rvs() for z in simul_Z], simul_Z


# %%
# def pdf_X_1D(x, theta):
#     # print("*", 1 / (np.sqrt(2 * np.pi * theta[2])))
#     return np.sum(theta[0] * 1 / (np.sqrt(2 * np.pi * theta[2])) * np.exp(-1/2 * (x - theta[1]) ** 2 / theta[2]))

def pdf_X(x, theta: tuple[np.ndarray]):
    """Probability density function of the mixture at point ``x``."""
    return theta[0].dot(
        np.array([sps.multivariate_normal(theta[1][k], theta[2][k]).pdf(x) for k in range(len(theta[0]))])
    )

# def pdf_X2(x, theta):
#     d = x.shape[0]
#     diff = x - theta[1]
#     exp_fact = np.exp(-0.5 * np.array([d.T @ sigma @ d for d, sigma in zip(diff, np.linalg.inv(theta[2]))]))
#     # print("**", x, "*", theta[1], "\n***", diff)
#     # print("*", diff.T.shape)
#     # print("*", np.linalg.inv(theta[2]).shape)
#     # print("*", diff.shape)
#     return np.sum(theta[0] * 1 / ((2 * np.pi) ** (d/2) * np.sqrt(np.linalg.det(theta[2]))) * exp_fact)

# %%

# %% [markdown]
# ## 2 Algorithme d’estimation de paramètres de mélange gaussien

# %%
def likelihood(K, theta, X):
    """Return the log-likelihood of data ``X`` under parameters ``theta``."""
    return np.log([pdf_X(x, theta) for x in X]).sum()

# %%
def pdf_Z_X(x, theta):
    """Posterior probabilities of latent classes given ``x`` and ``theta``."""
    return theta[0] * np.array(
        [sps.multivariate_normal(theta[1][k], theta[2][k]).pdf(x) for k in range(len(theta[0]))]
    ) / pdf_X(x, theta)


def theta_estim(X, K, n_iter=100):
    """Estimate mixture parameters using the EM algorithm."""
    n = len(X)
    Z = npr.choice(range(1, K+1), size=n)

    # Init of Mu and Sigma to deal woth cases where we can have some n_k = 0
    d = X.shape[1]
    Mu = np.random.randn(K, d)
    Sigma = np.array([np.random.uniform(0, abs(mu)**2, size=(d,d)) for mu in Mu])

    Mu_hist = []
    
    while (n_iter > 0):
        # print("Z:", Z)
        # M-step
        # To optimise
        filter = np.array([Z == k for k in range(1, K+1)])
        # print("filter: ", filter.shape)
        N = np.sum(filter, axis=1)
        print("N:", N)
        Pi = N / n
        # print("Pi:", Pi)
        X_filtered = np.array([X * filter[k].reshape(n, 1) for k in range (K)])
        # print("X_filtered: ", X_filtered.shape)
        _Mu = np.sum(X_filtered, axis=1)
        Mu = np.array([_Mu[k] / N[k] if N[k] != 0 else Mu[k] for k in range(K)]) # Case where n_k = 0, we consider old Mu_k
        # print("Mu: ", Mu)
        Mu_hist.append(Mu)
        # We have to transpose first to have d * d at the end
        diff_T = np.array([(X_filtered[k] - Mu[k]) for k in range(K)])
        # print("diff: ", diff_T.shape)
        _Sigma = np.array([d.T.dot(d) for d in diff_T])
        Sigma = np.array([_Sigma[k] / N[k] if N[k] != 0 else Sigma[k] for k in range(K)])  # Case where n_k = 0, we consider old Sigma_k
        # print("Sigma: ", Sigma)
        theta = (Pi, Mu, Sigma)

        # E-step
        Z = np.array([np.argmax(pdf_Z_X(x, theta)) + 1 for x in X])
        n_iter -= 1

    return theta, np.array(Mu_hist)





if __name__ == "__main__":
    # Example 1: generate and plot a 1D Gaussian mixture
    d = 1
    K = 5
    np.random.seed(0)
    Pi = np.random.dirichlet(np.ones(K))
    print("Pi:", Pi)
    Mu = np.random.randn(K, 1)
    print("Mu:", Mu)
    Sigma = np.array([np.random.uniform(0, abs(mu) ** 2, size=(1, 1)) for mu in Mu])
    print("Sigma:", Sigma)
    theta = (Pi, Mu, Sigma)

    N = 1000
    b = int(N ** (1.0 / 3.0)) * 7
    X_samples, Z_samples = simulation_X(K, theta, N)
    plt.hist(X_samples, density=True, label=f"Echantillon de X de taille {N}", bins=b)
    X = np.linspace(-7, 7, 1000).reshape(1000, 1)
    plt.plot(X, np.array([pdf_X(x, theta) for x in X]), label="Densité de probabilité de X")
    plt.legend()
    plt.show()

    # Example 2: estimation with synthetic 2D data
    from sklearn.datasets import make_blobs

    n = 300
    K = 3
    X, y = make_blobs(n_samples=n, centers=K, n_features=2, cluster_std=1.0, random_state=42)

    theta, Mu_hist = theta_estim(X, K, n_iter=15)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30)
    plt.title('Jeu de données simulé (3 composantes)')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    m = Mu_hist.shape[1]
    for k in range(m):
        traj = Mu_hist[:, k, :]
        plt.plot(traj[:, 0], traj[:, 1], marker='o', label=f'Centre {k+1}')
        for t in range(len(traj) - 1):
            plt.arrow(
                traj[t, 0],
                traj[t, 1],
                traj[t + 1, 0] - traj[t, 0],
                traj[t + 1, 1] - traj[t, 1],
                head_width=0.15,
                head_length=0.2,
                fc=f"C{k}",
                ec=f"C{k}",
                alpha=0.7,
                length_includes_head=True,
            )

    plt.title("Trajectoire des centres au fil des itérations")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()




