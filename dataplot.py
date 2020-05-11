import numpy as np
import matplotlib.pyplot as plt


def plot_dataset(X, y):
    X_reduce = PCA(X)
    # print(X_reduce[0, :].reshape(1, -1))
    plt.scatter(X_reduce[0], X_reduce[1])
    plt.show()


def PCA(X):
    U, s, v = np.linalg.svd(X.T)
    U_reduce = U[:, :2]
    # print(sum(s[:2]) / sum(s))
    X_reduce = np.dot(U_reduce.T, X.T)
    return X_reduce


def plot_training_costs(costs_iterations_train):
    plt.plot(costs_iterations_train["iterations"], costs_iterations_train["costs"])
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
