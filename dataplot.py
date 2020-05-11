import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def plot_dataset(X, y):
    X_reduce = PCA(X)
    print(X_reduce[0, :].reshape(1, -1))
    plt.scatter(X_reduce[0], X_reduce[1])
    plt.show()


def plot_training_costs(costs_iterations_train):
    plt.style.use("seaborn")
    plt.plot(costs_iterations_train["iterations"], costs_iterations_train["costs"])
    plt.xlabel("Number of iterations", fontsize=14)
    plt.ylabel("Cost", fontsize=14)
    plt.title("Cross-entropy (log-loss)", fontsize=18, y=1.03)
    plt.show()



def plot_learning_curves(costs_train, costs_test, m_examples):
    plt.style.use("seaborn")
    plt.plot(m_examples, costs_train, label='Training cost')
    plt.plot(m_examples, costs_test, label='Test cost')
    plt.ylabel('Cost', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


