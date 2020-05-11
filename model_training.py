import numpy as np

    

def init_parameters(n_features):
    W = np.zeros((1, n_features))
    b = 0
    return {"W": W, "b": b}



def train_various_sizes(X_train, X_test, y_train, y_test, parameters, iterations, learning_rate):
    costs_train, costs_test, m_examples = [], [], []
    for i in range(1, X_train.shape[1], 20):
        parameters = init_parameters(X_train.shape[0])
        costs_iterations, parameters = train_classifier(X_train[:, :i], y_train[:, :i], parameters, iterations, learning_rate)
        cost_test = compute_cost(compute_A(X_test, parameters), y_test, X_test.shape[1])

        costs_train.append(costs_iterations["costs"][-1])
        costs_test.append(cost_test)
        m_examples.append(i)

    return costs_train, costs_test, m_examples



def train_classifier(X, y, parameters, iterations, learning_rate):
    m = X.shape[1]
    logging_frequency = 10
    costs_iterations = {"costs": [], "iterations": []}

    for i in range(iterations):
        A = compute_A(X, parameters)
        cost, gradients = propagate(X, A, y, m)

        if i % logging_frequency == 0:
            costs_iterations["costs"].append(cost)
            costs_iterations["iterations"].append(i)
                         
        update_parameters(parameters, gradients, learning_rate)

    return costs_iterations, parameters


def compute_A(X, parameters):
    z = np.dot(parameters["W"], X) + parameters["b"]
    a = sigmoid(z)
    return a


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(A, y, m):
    return -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))


def compute_gradients(X, A, y, m):
    dW = (1 / m) * np.dot(A - y, X.T)
    db = (1 / m) * np.sum(A - y)
    return {"dW": dW, "db": db}
    

def propagate(X, A, y, m):
    cost = compute_cost(A, y, m)
    gradients = compute_gradients(X, A, y, m)
    return cost, gradients


def update_parameters(parameters, gradients, learning_rate):
    parameters["W"] = parameters["W"] - learning_rate * gradients["dW"]
    parameters["b"] = parameters["b"] - learning_rate * gradients["db"]
    

def find_accuracy(X_train, X_test, y_train, y_test, parameters):
    A_train = compute_A(X_train, parameters)
    A_test = compute_A(X_test, parameters)

    y_pred_train = np.where(A_train >= 0.5, 1, 0)
    y_pred_test = np.where(A_test >= 0.5, 1, 0)

    train_accuracy = compute_accuracy(y_pred_train, y_train)
    test_accuracy = compute_accuracy(y_pred_test, y_test)
    return train_accuracy, test_accuracy


def compute_accuracy(y_pred, y):
    comparison = np.where(y_pred == y, 1, 0)
    return np.sum(comparison) / len(y[0])
