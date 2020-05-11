import numpy as np
import data_processing
import dataplot
import model_training as mt
from sklearn.model_selection import train_test_split


def heart_disease(data_file, iterations=1000, learning_rate=0.01, plot_training_costs=False, plot_learning_curves=False):
    dataset = data_processing.read_dataset(data_file)

    X, y = data_processing.preprocess(dataset)
    n_features = X.shape[1]
    parameters = mt.init_parameters(n_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_test = data_processing.standardize(X_train, X_test)

    costs_iterations_train, parameters = mt.train_classifier(X_train.T, y_train.T, parameters, iterations, learning_rate)

    
    if plot_training_costs:
        dataplot.plot_training_costs(costs_iterations_train)

        
    if plot_learning_curves:
        costs_train, costs_test, m_examples = mt.train_various_sizes(X_train.T, X_test.T, y_train.T, y_test.T, parameters, iterations, learning_rate)
        dataplot.plot_learning_curves(costs_train, costs_test, m_examples)
    
    
    train_accuracy, test_accuracy = mt.find_accuracy(X_train.T, X_test.T, y_train.T, y_test.T, parameters)
    print(f"Train accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")


    



