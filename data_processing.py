import pandas as pd
from sklearn.preprocessing import StandardScaler


def read_dataset(data_file):
    dataset = pd.read_csv(data_file)
    return dataset.values


def preprocess(dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y


def standardize(X1, X2):
    sc = StandardScaler()
    X1 = sc.fit_transform(X1)
    X2 = sc.transform(X2)
    return X1, X2
