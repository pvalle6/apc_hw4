import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessing():
    def __init__(self):
        pass

    def load_data(self, path):
        data = pd.read_csv(path, sep="\t", header=None).dropna()

        print(data.head())

        data = data.to_numpy()

        return data

    def split_data(self, data):
        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test

