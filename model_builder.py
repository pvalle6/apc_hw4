from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# Importing the parent: DataPreprocessing class from data_preprocess.py
from data_preprocess import DataPreprocessing


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create DT model
        DT_classifier = DecisionTreeClassifier()

        #Train the model
        DT_classifier.fit(X_train, y_train)

        #Test the model
        DT_predicted = DT_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(DT_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, DT_predicted)

        return DT_classifier

    def neural(self, X_train, X_test, y_train, y_test):
        #Create Neural Network model
        neural_classifier = MLPClassifier()

        #Train the model
        neural_classifier.fit(X_train, y_train)

        #Test the model
        neural_predicted = neural_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(neural_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, neural_predicted)

        return neural_classifier


if __name__ == "__main__":
    MB = ModelBuilder()
    data = MB.load_data("data.txt")
    X_train, X_test, y_train, y_test = MB.split_data(data)
    MB.dt(X_train, X_test, y_train, y_test)
    MB.neural(X_train, X_test, y_train, y_test)

