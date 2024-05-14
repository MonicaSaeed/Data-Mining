import numpy as np

class NaiveBayes:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        # get the number of classes in the target
        n_classes = len(np.unique(y))
        # get probabilities for y
        self.class_probs = {}
        for i in range(n_classes):
            self.class_probs[i] = len(np.where(y == i)[0]) / len(y)

        # get the number of features in the input
        n_features = X.shape[1]

        # for each feature, calculate the probabilities for each class
        self.model = {}
        for i in range(n_classes):
            # get the indices of the rows where the target class is equal to the current class
            indices = np.where(y == i)[0]
            self.model[i] = {}
            for j in range(n_features):
                # get the unique values in the current feature
                unique_values = np.unique(X[:, j])
                self.model[i][j] = {}
                for value in unique_values:
                    # get the indices of the rows where the current feature is equal to the current value
                    value_indices = np.where(X[:, j] == value)[0]
                    # calculate the probability of the current value given the current class
                    prob = len(np.intersect1d(indices, value_indices)) / len(indices)
                    # Handle zero probability
                    if prob == 0:
                        total = len(indices)+len(unique_values)
                        for k in unique_values:
                            if k == value:
                                self.model[i][j][k] = 1/total
                            else:
                                self.model[i][j].setdefault(k, (len(np.intersect1d(indices, value_indices)) + 1) / total)
                        continue
                    self.model[i][j][value] = prob

    def predict(self, X):
        predictions = []
        for x in X:
            probs = []
            for i in range(len(self.class_probs)):
                prob = self.class_probs[i]
                for j in range(len(x)):
                    prob *= self.model[i][j][x[j]]
                probs.append(prob)
            # get the class with the highest probability
            predictions.append(np.argmax(probs))
        return predictions

    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

    def print_model(self):
        print("Class Probabilities:")
        for i in self.class_probs:
            print(f"Class {i}: {self.class_probs[i]}")
        print("\nModel:")
        for i in self.model:
            print(f"Class {i}:")
            for j in self.model[i]:
                print(f"Feature " + str(j) + ":")
                for k in self.model[i][j]:
                    print(f"Value {k}: {self.model[i][j][k]}")
                print()
            print()
