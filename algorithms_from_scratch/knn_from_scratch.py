import numpy as np
import warnings
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.tight_layout()
# the most basic version
# n_neighbors: choice
# metric: minkowski - L1, L2, L3, etc


class KNN:
    # the defaults based on scikit-learn, which are sensible
    def __init__(self, n_neighbors=5, p=2):
        self.n_neighbors = n_neighbors
        self.p = p

    # according to scikit-learn, fit method chooses the appropriate algorithm
    # based on data type. However, this code only applies brute-force based distance metric
    def fit(self, X, y):
        self.X_train = np.array(X) if not isinstance(X, np.ndarray) else X
        self.y_train = np.array(y) if not isinstance(y, np.ndarray) else y

    def predict(self, X_test):
        assert self.n_neighbors <= len(
            self.X_train), "There is not enough nearest neighbors"

        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)

        if self.X_train.shape[-1] != X_test.shape[-1]:
            raise ValueError(
                'test set must have equal features with training set')

        y_pred = []
        for test_data in X_test:
            distances = Counter()
            for i, train_data in enumerate(self.X_train):
                distances[i] = - \
                    np.linalg.norm(train_data - test_data, self.p)

            shortest_distances = distances.most_common(self.n_neighbors)
            most_votings = Counter([self.y_train[i]
                                    for i, dist in shortest_distances])
            y_pred.append(most_votings.most_common(1)[0][0])

        return y_pred

    def predict_proba(self, X_test):  # for voting classifier with "soft" voting
        assert self.n_neighbors <= len(
            self.X_train), "There is not enough nearest neighbors"

        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)

        if self.X_train.shape[-1] != X_test.shape[-1]:
            raise ValueError(
                'test set must have equal features with training set')

        y_pred = []
        for test_data in X_test:
            distances = Counter()
            for i, train_data in enumerate(self.X_train):
                distances[i] = - \
                    np.linalg.norm(train_data - test_data, self.p)

            shortest_distances = distances.most_common(self.n_neighbors)
            most_votings = Counter([self.y_train[i]
                                    for i, dist in shortest_distances])

            total = sum(most_votings.values())
            y_pred.append([(key, freq / total)
                           for key, freq in most_votings.items()])

        return y_pred



# for the sake of visualizing
X = [[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6], [6.5, 3], [7, 2], [8, 3]]
y = ['r', 'r', 'r', 'g', 'g', 'g', 'b', 'b', 'b']

[plt.scatter(X[i][0], X[i][1], color=y[i]) for i in range(len(X))]

knn = KNN(5)
knn.fit(X, y)
X_test = [[6, 8], [3.5, 4], [7, 2.5]]
result = knn.predict(X_test)
print(result)

probas = knn.predict_proba(X_test)
print(probas)

[plt.scatter(X_test[i][0], X_test[i][1], color=result[i],
             marker="*", s=200) for i in range(len(result))]
plt.show()
