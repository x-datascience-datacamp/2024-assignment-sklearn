"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `check_*` functions imported in the file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `fon january etc.lake8`. You can check that there is no flake8
errors by calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score
from collections import Counter


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        X, y = check_X_y(X, y)
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
        return self

    def euclideanDistance(x1, x2):
        """
        compute the euclidean distance between two points x1 and x2
        """
        return pairwise_distances([x1], [x2], metric='euclidean')[0][0]

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, ['X_train', 'y_test'])
        X = check_array(X)
        X = np.array(X)  # X is the test set ( or unseen data)
        y_pred = np.zeros(X.shape[0])
        N = X.shape[0]
        # for each sample in the X_test
        for n in range(N):
            onePointDistances = []
            # step1: computing the distances from X[n] to each point X_train[i]

            for i in range(self.X_train.shape[0]):
                dist = self.euclideanDistance(X[n], self.X_train[i])
                onePointDistances.append((dist, self.y_train[i]))

            # sorting the distances in order to take the nearest one

            onePointDistances.sort(key=lambda x: x[0])

            # taking the k nearest neighbors

            kNearestNeighborsPoint = [
                label for _, label in onePointDistances[:self.n_neighbors]
                ]

            # majority voting by counting the occurences among the neighbors

            counts = Counter(kNearestNeighborsPoint).most_common(1)
            y_pred[n] = counts[0][0]

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        X = check_array(X)
        y = check_classification_targets(y)
        score = 0.0
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        # verify X and y
        X, y = check_X_y(X, y)
        # get the first date in X and the last one
        firstDate = X[self.time_col].min()
        lastDate = X[self.time_col].max()
        # get the number of months from the first to the last date
        # Extract years and months
        if lastDate >= firstDate:
            year_diff = lastDate.year - firstDate.year
            month_diff = lastDate.month - firstDate.month
        else:
            year_diff = 0
            month_diff = 0

        # Total months difference
        total_months = year_diff * 12 + month_diff
        # compute the number of split that we can do
        splits = (total_months + (total_months - 2)) / 2
        return splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """

        X_df = pd.DataFrame(X)
        n_samples = X_df.shape[0]
        n_splits = self.get_n_splits(X, y, groups)
        for i in range(n_splits):
            idx_train = range(n_samples)
            idx_test = range(n_samples)
            yield (
                idx_train, idx_test
            )
