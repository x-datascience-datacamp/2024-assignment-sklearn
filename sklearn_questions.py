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
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import validate_data
from sklearn.utils.multiclass import check_classification_targets


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the KNearestNeighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use by default for kneighbors queries.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the KNearestNeighbors classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels for the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = validate_data(self, X, reset=False)
        distances = pairwise_distances(X, self.X_train_, metric="euclidean")
        neighbors_idx = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self.y_train_[neighbors_idx]

        y_pred = [
            np.bincount(labels).argmax() for labels in neighbors_labels
        ]
        return np.array(y_pred)

    def score(self, X, y):
        """Calculate the accuracy of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to score on.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        score : float
            Accuracy of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Monthly cross-validator.

    Parameters
    ----------
    time_col : str, default="index"
        Name of the datetime column used for splitting.
    """

    def __init__(self, time_col="index"):
        """Initialize the MonthlySplit cross-validator.

        Parameters
        ----------
        time_col : str, default="index"
            Name of the datetime column used for splitting.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored.
        groups : array-like of shape (n_samples,)
            Always ignored.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        time_col = X.index if self.time_col == "index" else X[self.time_col]
        if not isinstance(time_col, pd.DatetimeIndex):
            time_col = pd.to_datetime(time_col)
        unique_months = time_col.to_period("M").unique()
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Always ignored.
        groups : array-like of shape (n_samples,)
            Always ignored.

        Yields
        ------
        idx_train : ndarray
            Training set indices for the split.
        idx_test : ndarray
            Test set indices for the split.
        """
        time_col = X.index if self.time_col == "index" else X[self.time_col]
        if not isinstance(time_col, pd.DatetimeIndex):
            time_col = pd.to_datetime(time_col)
        months = time_col.to_period("M")

        unique_months = months.unique()
        for i in range(len(unique_months) - 1):
            train_mask = months == unique_months[i]
            test_mask = months == unique_months[i + 1]

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            train_idx.sort()
            test_idx.sort()

            yield train_idx, test_idx
