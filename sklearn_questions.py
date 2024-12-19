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
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """
        Initialize the classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to consider for classification.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the KNearestNeighbors classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["X_", "y_"])
        X = check_array(X)

        # Compute pairwise distances and find the nearest neighbors
        distances = pairwise_distances(X, self.X_)
        nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

        # Predict the majority class among neighbors
        y_pred = np.array([
            np.argmax(np.bincount(self.y_[indices]))
            for indices in nearest_indices
        ])
        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Accuracy of the classifier.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator based on monthly split.

    Splits data into train and test sets where each split corresponds
    to one month of data for training and the next month of data for testing.

    Parameters
    ----------
    time_col : str, default='index'
        Column of the DataFrame that will be used to split the data.
        This column should be of type datetime.
        If using the index, set `time_col='index'`.
    """

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : array-like of shape (n_samples,), optional
            Group labels for the samples.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        time_index = self._get_time_index(X)
        months = time_index.to_period('M').unique()
        return len(months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and testing sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), optional
            Target values.
        groups : array-like of shape (n_samples,), optional
            Group labels for the samples.

        Yields
        ------
        train_idx : ndarray
            Training set indices for that split.
        test_idx : ndarray
            Testing set indices for that split.
        """
        time_index = self._get_time_index(X)
        months = time_index.to_period('M')

        unique_months = months.unique()
        for i in range(len(unique_months) - 1):
            train_mask = (months == unique_months[i])
            test_mask = (months == unique_months[i + 1])
            yield np.where(train_mask)[0], np.where(test_mask)[0]

    def _get_time_index(self, X):
        """
        Extract the time index from the input data.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_index : Series
            Time index of the data.
        """
        if self.time_col == 'index':
            time_index = X.index
        else:
            time_index = X[self.time_col]

        if not np.issubdtype(time_index.dtype, np.datetime64):
            raise ValueError(f"The time column '{self.time_col}' must be of datetime type.")

        return time_index
