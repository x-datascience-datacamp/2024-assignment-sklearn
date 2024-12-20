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
- You can use the:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    check_X_y, check_array, check_is_fitted, validate_data
)
from sklearn.utils.multiclass import unique_labels


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
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
        self : instance of KNearestNeighbors
            The current instance of the classifier.
        """
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        distances = euclidean_distances(X, self.X_)
        k_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        k_labels = self.y_[k_indices]

        y_pred = [
            unique_labels[np.argmax(counts)]
            for labels in k_labels
            for unique_labels, counts in [np.unique(labels, return_counts=True)]
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
        Name of the datetime column used for splitting. Set to 'index' to use the index.
    """

    def __init__(self, time_col="index"):
        self.time_col = time_col

    def _prepare_data(self, X):
        X_copy = X.copy()
        if self.time_col == "index":
            X_copy = X_copy.reset_index()
        if not pd.api.types.is_datetime64_any_dtype(X_copy[self.time_col]):
            raise ValueError(f"The column '{self.time_col}' is not a datetime.")
        return X_copy.sort_values(by=self.time_col)

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
        X = self._prepare_data(X)
        unique_months = X[self.time_col].dt.to_period("M").unique()
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
        X = self._prepare_data(X)
        grouped = X.groupby(pd.Grouper(key=self.time_col, freq="M"))
        idxs = [group.index for _, group in grouped]

        for i in range(len(idxs) - 1):
            yield list(idxs[i]), list(idxs[i + 1])