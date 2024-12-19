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
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.metrics.pairwise import pairwise_distances
import pandas.api.types as pdtypes


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """
        Initialize the KNearestNeighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use for classification.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        """
        Predict class labels for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["X_", "y_"])
        X = validate_data(self, X, reset=False)
        y_pred = np.empty(X.shape[0], dtype=self.y_.dtype)

        for i, x in enumerate(X):
            distances = np.sum((self.X_ - x) ** 2, axis=1)
            nearest_indices = np.argpartition(distances, self.n_neighbors)[:self.n_neighbors]
            nearest_labels = self.y_[nearest_indices]
            y_pred[i] = Counter(nearest_labels).most_common(1)[0][0]

        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels for `X`.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator based on monthly split."""

    def __init__(self, time_col="index"):
        """
        Initialize the MonthlySplit cross-validator.

        Parameters
        ----------
        time_col : str, default="index"
            Column of the input DataFrame that contains datetime values
            or "index" to use the DataFrame index.
        """
        self.time_col = time_col


    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splits.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : Ignored
            Compatibility parameter.
        groups : Ignored
            Compatibility parameter.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        time_values = self._get_time_values(X)
        unique_months = time_values.to_period("M").unique()
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices for training and test sets.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : Ignored
            Compatibility parameter.
        groups : Ignored
            Compatibility parameter.

        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """
        time_values = self._get_time_values(X)
        unique_months = time_values.to_period("M").unique()

        for i in range(len(unique_months) - 1):
            train_mask = time_values.to_period("M") == unique_months[i]
            test_mask = time_values.to_period("M") == unique_months[i + 1]

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def _get_time_values(self, X):
        """
        Extract and validate datetime values.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_values : Series
            A series of datetime values.
        """
        if self.time_col == "index":
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("The index must be a DatetimeIndex when time_col='index'.")
            return X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}' not found in the DataFrame.")
            time_values = pd.to_datetime(X[self.time_col], errors="coerce")
            if time_values.isnull().any():
                raise ValueError(f"Column '{self.time_col}' contains invalid datetime values.")
            return time_values
