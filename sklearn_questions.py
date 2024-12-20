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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import validate_data, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances

from collections import Counter


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
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
        self.X_, self.y_ = validate_data(self, X, y)
        check_classification_targets(y)
        check_array(X)
        self.classes_ = np.unique(self.y_)
        self.n_features_in_ = self.X_.shape[1]
        return self

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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        distances = pairwise_distances(X, self.X_)
        y_pred = []
        for i in range(len(X)):
            neighbor_idxs = np.argsort(distances[i])[: self.n_neighbors]
            neighbor_classes = self.y_[neighbor_idxs]
            most_common = Counter(neighbor_classes).most_common(1)[0][0]
            y_pred.append(most_common)
        return np.array(y_pred)


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

    def __init__(self, time_col="index"):  # noqa: D107
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
        time_col = self._extract_time_column(X)
        time_col = pd.to_datetime(time_col)
        unique_months = time_col.dt.to_period("M").unique()
        return len(unique_months) - 1

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
        self._extract_time_column(X)

        # Sort the data by the time column or by index if 'index'
        if self.time_col == "index":
            X_sorted = X.sort_index()
        else:
            X_sorted = X.sort_values(by=self.time_col)

        # After sorting, extract the sorted time column
        if self.time_col == "index":
            # sorted_time_col is a DatetimeIndex
            sorted_time_col = pd.Series(X_sorted.index, index=X_sorted.index)
        else:
            # sorted_time_col is a Series already
            sorted_time_col = X_sorted[self.time_col]
        sorted_time_col = pd.to_datetime(sorted_time_col)

        # Convert to periods and find unique months
        time_periods = sorted_time_col.dt.to_period("M")
        unique_months = time_periods.unique()

        # For each pair of consecutive months, yield train/test indices
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = time_periods == train_month
            test_mask = time_periods == test_month

            # Indices from the sorted dataframe
            idx_train_sorted = X_sorted.index[train_mask]
            idx_test_sorted = X_sorted.index[test_mask]

            # Map sorted indices back to original indices positions
            idx_train = X.index.get_indexer(idx_train_sorted)
            idx_test = X.index.get_indexer(idx_test_sorted)

            yield idx_train, idx_test

    def _extract_time_column(self, X):
        """Extract the time column or index as a Series.

        Parameters
        ----------
        X : DataFrame
            The input data from which the time column or index is extracted.

        Returns
        -------
        Series
            A Series representing the time column.

        Raises
        ------
        ValueError
            If the time_col is not found or not a datetime.
        """
        if self.time_col == "index":
            time_col = pd.Series(X.index, name="time_col").reset_index(
                drop=True
            )
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"{self.time_col} column not found in input data."
                )
            time_col = X[self.time_col].reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(time_col):
            raise ValueError(
                f"{self.time_col} must be a datetime column or index."
            )

        return time_col
