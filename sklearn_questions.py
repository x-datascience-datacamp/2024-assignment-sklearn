import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import (
    check_is_fitted, validate_data
)
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
        X, y = validate_data(X, y, accept_sparse=True, multi_output=False)
        check_classification_targets(y)
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
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
        check_is_fitted(self, ['_X_train', '_y_train'])
        X = validate_data(X, accept_sparse=True, reset=False)
        y_pred = np.zeros(X.shape[0], dtype=self._y_train.dtype)
        distances = pairwise_distances(X, self._X_train, metric='euclidean')
        closest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        closest_labels = self._y_train[closest_indices]

        for i, labels in enumerate(closest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]

        return y_pred

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

    def _prepare_data(self, X):
        """Prepare and validate the data for splitting.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_series : Series
            Validated time series data.
        """
        if self.time_col == "index":
            time_series = pd.Series(X.index, name="time_col").reset_index(
                drop=True
            )
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column {self.time_col} not found in data.")
            time_series = X[self.time_col].reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(time_series):
            raise ValueError(f"{self.time_col} must be a datetime.")

        return time_series

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
        time_series = self._prepare_data(X)
        unique_months = pd.to_datetime(time_series).dt.to_period('M').unique()
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
        time_series = self._prepare_data(X)
        unique_months = pd.to_datetime(time_series).dt.to_period('M').unique()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = time_series.dt.to_period('M') == train_month
            test_mask = time_series.dt.to_period('M') == test_month

            idx_train = X.index[train_mask]
            idx_test = X.index[test_mask]

            yield X.index.get_indexer(idx_train), X.index.get_indexer(idx_test)
