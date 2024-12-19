"""Assignment - making a sklearn estimator and cv splitter."""
import numpy as np

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


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
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.X_train_ = X
        self.y_train_ = y
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
        y_pred = []
        for _, x in enumerate(X):
            distances = pairwise_distances(x.reshape(1, -1), self.X_train_)
            idx = np.argsort(distances, axis=1)[0][:self.n_neighbors]
            values, counts = np.unique(self.y_train_[idx], return_counts=True)
            y_pred.append(values[np.argmax(counts)])
        y_pred = np.array(y_pred)
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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


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
        """Return the number of splitting iterations in the cross-validator."""
        if self.time_col == 'index':
            X_time = X.reset_index()
        else:
            X_time = X.copy()
        if X_time[self.time_col].dtype != 'datetime64[ns]':
            raise ValueError(f"Column '{self.time_col}' is not a datetime.")
        sorted = X_time.sort_values(by=self.time_col)
        n_splits = len(sorted[self.time_col].dt.to_period('M').unique()) - 1
        return n_splits

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set."""
        X_copy = X.reset_index()
        n_splits = self.get_n_splits(X_copy, y, groups)
        X_grouped = (
            X_copy.sort_values(by=self.time_col)
            .groupby(pd.Grouper(key=self.time_col, freq="M"))
        )
        idxs = [group.index for _, group in X_grouped]
        for i in range(n_splits):
            idx_train = list(idxs[i])
            idx_test = list(idxs[i+1])
            yield (idx_train, idx_test)
