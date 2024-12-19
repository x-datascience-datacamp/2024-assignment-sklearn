"""Assignment - making a sklearn estimator and CV splitter.

The goal of this assignment is to implement by yourself:
- A scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- A scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions are provided in the original problem description.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from pandas.api.types import is_datetime64_any_dtype as is_datetime


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
        Fit the classifier using the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_, self.y_ = X, y
        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=None, reset=False)
        y_pred = np.zeros(X.shape[0], dtype=self.y_.dtype)
        for i, x_test in enumerate(X):
            distances = pairwise_distances(x_test.reshape(1, -1), self.X_)
            indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
            unique, counts = np.unique(self.y_[indices], return_counts=True)
            y_pred[i] = unique[np.argmax(counts)]
        return y_pred

    def score(self, X, y):
        """
        Calculate the accuracy of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to score on.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        return np.mean(self.predict(X) == y)


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator based on monthly splits.

    Each split corresponds to training on one month's data and testing on the
    following month's data.

    Parameters
    ----------
    time_col : str, default='index'
        Column of the input DataFrame that will be used to split the data.
        This column should be of type datetime. If split is called with a
        DataFrame where this column is not datetime, it will raise .
    """

    def __init__(self, time_col='index'):
        """
        Initialize the MonthlySplit cross-validator.

        Parameters
        ----------
        time_col : str, default='index'
            Column used for datetime-based splitting.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : None
            Ignored.
        groups : None
            Ignored.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        newX = X.reset_index() if self.time_col == 'index' else X.copy()

        if not is_datetime(newX[self.time_col]):
            raise ValueError(f"{self.time_col} should be of type datetime.")

        start_date = newX[self.time_col].max()
        end_date = newX[self.time_col].min()
        return (
            12 * (start_date.year - end_date.year)
            + start_date.month
            - end_date.month
        )

    def split(self, X, y=None, groups=None):
        """
        Generate indices for training and test sets.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : None
            Ignored.
        groups : None
            Ignored.

        Yields
        ------
        idx_train : ndarray
            Training set indices.
        idx_test : ndarray
            Test set indices.
        """
        newX = X.reset_index()
        n_splits = self.get_n_splits(newX, y, groups)

        Xtodivide = (
            newX.sort_values(self.time_col)
            .groupby(pd.Grouper(key=self.time_col, freq="ME"))
        )
        idxs = [batch.index for _, batch in Xtodivide]

        for i in range(n_splits):
            idx_train = list(idxs[i])
            idx_test = list(idxs[i + 1])
            yield idx_train, idx_test
