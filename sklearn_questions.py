import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fitting function.

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
        X, y = self._validate_data(
            X, y, accept_sparse=True, multi_output=False
        )
        check_classification_targets(y)
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self, ['_X_train', '_y_train'])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        y_pred = np.zeros(X.shape[0], dtype=self._y_train.dtype)
        distances = pairwise_distances(X, self._X_train, metric='euclidean')
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self._y_train[nearest_indices]
        for i, labels in enumerate(nearest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]
        return y_pred

    def score(self, X, y):
        """
        Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        check_is_fitted(self, ['_X_train', '_y_train'])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        y = self._validate_data(y, ensure_2d=False, reset=False)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy


class MonthlySplit(BaseCrossValidator):
    """
    CrossValidator based on monthly split.

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

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

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
        if not isinstance(X, pd.DataFrame):
            x_df = pd.DataFrame({'date': X.index, 'val': X.values})
            x_df['date'] = pd.to_datetime(x_df['date'])
        elif self.time_col == 'index':
            x_df = X.reset_index().copy()
            x_df = x_df.rename(columns={'index': 'date'})
        else:
            x_df = X.copy()
            x_df['date'] = pd.to_datetime(x_df[self.time_col])
        months = pd.to_datetime(x_df['date']).dt.to_period('M')
        return len(months.unique()) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

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
        if self.time_col != 'index':
            if not isinstance(X[self.time_col].iloc[0], pd.Timestamp):
                raise ValueError(
                    'The specified time column is not datetime.'
                )
        else:
            if not isinstance(X.index[0], pd.Timestamp):
                raise ValueError('The index is not datetime.')

        if not isinstance(X, pd.DataFrame):
            x_df = pd.DataFrame({'date': X.index, 'val': X.values})
            x_df['date'] = pd.to_datetime(x_df['date'])
        elif self.time_col == 'index':
            x_df = X.reset_index().copy()
            x_df = x_df.rename(columns={'index': 'date'})
        else:
            x_df = X.copy()
            x_df['date'] = pd.to_datetime(x_df[self.time_col])

        x_df['month'] = pd.to_datetime(x_df['date']).dt.to_period('M')
        unique_months = x_df['month'].unique()

        for i in range(len(unique_months) - 1):
            train_idx = x_df[x_df['month'] == unique_months[i]].index
            test_idx = x_df[x_df['month'] == unique_months[i + 1]].index
            yield train_idx, test_idx
