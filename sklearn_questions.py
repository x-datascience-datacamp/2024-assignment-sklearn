import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from collections import Counter


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
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
        check_is_fitted(self, ['_X_train', '_y_train'])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        y_pred = np.zeros(X.shape[0], dtype=self._y_train.dtype)
        dist = pairwise_distances(X, self._X_train, metric='euclidean')
        sorted_indi = np.argsort(dist, axis=1)
        closest_indi = sorted_indi[:, :self.n_neighbors]
        closest_labels = self._y_train[closest_indi]
        for i, labels in enumerate(closest_labels):
            counts = Counter(labels)
            y_pred[i] = max(counts, key=counts.get)
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit:
    def __init__(self, time_col="index"):
        self.time_col = time_col

    def __repr__(self):
        return f"MonthlySplit(time_col='{self.time_col}')"

    def get_n_splits(self, X, y=None, groups=None):
        time_series = X.index.to_series() if self.time_col == 'index' else \
            pd.to_datetime(X[self.time_col])
        if self.time_col != 'index' and self.time_col not in X.columns:
            raise ValueError(f"Column '{self.time_col}' not found in data.")
        return len(time_series.dt.to_period('M').unique()) - 1

    def split(self, X, y=None, groups=None):
        if self.time_col == 'index':
            time_col = pd.Series(X.index, name='time_col') \
                .reset_index(drop=True)
            X_sorted = X.sort_index()
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} column not found .")
            time_col = X[self.time_col].reset_index(drop=True)
            if not pd.api.types.is_datetime64_any_dtype(time_col):
                raise ValueError(f"{self.time_col} must be a datetime column.")
            X_sorted = X.sort_values(by=self.time_col)

        time_col_sorted = pd.Series(
            X_sorted.index
            if self.time_col == 'index'
            else X_sorted[self.time_col]
        ).reset_index(drop=True)

        time_periods = pd.to_datetime(time_col_sorted).dt.to_period('M')
        unique_months = time_periods.unique()
        for i in range(len(unique_months) - 1):
            train_month, test_month = unique_months[i], unique_months[i + 1]
            train_mask = time_periods == train_month
            test_mask = time_periods == test_month
            idx_train = X_sorted.index[train_mask]
            idx_test = X_sorted.index[test_mask]
            yield X.index.get_indexer(idx_train), X.index.get_indexer(idx_test)
