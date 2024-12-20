#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import pandas.api.types as pdtypes

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data, check_is_fitted

from collections import Counter


class KNearestNeighbors(ClassifierMixin, BaseEstimator):

    """KNearestNeighbors classifier."""

    def __init__(self, num_neighbors=1):  # noqa: D107
        self.num_neighbors = num_neighbors

    def fit(self, features, labels):
        """Fitting function.

         Parameters
        ----------
        features : ndarray, shape (n_samples, n_features)
            Data to train the model.
        labels : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """

        (features, labels) = validate_data(self, features, labels)
        self.classes_ = unique_labels(labels)
        self.training_features_ = features
        self.training_labels_ = labels
        return self

    def predict(self, features):
        """Predict function.

        Parameters
        ----------
        features : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        predictions : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """

        check_is_fitted(self)
        features = validate_data(self, features, reset=False)

        predictions = np.full(features.shape[0], self.training_labels_[0])
        for idx in range(features.shape[0]):
            feature = features[idx]
            neighbor_labels = []

            distances = np.sum(
                (self.training_features_ - feature) ** 2, axis=1
            )
            nearest_indices = np.argpartition(
                distances, self.num_neighbors
            )[: self.num_neighbors]
            for neighbor_idx in nearest_indices:
                neighbor_labels += [self.training_labels_[neighbor_idx]]

            predictions[idx] = Counter(neighbor_labels).most_common(1)[0][0]
        return predictions

    def score(self, features, labels):
        """Calculate the score of the prediction.

        Parameters
        ----------
        features : ndarray, shape (n_samples, n_features)
            Data to score on.
        labels : ndarray, shape (n_samples,)
            Target values.

        Returns
        ----------
        accuracy : float
            Accuracy of the model computed for the (features, labels) pairs.
        """

        predictions = self.predict(features)
        correct_predictions = 0
        for idx in range(features.shape[0]):
            if labels[idx] == predictions[idx]:
                correct_predictions += 1
        return correct_predictions / features.shape[0]


class MonthlySplit(BaseCrossValidator):

    """CrossValidator based on monthly split.

    Split data based on the given `time_column` (or default to index).
    Each split corresponds to one month of data for the training
    and the next month of data for the test.

    Parameters
    ----------
    time_column : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_column` to `'index'`.
    """

    def __init__(self, time_column="index"):  # noqa: D107
        self.time_column = time_column

    def get_n_splits(
        self,
        data,
        labels=None,
        groups=None,
    ):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        labels : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        num_splits : int
            The number of splits.
        """

        if self.time_column == "index":
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("datetime")
            sorted_data = data.sort_index()
            months = sorted_data.index.month
        else:

            if not pdtypes.is_datetime64_dtype(data[self.time_column]):
                raise ValueError("datetime")
            sorted_data = data.sort_values(by=self.time_column)
            sorted_data.index = sorted_data[self.time_column]
            months = sorted_data.index.month

        num_splits = 0
        for idx in range(1, len(months)):
            if months[idx] != months[idx - 1]:
                num_splits += 1
        return num_splits

    def split(
        self,
        data,
        labels,
        groups=None,
    ):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        labels : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train_indices : ndarray
            The training set indices for that split.
        test_indices : ndarray
            The testing set indices for that split.
        """

        num_splits = self.get_n_splits(data, labels, groups)

        if self.time_column == "index":
            months_list = [sorted(data.index)[0]]
        else:

            months_list = [sorted(data["date"])[0]]

        for _ in range(num_splits):
            months_list += [months_list[-1] + pd.DateOffset(months=1)]

        for split_idx in range(num_splits):
            train_month = months_list[split_idx]
            test_month = months_list[split_idx + 1]
            train_indices = []
            test_indices = []

            for data_idx in range(len(data)):
                if self.time_column == "index":
                    current_date = data.index[data_idx]
                else:
                    current_date = data.iloc[data_idx]["date"]

                if (
                    current_date.month == train_month.month
                    and current_date.year == train_month.year
                ):
                    train_indices.append(data_idx)
                elif (
                    current_date.month == test_month.month
                    and current_date.year == test_month.year
                ):

                    test_indices.append(data_idx)

            yield (train_indices, test_indices)
