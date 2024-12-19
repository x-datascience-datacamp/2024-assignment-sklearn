"""

Assignment - making a sklearn estimator and cv splitter.

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

from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """
    This class implements the KNN algorithm for classification.

    Given a set of training data, it predicts the labels of new samples
    by finding the most common labels among the nearest neighbors
    in the training data.

    Attributes:
    n_neighbors : int (default=1)
        The number of neighbors to use when making predictions.

    Methods:
    fit(X, y) :
        Fit the model to the training data and the corresponding labels.

    predict(X) :
        Predict the labels for the input data.

    score(X, y) :
        Compute the accuracy of the model on the test data and true labels.

    Raises:
    ValueError :
        - If the number of features in the input data
          for prediction does not match
          the number of features in the training data.
    """

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the model to the provided training data.

        It stores input and target features,
        prepares class labels and the number
        of features for predictions.

        Parameters:
        X : array-like, shape (n_samples, n_features)
        The input data to train the model on, where n_samples is the number of
        samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
        The target values (labels).

        Returns:
        self : object.
        """
        X, y = validate_data(self, X, y)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        """
        Predict class labels for input data.

        This method performs the prediction based on the
        trained model using the KNN algorithm.
        It calculates the Euclidean distance between
        the test data and the training data, finds the nearest
        neighbors, and assigns the most common label among them
        as the predicted class label.

        Parameters:
        X : array-like, shape (n_samples, n_features)
        The input data, where n_samples is the number of samples and
        n_features is the number of features.

        Returns:
        y_pred : array, shape (n_samples,)
        The predicted labels for each sample in the input data.

        Raises:
        ValueError : if the number of features in the input data does not match
                 the number of features in the training data.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = validate_data(self, X, reset=False)
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
             f"X has {X.shape[1]} features, but the training data has"
             f" {self.X_train_.shape[1]} features."
            )
        y_pred = np.empty(X.shape[0], dtype=self.y_train_.dtype)
        distances = pairwise_distances(X, self.X_train_, metric='euclidean')
        nearest_inds = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_train_[nearest_inds]
        for i, labels in enumerate(nearest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]

        return y_pred

    def score(self, X, y):
        """
        Compute the accuracy of the model on the provided test data.

        This function calculates the accuracy of the model
        by comparing the predicted labels with the true ones
        and computes the proportion of correct predictions.
        Parameters:
        X : array-like, shape (n_samples, n_features)
        The input data for which to compute the predictions, where n_samples is
        the number of samples and n_features is the number of features.
        y : array-like, shape (n_samples,)
        The true labels.

        Returns:
        score : float
        The accuracy score, which is the ratio of
        the number of correct predictions
        to the total number of predictions.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = check_array(X)
        y = check_array(y, ensure_2d=False)
        y_pred = self.predict(X)
        score = np.mean(y_pred == y)
        return score


class MonthlySplit(BaseCrossValidator):
    """
    This class is for splitting data into training and test sets.

    The split is based on distinct months.
    It implements a custom cross-validation for time series data.
    It performs time-based splitting of the input data using
    either the index or column that contains datetime values.
    The data is split into distinct months,
    where the training set corresponds
    to one month and the test set corresponds to the next.

    Attributes:
    time_col : str (default='index')
        The column or index name that contains datetime values.
        If set to 'index', the method assumes the datetime
        values are in the index of the input data. Otherwise,
        it assumes a specific column in the dataframe contains
        the datetime values.
    Methods:
    get_n_splits(X, y=None, groups=None) :
        Returns the number of available splits for cross-validation.
    split(X, y=None, groups=None) :
        Yields indexes for time-based cross-validation splits,
        using distinct months for training and testing.
    Raises:
    ValueError :
        - If the specified `time_col` is not found in the data.
        - If the `time_col` is not of datetime type.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Get the number of splits available in the time series data.

        This function determines the number of time-based
        splits that can be used for cross-validation.
        It extracts the time information from either the
        index or a specified time column in the input data,
        and calculates the number of distinct months for splitting.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data (It must contain either a datetime index or
            a column with datetime values),
            where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, optional
            The target labels (not used).
        groups : array-like, optional
            Group labels (not used but included for compatibility).
        Returns:
        n_splits : int
                   The number of time-based splits available in the data.
        Raises:
                ValueError : if the time column is not found in the data
                             or if the time column is not of datetime type.
        """
        if self.time_col == 'index':
            extracted_time = pd.Series(
                X.index, name='extracted_time'
                ).reset_index(drop=True)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"{self.time_col} column not found in the input")
            extracted_time = X[self.time_col].reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(extracted_time):
            raise ValueError(
                f"{self.time_col} must be a datetime")
        extracted_time = pd.to_datetime(extracted_time)
        distinct_months = extracted_time.dt.to_period('M').unique()

        return len(distinct_months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indexes for time series cross-validation splits.

        This function performs time-based splitting of the input data `X` into
        training and test sets, using distinct months as the time periods.
        It iterates through the unique periods in the time column or index,
        yielding indexes for training and testing datasets based on time order.

        Parameters:
        X : array-like, shape (n_samples, n_features)
        The input data, where n_samples is the number of samples and n_features
        is the number of features. It must contain a datetime index or a column
        with datetime values.
        y : array-like, optional
        The target labels (not used).
        groups : array-like, optional
        Group labels (not used).
        Yields:
        (train_indexes, test_indexes) : tuple of arrays
        The indexes for the training and test datasets for each split.
        The training set corresponds to a time period and the test set
        corresponds to the next time period.
        Raises:
        ValueError : if the specified time column is not found in the data or
                if the time column is not of datetime type.
        """
        if self.time_col == 'index':
            extracted_time = pd.Series(
                X.index, name='extracted_time'
                ).reset_index(drop=True)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"{self.time_col} column not found in input data")
            extracted_time = X[self.time_col].reset_index(drop=True)

        if not pd.api.types.is_datetime64_any_dtype(extracted_time):
            raise ValueError(
                f"{self.time_col} must be a datetime")

        if self.time_col == 'index':
            sorted_data = X.sort_index()
        else:
            sorted_data = X.sort_values(by=self.time_col)

        sorted_time = pd.Series(
            sorted_data.index if self.time_col == 'index'
            else sorted_data[self.time_col]
        ).reset_index(drop=True)
        periods = pd.to_datetime(sorted_time).dt.to_period('M')
        distinct_months = periods.unique()

        for idx in range(len(distinct_months) - 1):
            train_month = distinct_months[idx]
            test_month = distinct_months[idx + 1]

            train_condition = periods == train_month
            test_condition = periods == test_month

            train_positions = sorted_data.index[train_condition]
            test_positions = sorted_data.index[test_condition]

            yield (
                X.index.get_indexer(train_positions),
                X.index.get_indexer(test_positions))
