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
number of samples correctly classified). You need to implement the `fit`,
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
datetime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict on the following. For example if you have data distributed from
november 2020 to march 2021, you have 4 splits. The first split
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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    A simple k-nearest neighbors classifier that finds the closest training
    samples to each query point and performs a majority vote to determine its
    class.
    """

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the KNearestNeighbors model according to given training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training samples.
        y : ndarray, shape (n_samples,)
            Target values (class labels).

        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator.
        """
        # Ensure data is 2D and that n_features_in_ is set
        X, y = self._validate_data(X, y, ensure_2d=True)
        check_classification_targets(y)
        self.X_train_ = X
        self.y_train_ = y
        self.classes_ = unique_labels(y)
        return self

    def predict(self, X):
        """Predict class labels for the provided data X.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray, shape (n_test_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['X_train_', 'y_train_'])
        # Validate X with reset=False to ensure compatibility with training set
        X = self._validate_data(X, reset=False, ensure_2d=True)

        # Compute distances: shape is (n_train_samples, n_test_samples)
        distances = pairwise_distances(self.X_train_, X, metric="euclidean")

        # Find the indices of the k-nearest neighbors for each test sample
        # Sorting along axis=0 since rows correspond to training samples
        # and columns correspond to test samples
        nearest_indices = np.argsort(distances, axis=0)[:self.n_neighbors, :]

        # Gather the labels of the nearest neighbors for each test point
        nearest_labels = self.y_train_[nearest_indices]

        # Perform majority vote across the k nearest neighbors
        y_pred = np.apply_along_axis(
            lambda labels: Counter(labels).most_common(1)[0][0],
            axis=0,
            arr=nearest_labels
        )
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.
        y : ndarray, shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        check_classification_targets(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator that splits data into train/test sets based on months.

    This cross-validator splits the data by months. Each split corresponds to
    learning on data from one month and predicting on data from the next month.

    For example, if data spans from November 2020 to March 2021, we get:
    - Train: November 2020, Test: December 2020
    - Train: December 2020, Test: January 2021
    - Train: January 2021, Test: February 2021
    - Train: February 2021, Test: March 2021

    Parameters
    ----------
    time_col : str, default='index'
        Column name to use for splitting by month. If 'index', the DataFrame's
        index is used. The column or index must be a datetime type.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def __repr__(self):
        """Return a string representation of the MonthlySplit instance."""
        return f"MonthlySplit(time_col='{self.time_col}')"

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), default=None
            Ignored.
        groups : array-like of shape (n_samples,), default=None
            Ignored.

        Returns
        -------
        n_splits : int
            Number of month-based splits.
        """
        time_index = self._get_time_index(X)
        unique_months = self._get_unique_months(time_index)
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set by month.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Input data.
        y : array-like of shape (n_samples,), default=None
            Ignored.
        groups : array-like of shape (n_samples,), default=None
            Ignored.

        Yields
        ------
        idx_train : ndarray
            Indices for the training set for that split.
        idx_test : ndarray
            Indices for the testing set for that split.
        """
        time_index = self._get_time_index(X)
        unique_months = self._get_unique_months(time_index)

        for i in range(len(unique_months) - 1):
            current_month = unique_months[i]
            next_month = unique_months[i + 1]

            # Training indices: all samples in current_month
            train_mask = ((time_index.year == current_month[0]) &
                          (time_index.month == current_month[1]))
            # Testing indices: all samples in next_month
            test_mask = ((time_index.year == next_month[0]) &
                         (time_index.month == next_month[1]))

            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]
            yield (idx_train, idx_test)

    def _get_time_index(self, X):
        """Retrieve the time index from the DataFrame.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_index : DatetimeIndex
            The datetime index (from a column or from the index).
        """
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("When using 'index', the DataFrame's index \
                                  must be a DatetimeIndex.")
            return X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} column not found in \
                                 DataFrame.")
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"{self.time_col} column must be a \
                                  datetime type.")
            return pd.DatetimeIndex(X[self.time_col])

    def _get_unique_months(self, time_index):
        """Get unique (year, month) tuples from the time_index in \
              ascending order.

        Parameters
        ----------
        time_index : DatetimeIndex
            Datetime index from which unique months are extracted.

        Returns
        -------
        unique_months : list of (year, month) tuples
            Unique months in chronological order.
        """
        years = time_index.year
        months = time_index.month
        unique_year_month = sorted(set(zip(years, months)))
        return unique_year_month
