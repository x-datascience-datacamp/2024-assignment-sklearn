"""Assignment - making a sklearn estimator and CV splitter.

The goal of this assignment is to implement:

- A scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- A scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples correctly classified). You need to implement the `fit`,
`predict`, and `score` methods for this class. The code you write should pass
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
sets when, for each pair of successive months, we learn on the first and
predict on the following. For example, if you have data distributed from
November 2020 to March 2021, you have 4 splits. The first split
will allow you to learn on November data and predict on December data, the
second split to learn December and predict on January, etc.

We also ask you to respect the PEP 8 convention: https://pep8.org. This will be
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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
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
            The current instance of the classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Encode labels for consistency
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)

        self.X_train_ = X
        self.y_train_ = y_encoded
        self.classes_ = self.label_encoder_.classes_
        self.n_features_in_ = X.shape[1]
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
        X = check_array(X)

        # Predict class for each test sample
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = pairwise_distances(X[i].reshape(1, -1), self.X_train_)
            closest = np.argsort(distances)[0][:self.n_neighbors]
            y_pred[i] = np.argmax(np.bincount(self.y_train_[closest]))

        return self.label_encoder_.inverse_transform(y_pred)

    def score(self, X, y):
        """Calculate the score of the prediction.

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
        check_is_fitted(self)
        X = check_array(X)
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
        dates = self._get_dates(X)
        unique_months = pd.Series(dates).dt.to_period('M').unique()
        return max(0, len(unique_months) - 1)

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
        dates = self._get_dates(X)

        # Sort data by date
        sorted_indices = np.argsort(dates)
        X = X.iloc[sorted_indices]
        if y is not None:
            y = y.iloc[sorted_indices]

        # Generate unique months
        unique_months = sorted(pd.Series(dates).dt.to_period('M').unique())

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            # Get boolean masks for train and test months
            train_mask = pd.Series(dates).dt.to_period('M') == train_month
            test_mask = pd.Series(dates).dt.to_period('M') == test_month

            # Ensure training data ends before testing data starts
            train_end = dates[train_mask].max()
            test_start = dates[test_mask].min()

            if train_end >= test_start or not np.any(test_mask):
                continue  # Skip invalid splits

            # Find indices for train and test sets
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            yield train_idx, test_idx

    def _get_dates(self, X):
        """Retrieve datetime values from the specified column or index."""
        temp = X.reset_index()[self.time_col]

        if self.time_col == 'index':
            dates = pd.to_datetime(X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"Column '{self.time_col}' not found in input data."
                )
            if temp.dtype != 'datetime64[ns]':
                raise ValueError(
                    "Column provided is not in datetime format."
                )
            dates = pd.to_datetime(temp, errors='coerce')

        if dates.isna().any():
            raise ValueError(
                f"Column '{self.time_col}' must contain only datetime values."
            )
        return dates.values  # Return as a numpy array

    def __repr__(self):
        """Return a string representation of the MonthlySplit object."""
        return f"MonthlySplit(time_col='{self.time_col}')"
