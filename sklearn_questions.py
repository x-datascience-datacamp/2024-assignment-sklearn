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
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    A simple nearest-neighbor classifier that assigns the class of the
    closest training sample(s) to each test sample.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model using the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._validate_data(X, y, ensure_2d=True, dtype='numeric')
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X):
        """Predict class labels for the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["X_train_", "y_train_"])
        X = self._validate_data(
            X,
            reset=False,
            ensure_2d=True,
            dtype='numeric'
        )
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"features numbers({X.shape[1]}) doesn't match "
                f"training data ({self.X_train_.shape[1]})."
            )
        distances = pairwise_distances(X, self.X_train_)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        y_pred_indices = np.array([
            np.bincount(self.y_train_[indices]).argmax()
            for indices in nearest_indices
        ])

        return self.classes_[y_pred_indices]

    def score(self, X, y):
        """Compute the accuracy of the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Mean accuracy.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator that splits data by month.

    This splitter generates train/test indices based on a datetime column or
    the DataFrame's index. Each split is defined by consecutive months:
    - The train set includes one month's data.
    - The test set includes the following month's data.

    Parameters
    ----------
    time_col : str, default='index'
        Column name in `X` to use for monthly splitting.
        If 'index', use `X.index`.
    """

    def __init__(self, time_col="index"):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations."""
        time_series = self._get_time_series(X)
        unique_months = time_series.to_period("M").unique()
        unique_months = unique_months.sort_values()
        return max(0, len(unique_months) - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets."""
        time_series = self._get_time_series(X)
        unique_months = time_series.to_period("M").unique()
        unique_months = unique_months.sort_values()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            idx_train = np.where(time_series.to_period("M") == train_month)[0]
            idx_test = np.where(time_series.to_period("M") == test_month)[0]

            # Sort by actual time to ensure correct order
            idx_train = idx_train[np.argsort(time_series[idx_train])]
            idx_test = idx_test[np.argsort(time_series[idx_test])]

            yield idx_train, idx_test

    def _get_time_series(self, X):
        """Extract the time series from X based on `time_col`."""
        if isinstance(X, pd.DataFrame):
            if self.time_col == "index":
                time_series = X.index
            else:
                if self.time_col not in X.columns:
                    raise ValueError(
                        f"Column '{self.time_col}'isn't present in DataFrame."
                    )
                time_series = X[self.time_col]
        elif isinstance(X, pd.Series):
            if self.time_col != "index":
                raise ValueError(
                    "When X is a Series, time_col must be 'index'."
                )
            time_series = X.index
        else:
            if self.time_col != "index":
                raise ValueError(
                    "When X isn't a Series, time_col must be 'index'."
                )
            time_series = X

        if not pd.api.types.is_datetime64_any_dtype(time_series):
            raise ValueError(
                f"Failed to convert time '{self.time_col}' to datetime: "
                "Not a datetime dtype."
            )

        if not isinstance(time_series, pd.DatetimeIndex):
            time_series = pd.DatetimeIndex(time_series)

        return time_series
