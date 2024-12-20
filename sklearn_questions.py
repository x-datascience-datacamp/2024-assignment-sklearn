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
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):
        """Initialize the classifier with the specified number of neighbors."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the classifier using the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted classifier.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.X_ = X
        self.y_ = y
        self.classes_, self.y_mapped_ = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray, shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["X_", "y_mapped_", "n_features_in_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in input ({X.shape[1]}) does not match "
                f"number of features seen during fit ({self.n_features_in_})."
            )

        distances = pairwise_distances(X, self.X_)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_classes = self.y_mapped_[nearest_indices]
        y_pred_mapped = np.array([
            np.bincount(row, minlength=len(self.classes_)).argmax()
            for row in nearest_classes
        ])

        return self.classes_[y_pred_mapped]

    def score(self, X, y):
        """Calculate the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test samples.
        y : ndarray, shape (n_samples,)
            True class labels.

        Returns
        -------
        score : float
            Mean accuracy of the predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator based on monthly splits.

    Generates train-test splits where training data is from one month
    and test data is from the following month.
    """

    def __init__(self, time_col='index'):
        """Initialize the cross-validator.

        Parameters
        ----------
        time_col : str, optional
            Column name of the DataFrame to use for splitting.
            Defaults to 'index'.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Get the number of splits.

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
            Number of splits based on unique months.
        """
        time_index = self._get_time_index(X)
        unique_months = time_index.to_period('M').unique()
        return len(unique_months) - 1

    def _get_time_index(self, X):
        """Helper method to extract the time index."""
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex "
                                 + "for time_col='index'.")
            return X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}'"
                                 + " not found in DataFrame.")
            time_column = X[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(time_column):
                raise ValueError(f"Column '{self.time_col}' "
                                 + "must be of datetime type.")
            return pd.DatetimeIndex(time_column)

    def split(self, X, y=None, groups=None):
        """Generate train-test splits.

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
        train_indices : ndarray
            Indices for the training set.
        test_indices : ndarray
            Indices for the testing set.
        """
        time_index = self._get_time_index(X)
        months = time_index.to_period('M')
        unique_months = months.unique().sort_values()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_indices = np.where(months == train_month)[0]
            test_indices = np.where(months == test_month)[0]

            train_indices = np.sort(train_indices)
            test_indices = np.sort(test_indices)

            yield train_indices, test_indices
