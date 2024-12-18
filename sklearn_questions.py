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
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Identify the unique classes and map them to integers
        self.classes_ = np.unique(y)
        class_to_int = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_int = np.array([class_to_int[val] for val in y], dtype=int)

        self.X_ = X
        self.y_int_ = y_int
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
        check_is_fitted(self, ["X_", "y_int_", "classes_"])
        X = check_array(X)

        distances = pairwise_distances(X, self.X_)
        partitions = np.argpartition(distances, self.n_neighbors, axis=1)
        neighbors_idx = partitions[:, :self.n_neighbors]
        n_labls = self.y_int_[neighbors_idx]
        y_pred_int = np.array([np.argmax(np.bincount(row)) for row in n_labls])
        y_pred = self.classes_[y_pred_int]
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

    def _check_time_column(self, X):
        """Check if the time column is valid and return it as a DatetimeIndex.

        Parameters
        ----------
        X : DataFrame or Series
            Input data.

        Returns
        -------
        time_values : DatetimeIndex
            The time values extracted from X.
        X : DataFrame
            The input data as a DataFrame.
        """
        if isinstance(X, pd.Series):
            X = X.to_frame()

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X should be a DataFrame or Series.")

        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("X must have a DatetimeIndex.")
            time_values = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}' not found in X.")
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError("Must be of a datetime type.")
            time_values = pd.DatetimeIndex(X[self.time_col])

        return time_values, X

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
        time_values, _ = self._check_time_column(X)
        sorted_idx = np.argsort(time_values)
        sorted_time_values = time_values[sorted_idx]
        unique_months = sorted_time_values.to_period('M').unique()
        return max(0, len(unique_months) - 1)

    def split(self, X, y=None, groups=None):
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
        time_values, X = self._check_time_column(X)

        df = X.copy()
        df['_orig_idx'] = np.arange(len(df))
        df['_time'] = time_values

        df = df.sort_values('_time')
        year_month = df['_time'].dt.to_period('M')
        unique_months = year_month.unique()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_idx = df.loc[year_month == train_month, '_orig_idx'].values
            test_idx = df.loc[year_month == test_month, '_orig_idx'].values

            yield train_idx, test_idx

        df.drop(columns=['_orig_idx', '_time'], inplace=True, errors='ignore')
