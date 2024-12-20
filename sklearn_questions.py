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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
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
        # X_checked = check_array(X)
        check_classification_targets(y)
        # # y_checked = check_array(y, ensure_2d=False)
        # X_checked, y_checked = check_X_y(X_checked, y)
        X_checked, y_checked = validate_data(self, X, y)
        self.X_ = X_checked
        self.y_ = y_checked
        self.classes_ = np.unique(y_checked)
        self.n_features_in_ = X_checked.shape[1]
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
        X = validate_data(self, X, reset=False)
        # X = check_array(X)
        # if X.shape[1] != self.n_features_in_:
        #     raise ValueError()
        y_pred = []
        for row in X:
            row_and_X = np.append([row], self.X_, axis=0)
            distances = pairwise_distances(row_and_X)[0][1:]
            closest_n = distances.argsort()[:self.n_neighbors]
            values, counts = np.unique(self.y_[closest_n], return_counts=True)
            y_pred.append(values[counts.argmax()])
        return np.array(y_pred, dtype=self.y_.dtype)

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
        X_checked = check_array(X)
        check_classification_targets(y)
        X_checked, y_checked = check_X_y(X_checked, y)

        y_pred = self.predict(X_checked)
        accuracy = np.mean(y_pred == y)

        return accuracy


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
        all_months = set()
        if self.time_col == "index":
            for d in X.index:
                if not isinstance(X.index[0], pd.Timestamp):
                    raise ValueError("Not a TimeStamp or datetime")
                if not (d.year, d.month) in all_months:
                    all_months.add((d.year, d.month))
        else:
            if not isinstance(X[self.time_col][0], pd.Timestamp):
                raise ValueError("Not a TimeStamp or datetime")
            for d in X[self.time_col]:
                if not (d.year, d.month) in all_months:
                    all_months.add((d.year, d.month))
        return len(all_months)-1

        # if no return encountered
        raise ValueError("No column contain a datetime")

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
        all_months = self.get_all_months(X, y, groups)
        n_splits = len(all_months)-1
        months = list(all_months)
        months.sort()
        for i in range(n_splits):
            if self.time_col == "index":
                if not isinstance(X.index[0], pd.Timestamp):
                    raise ValueError("Not a TimeStamp or datetime")
                idx_train = np.nonzero(np.logical_and(
                    X.index.year == months[i][0],
                    X.index.month == months[i][1]))[0]
                idx_test = np.nonzero(np.logical_and(
                    X.index.year == months[i+1][0],
                    X.index.month == months[i+1][1]))[0]
            else:
                if not isinstance(X[self.time_col][0], pd.Timestamp):
                    raise ValueError("Not a TimeStamp or datetime")
                idx_train = np.nonzero(np.logical_and(
                     X[self.time_col].dt.year == months[i][0],
                     X[self.time_col].dt.month == months[i][1]))[0]
                idx_test = np.nonzero(np.logical_and(
                     X[self.time_col].dt.year == months[i+1][0],
                     X[self.time_col].dt.month == months[i+1][1]))[0]
            yield (
                idx_train, idx_test
            )

    def get_all_months(self, X, y, groups=None):
        """Return all the months present in the data.

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
        n_splits : set
            set of all the months in the data.
        """
        all_months = set()
        if self.time_col == "index":
            if not isinstance(X.index[0], pd.Timestamp):
                raise ValueError("Not a TimeStamp or datetime")
            for d in X.index:
                if not (d.year, d.month) in all_months:
                    all_months.add((d.year, d.month))
        else:
            if not isinstance(X[self.time_col][0], pd.Timestamp):
                raise ValueError("Not a TimeStamp or datetime")
            for d in X[self.time_col]:
                if not (d.year, d.month) in all_months:
                    all_months.add((d.year, d.month))
        return all_months
