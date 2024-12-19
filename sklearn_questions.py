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
    """KNearestNeighbors classifier for classification tasks."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the KNearestNeighbors model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : KNearestNeighbors
            The fitted instance of the classifier.
        """
        X, y = self._validate_data(X, y, accept_sparse=False, ensure_2d=True)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray of shape (n_test_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = self._validate_data(X, accept_sparse=False, reset=False)

        y_pred = []
        for x in X:
            distances = pairwise_distances(x.reshape(1, -1), self.X_)
            nearest_indices = np.argsort(
                distances, axis=1)[0][:self.n_neighbors]
            values, counts = np.unique(
                self.y_[nearest_indices], return_counts=True)
            y_pred.append(values[np.argmax(counts)])

        return np.array(y_pred)

    def score(self, X, y):
        """
        Compute the accuracy score of the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.
        y : ndarray of shape (n_samples,)
            True labels for the test data.

        Returns
        -------
        score : float
            Accuracy of the model on the test data.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = self._validate_data(X, accept_sparse=False, reset=False)
        y = self._validate_data(y, ensure_2d=False, reset=False)

        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator for monthly data splitting.

    Parameters
    ----------
    time_col : str, default='index'
        Column name to be used for splitting the data based on time. If set to
        'index', the index will be used.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Input data with a datetime column.
        y : Ignored.
        groups : Ignored.

        Returns
        -------
        n_splits : int
            Number of splits.
        """
        X_copy = X.reset_index() if self.time_col == 'index' else X.copy()

        if not pd.api.types.is_datetime64_any_dtype(X_copy[self.time_col]):
            raise ValueError(
                f"The column '{self.time_col}' is not a datetime.")

        unique_months = X_copy[self.time_col].dt.to_period('M').unique()
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Input data with a datetime column.
        y : Ignored.
        groups : Ignored.

        Yields
        ------
        idx_train : ndarray
            Training set indices for the split.
        idx_test : ndarray
            Testing set indices for the split.
        """
        X_copy = X.reset_index()
        n_splits = self.get_n_splits(X_copy, y, groups)
        X_grouped = (
            X_copy.sort_values(by=self.time_col)
            .groupby(pd.Grouper(key=self.time_col, freq="ME"))
        )
        idxs = [group.index for _, group in X_grouped]
        for i in range(n_splits):
            idx_train = list(idxs[i])
            idx_test = list(idxs[i+1])
            yield (idx_train, idx_test)
