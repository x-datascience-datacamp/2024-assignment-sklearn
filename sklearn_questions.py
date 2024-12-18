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
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    A simple implementation of the K-Nearest Neighbors algorithm for
    classification tasks. It predicts the target of a test sample
    based on the majority label of its `k` closest training neighbors,
    measured with Euclidean distance.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the classifier with the number of neighbors.

        Parameters
        ----------
        n_neighbors : int, default=1
            The number of nearest neighbors to use for prediction.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data.
        y : ndarray, shape (n_samples,)
            Training labels.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The fitted classifier.
        """
        X, y = self._validate_data(X, y, accept_sparse=True,
                                   multi_output=False)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Predict the class labels for the input samples.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Test data to predict labels for.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test sample.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_'])
        X = self._validate_data(X, accept_sparse=False, reset=False)
        y_pred = []
        for x in X:
            distances = pairwise_distances(x.reshape(1, -1), self.X_)
            nearest_indices = np.argsort(distances,
                                         axis=1)[0][:self.n_neighbors]
            values, counts = np.unique(self.y_[nearest_indices],
                                       return_counts=True)
            y_pred.append(values[np.argmax(counts)])
        y_pred = np.array(y_pred)

        return y_pred

    def score(self, X, y):
        """Calculate the accuracy of the model on the test data.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Test data.
        y : ndarray, shape (n_samples,)
            True labels.

        Returns
        ----------
        score : float
            Accuracy of the classifier predictions.
        """
        # Check if fit has been called and validate inputs
        check_is_fitted(self, ['X_', 'y_'])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        y = self._validate_data(y, ensure_2d=False, reset=False)

        y_pred = self.predict(X)
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

    def __init__(self, time_col='index'):
        """Initialize the MonthlySplit object.

        Parameters
        ----------
        time_col : str, default='index'
            Column of the input DataFrame that will be used to split the data.

        Raises
        ------
        ValueError
            If the specified `time_col` is not present or is not
            of datetime type.

        Returns
        -------
        None
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of monthly splits.

        Parameters
        ----------
        X : DataFrame
            Input data, where rows correspond to samples.

        Returns
        -------
        n_splits : int
            The number of monthly splits.
        """
        dates = self._get_time_column(X)
        unique_months = dates.dt.to_period('M').unique()
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Input data, where rows correspond to samples.

        Yields
        ------
        idx_train : ndarray
            Indices for the training set.
        idx_test : ndarray
            Indices for the testing set.
        """
        dates = self._get_time_column(X)
        unique_months = dates.dt.to_period('M').unique()
        unique_months = sorted(unique_months)

        n_splits = self.get_n_splits(X, y, groups)

        for i in range(n_splits):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            idx_train = X.index[dates.dt.to_period('M') == train_month]
            idx_train = idx_train.to_list()
            idx_test = X.index[dates.dt.to_period('M') == test_month].to_list()

            idx_train_int = [X.index.get_loc(idx) for idx in idx_train]
            idx_test_int = [X.index.get_loc(idx) for idx in idx_test]

            yield np.array(idx_train_int), np.array(idx_test_int)

    def _get_time_column(self, X):
        """
        Extract the datetime column or index from the input DataFrame.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_column : Series
            The datetime column or index.

        Raises
        ------
        ValueError
            If the specified `time_col` is not present or is not of
            datetime type.
        """
        if self.time_col == 'index':
            if not pd.api.types.is_datetime64_any_dtype(X.index):
                raise ValueError("The index must be datetime.")
            return pd.Series(X.index, index=X.index)
        elif self.time_col in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"'{self.time_col}' must be datetime.")
            return X[self.time_col]
        else:
            raise ValueError(f"'{self.time_col}' does not exist.")
