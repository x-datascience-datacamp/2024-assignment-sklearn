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

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import validate_data, check_is_fitted
from collections import Counter


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
        X, y = validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            multi_output=False
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        if len(self.classes_) < 2:
            raise ValueError("Only 1 class is present.")

        self.X_ = X
        self.y_ = y

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
        X = validate_data(self, X, accept_sparse=True, reset=False)

        y_pred = np.empty(shape=0, dtype=np.int64)

        for x_pred in X:
            k = self.n_neighbors
            arr_idx_nearest = np.empty(shape=0, dtype=np.int64)
            while k > 0:
                idx_nearest = None
                for (i, x) in enumerate(self.X_):
                    if i in arr_idx_nearest:
                        continue
                    if idx_nearest is None:
                        idx_nearest = i
                        continue
                    d_x = np.linalg.norm(x - x_pred)
                    d_nearest = np.linalg.norm(self.X_[idx_nearest] - x_pred)
                    if d_x < d_nearest:
                        idx_nearest = i

                if idx_nearest is not None:
                    arr_idx_nearest = np.append(arr_idx_nearest, idx_nearest)

                k -= 1

            y_all_pred = self.y_[arr_idx_nearest]
            most_common = Counter(y_all_pred).most_common(1)[0][0]
            y_pred = np.append(y_pred, most_common)

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
        return (y == y_pred).sum()/y.size


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

    def _get_time_col(self, X):
        """
        Extracts the time column from the given DataFrame and validates
        its datatype.

        Parameters
        ----------
        X : pandas.DataFrame
            The input DataFrame from which the time column is extracted.
            The DataFrame may have the time column as part of its index or as
            a regular column.

        Returns
        -------
        pandas.Series
            The extracted time column as a pandas Series.

        Raises
        ------
        ValueError
            If the extracted time column is not of type `datetime64`.

        Notes
        -----
        - The function first resets the index of the DataFrame to ensure
        the time column is accessible, whether it is an index or a column.
        - The column specified by `self.time_col` is then validated to
        confirm it is of a datetime type.
        """
        time_col = X.reset_index()[self.time_col]

        if not np.issubdtype(time_col.dtype, np.datetime64):
            raise ValueError('Error with datetime column or index.')

        return time_col

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

        time_column = self._get_time_col(X)

        return (len(time_column.dt.to_period('M').unique()) - 1)

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

        time_column = self._get_time_col(X)

        time_column_month = time_column.dt.to_period('M')
        cat = sorted(time_column_month.unique())
        n_splits = self.get_n_splits(X, y, groups)

        for i in range(0, n_splits):
            cat_train = cat[i]
            cat_test = cat[i + 1]

            idx_train = time_column_month[time_column_month == cat_train].index
            idx_test = time_column_month[time_column_month == cat_test].index

            yield (
                idx_train.to_list(), idx_test.to_list()
            )
