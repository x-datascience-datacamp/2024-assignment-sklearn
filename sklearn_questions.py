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

from sklearn.utils.validation import check_is_fitted
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
        X, y = self._validate_data(X, y, accept_sparse=True,
                                   multi_output=False)
        check_classification_targets(y)
        self._X_train = X
        self._y_train = y
        self.classes_ = np.unique(y)
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
        check_is_fitted(self, ['_X_train', '_y_train'])
        X = self._validate_data(X, accept_sparse=True, reset=False)
        y_pred = np.zeros(X.shape[0], dtype=self._y_train.dtype)
        distances = pairwise_distances(X, self._X_train, metric='euclidean')
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self._y_train[nearest_indices]
        for i, labels in enumerate(nearest_labels):
            unique_labels, counts = np.unique(labels, return_counts=True)
            y_pred[i] = unique_labels[np.argmax(counts)]

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
        check_is_fitted(self, ['_X_train', '_y_train'])
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
        if not isinstance(X, type(pd.DataFrame())):
            x_df = pd.DataFrame({'date': X.index, 'val': X.values})
            x_df['date'] = pd.to_datetime(x_df['date'])
        elif self.time_col == 'index' and 'date' not in X.columns[0]:
            x_df = X.reset_index().copy()
            x_df = x_df.rename(columns={'index': 'date'}, inplace=False)
        else:
            x_df = X.copy()
            if 'date' not in x_df.columns[0]:
                x_df = x_df.rename({self.time_col: 'date'})
        month = pd.to_datetime(x_df['date']).dt.strftime('%b-%Y')
        return len(set(month)) - 1

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
        if self.time_col != 'index':
            if not isinstance(X[self.time_col].iloc[0],
                              type(pd.Timestamp('now'))):
                raise ValueError('datetime')
        else:
            if not isinstance(X.index[0], type(pd.Timestamp('now'))):
                raise ValueError('datetime')
        if not isinstance(X, type(pd.DataFrame())):
            x_df = pd.DataFrame({'date': X.index, 'val': X.values})
            x_df['date'] = pd.to_datetime(x_df['date'])
        elif self.time_col == 'index':
            x_df = X.reset_index().copy()
            x_df = x_df.rename(columns={'index': 'date'})
        else:
            x_df = X.copy()
            if 'date' not in x_df.columns[0]:
                x_df = x_df.rename(columns={self.time_col: 'date'},
                                   inplace=False)
        n_splits = self.get_n_splits(x_df, y, groups)
        x_df['month_year'] = pd.to_datetime(x_df['date']).dt.strftime('%b-%Y')
        # Spécifiez le format explicitement pour éviter les avertissements
        months_years = np.unique(np.sort(pd.to_datetime(x_df['month_year'],
                                                        format='%b-%Y')))
        x_df['month_year'] = pd.to_datetime(x_df['month_year'], format='%b-%Y')
        x_df = x_df.reset_index()
        for i in range(n_splits):
            idx_train = list(
                x_df[x_df['month_year'] == months_years[i]].index)
            idx_test = list(
                x_df[x_df['month_year'] == months_years[i + 1]].index)
            yield (idx_train, idx_test)
