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

import pandas.api.types as pdtypes

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import unique_labels
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
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
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
        X = validate_data(self, X, reset=False)

        y_pred = np.full(X.shape[0], self.y_[0])
        for id in range(X.shape[0]):
            x = X[id]
            liste_y = []

            list_dis = np.sum((self.X_ - x) ** 2, axis=1)
            list_Id_min = np.argpartition(list_dis,
                                          self.n_neighbors)[:self.n_neighbors]
            for Id_min in list_Id_min:
                liste_y += [self.y_[Id_min]]

            y_pred[id] = Counter(liste_y).most_common(1)[0][0]
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
        Accu = 0
        for id in range(X.shape[0]):
            if y[id] == y_pred[id]:
                Accu += 1
        return Accu / X.shape[0]


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
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError('datetime')
            df_tri = X.sort_index()
            liste_mois = df_tri.index.month

        else:
            if not pdtypes.is_datetime64_dtype(X[self.time_col]):
                raise ValueError('datetime')
            df_tri = X.sort_values(by=self.time_col)
            df_tri.index = df_tri[self.time_col]
            liste_mois = df_tri.index.month

        n_splits = 0
        for id in range(1, len(liste_mois)):
            if liste_mois[id] != liste_mois[id - 1]:
                n_splits += 1
        return n_splits

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
        n_splits = self.get_n_splits(X, y, groups)

        if self.time_col == 'index':
            liste_mois = [sorted(X.index)[0]]

        else:
            liste_mois = [sorted(X['date'])[0]]

        for mois in range(n_splits):
            liste_mois += [liste_mois[-1] + pd.DateOffset(months=1)]

        for split in range(n_splits):
            mois_train = liste_mois[split]
            mois_test = liste_mois[split + 1]
            idx_train = []
            idx_test = []

            for Idx in range(len(X)):
                if self.time_col == 'index':
                    date = X.index[Idx]
                else:
                    date = X.iloc[Idx]['date']

                if (date.month == mois_train.month and
                        date.year == mois_train.year):
                    idx_train.append(Idx)

                elif (date.month == mois_test.month and
                      date.year == mois_test.year):
                    idx_test.append(Idx)

            yield (idx_train, idx_test)
