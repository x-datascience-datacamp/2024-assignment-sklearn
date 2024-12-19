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

from sklearn.utils.validation import validate_data
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances

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
            The fitted instance of the classifier.

        Raises
        ----------
        ValueError
            If the input data is invalid or the labels are not suitable
            for classification.
        """
        # Verifying input (utilisation de validate_input à la place de
        # check_X_y())
        X, y = validate_data(
                            self, X, y,
                            accept_sparse=True,
                            ensure_2d=True,
                            multi_output=False)

        check_classification_targets(y)

        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Recover X and y for train set
        self._X_train = X
        self._y_train = y

        self._is_fitted = True

        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y_pred : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.

        Raises
        ----------
        ValueError
            If the number of feature in 'X' does not match the training data.
        NotFittedError
            If the model has not been fitted before calling "predict".
        """
        # Check if fit has been called
        check_is_fitted(self, "_is_fitted")

        # Validation of inputs
        X = validate_data(self, X, accept_sparse=True, reset=False)

        # Check for consistency between input number
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of features in X not match \
                            with the number of features in the training set.")

        # Calculating pairwise distances between test and train set
        distances = pairwise_distances(X, self._X_train)

        # Find index of the nearest neighbors and extract labels
        closest = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        neighbors_labels = self._y_train[closest]

        # Prediction
        y_pred = []
        for neighbors in neighbors_labels:
            counts = Counter(neighbors)
            most_common_label = counts.most_common(1)[0][0]
            y_pred.append(most_common_label)

        y_pred = np.array(y_pred)

        return y_pred

    def score(self, X, y):
        """Calculate the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            The accuracy of the model on the test data.
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly time-based split.

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

    Raises
    ----------
    ValueError
        If 'time_col' is not of datetime type when
        the split function is called.
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

        Raises
        ------
        ValueError
            If the `time_col` is not of datetime type.
        """
        if self.time_col == "index":
            X_reset = X.reset_index()
        else:
            X_reset = X.copy()

        if not pd.api.types.is_datetime64_any_dtype(X_reset[self.time_col]):
            raise ValueError("The time_col column must be datetime type")

        X_reset = X_reset.sort_values(by=self.time_col)
        time_col = X_reset[self.time_col]

        # Get months (uniques)
        months = time_col.dt.to_period("M").unique()
        n_split = len(months) - 1

        return n_split

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
        X_reset = X.reset_index()

        n_splits = self.get_n_splits(X_reset, y, groups)
        X_reset = X_reset.sort_values(by=self.time_col)

        months = X_reset[self.time_col].dt.to_period("M").unique()

        for i in range(n_splits):
            train_months = months[i]
            test_months = months[i+1]

            idx_train = X_reset[
                X_reset[self.time_col].dt.to_period('M') == train_months
            ].index.to_numpy()

            idx_test = X_reset[
                X_reset[self.time_col].dt.to_period('M') == test_months
            ].index.to_numpy()

            idx_train = np.sort(idx_train)
            idx_test = np.sort(idx_test)

            yield idx_train, idx_test
