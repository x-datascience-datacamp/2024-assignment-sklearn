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
number of samples corectly classified). You need to implement the fit,
predict and score methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo pytest test_sklearn_questions.py. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to fit and
predict are correct using the check_* functions imported in the file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass test_nearest_neighbor_check_estimator.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with flake8. You can check that there is no flake8 errors by
calling flake8 at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using pydocstyle that you can also
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

from sklearn.utils.validation import (
    check_X_y, check_is_fitted, validate_data
)
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    This classifier uses the k-nearest neighbors algorithm for classification.
    It predicts the class label of a test point based on the majority class
    among its nearest neighbors in the training dataset.
    Attributes
    ----------
    n_neighbors : int
        Number of neighbors to use for classification.
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data after the `fit` method is called.
    y_train_ : ndarray of shape (n_samples,)
        Target values corresponding to `X_train_`.
    n_features_in_ : int
        Number of features in the training data.
    classes_ : ndarray
        Unique class labels present in the training data.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the classifier with the number of neighbors."""
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier.

        Stores the training data and target labels for future predictions.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        Returns
        -------
        self : KNearestNeighbors
            Fitted estimator instance.
        """
        X, y = check_X_y(X, y)
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Predict the class labels for the given data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Test data to predict labels for.

        Returns
        -------
        y : ndarray of shape (n_test_samples,)
            Predicted class labels for each sample in `X`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        distances = pairwise_distances(X, self.X_train_)
        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        unique_classes, y_indices = np.unique(
            self.y_train_, return_inverse=True)
        neighbor_labels = y_indices[nearest_neighbors]
        y_pred = np.array([
            unique_classes[np.bincount(labels).argmax()]
            for labels in neighbor_labels
        ])
        return y_pred

    def score(self, X, y):
        """Compute the accuracy score for the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.
        y : ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        score : float
            Proportion of correctly classified samples.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Custom cross-validator for splitting data based on months.

    Splits the data into training and test sets such that for each split,
    training data corresponds to one month and testing data to the next month.

    Attributes
    ----------
    time_col : str
    Name of the column in the input DataFrame that contains datetime
    information for splitting. If 'index', uses the DataFrame index.
    """

    def __init__(self, time_col='index'):
        """Initialize the cross-validator.

        Parameters
        ----------
        time_col : str, default='index'
            Column or index to use for datetime-based splitting.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Determine the number of splits.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for compatibility.
        groups : Ignored
            Not used, present for compatibility.

        Returns
        -------
        n_splits : int
            Number of month-based splits.
        """
        if self.time_col == 'index':
            dates = pd.Series(X.index)
        else:
            dates = X[self.time_col]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError(
                "The specified column is not in datetime format."
            )
        unique_months = dates.dt.to_period("M").unique()
        return len(unique_months) - 1

    def split(self, X, y, groups=None):
        """Generate indices for splitting data into train and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        y : Ignored
            Not used, present for compatibility.
        groups : Ignored
            Not used, present for compatibility.

        Yields
        ------
        idx_train : ndarray
            Indices of training samples for the split.
        idx_test : ndarray
            Indices of testing samples for the split.
        """
        n_splits = self.get_n_splits(X, y, groups)
        X_copy = X.copy()

        if self.time_col == 'index':
            dates_copy = pd.Series(X_copy.index)
        else:
            dates_copy = X_copy[self.time_col]

        if self.time_col == 'index':
            X_sorted = X.sort_index()
        else:
            X_sorted = X.sort_values(by=self.time_col)

        if self.time_col == 'index':
            dates_sorted = pd.Series(X_sorted.index)
        else:
            dates_sorted = X_sorted[self.time_col]

        unique_months = dates_sorted.dt.to_period("M").unique()

        for i in range(n_splits):
            if not pd.api.types.is_datetime64_any_dtype(dates_sorted):
                raise ValueError(
                    "The specified column is not in datetime format."
                )

            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_idx_sorted = dates_sorted[
                dates_sorted.dt.to_period("M") == train_month
            ].index
            test_idx_sorted = dates_sorted[
                dates_sorted.dt.to_period("M") == test_month
            ].index

            idx_train = np.where(
                dates_copy.isin(dates_sorted[train_idx_sorted])
            )[0]
            idx_test = np.where(
                dates_copy.isin(dates_sorted[test_idx_sorted])
            )[0]

            yield idx_train, idx_test
