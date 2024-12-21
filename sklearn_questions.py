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

from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors (KNN) classifier.

    This classifier predicts the class label of a sample based on the majority
    class among its K nearest neighbors in the training data.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for classification.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data stored during the fit method.

    y_ : ndarray of shape (n_samples,)
        Target labels stored during the fit method.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels identified during fitting.

    n_features_in_ : int
        Number of features in the training data.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the KNN classifier using the training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target labels for training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if the classifier has been fitted
        check_is_fitted(self, ["X_", "y_", "classes_"])

        # Validate input
        X = check_array(X)

        # Compute pairwise distances between test samples and training samples
        distances = pairwise_distances(X, self.X_)

        # Identify the indices of the K nearest neighbors
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Retrieve the corresponding labels of the nearest neighbors
        nearest_labels = self.y_[nearest_indices]

        # Determine the most common class label among the neighbors
        y_pred, _ = mode(nearest_labels, axis=1)
        y_pred = y_pred.ravel()

        return y_pred

    def score(self, X, y):
        """
        Compute the mean accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        y : ndarray of shape (n_samples,)
            True labels for test data.

        Returns
        -------
        score : float
            Mean accuracy of the classifier.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator that splits data based on monthly intervals.

    This splitter generates training and testing indices such that each split
    consists of data from one month for training and the subsequent month for testing.

    Parameters
    ----------
    time_col : str, default='index'
        Column name to use for date-based splitting. If set to 'index',
        the DataFrame's index is used and must be a DatetimeIndex.

    Methods
    -------
    get_n_splits(X, y=None, groups=None)
        Returns the number of splits.

    split(X, y=None, groups=None)
        Generates indices for training and testing splits.

    Raises
    ------
    ValueError
        If the specified `time_col` does not exist or is not of datetime type.
    """

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Determine the number of splits based on unique months in the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with datetime information.

        y : None
            Ignored, exists for API compatibility.

        groups : None
            Ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Number of month-based splits.
        """
        time_data = self._get_time_data(X)
        unique_months = time_data.dt.to_period("M").unique()
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and testing sets based on months.

        For each split, data from one month is used for training and the next month
        for testing.

        Parameters
        ----------
        X : pd.DataFrame
            The input data with datetime information.

        y : None
            Ignored, exists for API compatibility.

        groups : None
            Ignored, exists for API compatibility.

        Yields
        ------
        train_indices : ndarray
            Indices for training data.

        test_indices : ndarray
            Indices for testing data.
        """
        time_data = self._get_time_data(X)
        unique_months = time_data.dt.to_period("M").unique()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = time_data.dt.to_period("M") == train_month
            test_mask = time_data.dt.to_period("M") == test_month

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def _get_time_data(self, X):
        """
        Extract datetime data from the specified column or index.

        Parameters
        ----------
        X : pd.DataFrame
            The input data.

        Returns
        -------
        time_data : pd.Series
            Series of datetime values.

        Raises
        ------
        ValueError
            If the specified `time_col` does not exist or is not datetime-like.
        """
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("DataFrame index must be a DatetimeIndex when time_col is set to 'index'.")
            return X.index.to_series()
        elif self.time_col in X.columns:
            time_data = X[self.time_col]
            if not np.issubdtype(time_data.dtype, np.datetime64):
                raise ValueError(f"Column '{self.time_col}' must be of datetime type.")
            return time_data
        else:
            raise ValueError(f"Column '{self.time_col}' not found in input data.")
