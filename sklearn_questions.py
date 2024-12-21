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
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for predictions.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data stored during fit.

    y_ : ndarray of shape (n_samples,)
        Labels stored during fit.

    n_features_in_ : int
        Number of features in the training data.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the KNN classifier on training data.

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
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict the class labels for the given data.

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
        check_is_fitted(self, ["X_", "y_"])

        # Validate input
        X = check_array(X)

        # Compute distances and predict
        distances = pairwise_distances(X, self.X_)
        nearest_indices = np.argmin(distances, axis=1)
        return self.y_[nearest_indices]

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        y : ndarray of shape (n_samples,)
            True labels for test data.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
Updated MonthlySplit Class
python
Copier le code
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator that splits data based on months.

    Parameters
    ----------
    time_col : str, default='index'
        Column to use for date-based splitting. If 'index', the index of the
        DataFrame is used as the date column.

    Methods
    -------
    get_n_splits(X, y=None, groups=None)
        Return the number of splits.

    split(X, y=None, groups=None)
        Generate indices for training and testing splits.

    Raises
    ------
    ValueError
        If the `time_col` is not found or not a datetime type.
    """

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime information.

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
        return len(time_data.dt.to_period("M").unique()) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime information.

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
        months = time_data.dt.to_period("M").unique()

        for i in range(len(months) - 1):
            train_mask = time_data.dt.to_period("M") == months[i]
            test_mask = time_data.dt.to_period("M") == months[i + 1]

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def _get_time_data(self, X):
        """
        Extract the datetime data from the specified column or index.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_data : Series
            Series of datetime values.

        Raises
        ------
        ValueError
            If the column is not found or is not datetime-like.
        """
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex.")
            return X.index
        elif self.time_col in X.columns:
            time_data = X[self.time_col]
            if not np.issubdtype(time_data.dtype, np.datetime64):
                raise ValueError(f"Column {self.time_col} must be of datetime type.")
            return time_data
        else:
            raise ValueError(f"Column {self.time_col} not found in input data.")
