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
from sklearn.utils.validation import validate_data
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
            The current instance of the classifier
        """
        check_classification_targets(y)
        X, y = validate_data(self, X, y, reset=False)
        self.X_ = X
        self.y_ = y
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
        check_is_fitted(self, ['X_', 'y_', 'classes_'])
        X = validate_data(self, X, reset=False)
        if len(self.classes_) == 1:
            return np.full(X.shape[0], self.classes_[0], dtype = int)
        y_pred = []
        for x in X:
            x = x.reshape(1, -1)
            distance = pairwise_distances(x, self.X_).flatten()
            nearest_index = np.argsort(distance)[:self.n_neighbors]
            nearest_label = self.y_[nearest_index]
            majority_vote = Counter(nearest_label).most_common(1)[0][0]
            y_pred.append(majority_vote)
        y_pred = np.array(y_pred)
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
        check_classification_targets(y)
        X, y = validate_data(self, X, y, reset=False)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


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

    def _get_time_data(self, X):
        """Extract and validate the time column or index."""
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise TypeError(f"Index of X must be a pandas DatetimeIndex,
                                 but got {type(X.index).__name__}")
            time_data = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"The specified time column '
                                {self.time_col}' is not in X.")
            elif not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(f"Expecting datetime data in column '{self.time_col}',
                                but got {X[self.time_col].dtype}")
            time_data = pd.to_datetime(X[self.time_col])  

        if not isinstance(time_data, pd.DatetimeIndex):
            time_data = pd.DatetimeIndex(time_data)

        if time_data.isna().any():
            raise ValueError(f"Invalid datetime values detected in column '{self.time_col}'.")

        return time_data

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        time_data = self._get_time_data(X)
        months = time_data.to_period('M').unique()
        return len(months) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        time_data = self._get_time_data(X)
        months = time_data.to_period('M').unique()s
        months = sorted(months)
        n_splits = self.get_n_splits(X)

        for i in range(n_splits):
            train_mask = time_data.to_period('M').isin([months[i]])
            test_mask = time_data.to_period('M').isin([months[i + 1]])
            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]
            yield idx_train, idx_test
