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
from sklearn.metrics import euclidean_distances

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels


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
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        return self

    def predict(self, X):
        """
        Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
        Predicted class labels.
        """
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False)
        distances = euclidean_distances(X, self.X_)
        k_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        k_labels = self.y_[k_indices]

        y_pred = []
        for labels in k_labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            most_frequent_label = unique_labels[np.argmax(counts)]
            y_pred.append(most_frequent_label)
        return np.array(y_pred)

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

    def __init__(self, time_col="index"):  # noqa: D107
        self.time_col = time_col

    def _get_unique_months(self, X):
        if self.time_col == "index":
            time_column = X.index
        else:
            time_column = X[self.time_col]

        # Ensure the time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(time_column):
            raise ValueError(
                f"Time column {self.time_col} must be of datetime type")

        # Extract unique months
        dates = pd.to_datetime(X.index)
        unique_months = dates.to_period("M").unique()
        return unique_months

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator."""
        unique_months = self._get_unique_months(X)
        return len(unique_months) - 1

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set."""
        if self.time_col == "index":
            time_column = X.index
        else:
            time_column = X[self.time_col]

        # Ensure the time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(time_column):
            raise ValueError(
                f"Time column {self.time_col} must be of datetime type")

        dates = pd.to_datetime(X.index)
        months = dates.to_period("M")
        unique_months = months.unique()

        n_splits = self.get_n_splits(X, y, groups)

        for i in range(n_splits):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            idx_train = np.where(months == train_month)[0]
            idx_test = np.where(months == test_month)[0]

            yield (idx_train, idx_test)
