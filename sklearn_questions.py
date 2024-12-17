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
second split to learn december and predict

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `fon january etc.lake8`. You can check that there is no flake8
errors by calling `flake8` at the root of the repo.

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
from sklearn.metrics import accuracy_score
from collections import Counter


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
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
        X, y = check_X_y(X, y)
        self.X_train_ = np.array(X)
        self.y_train_ = np.array(y)
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
        check_is_fitted(self, ['X_train_', 'y_train_'])
        X = check_array(X)
        X = np.array(X)  # X is the test set ( or unseen data)
        y_pred = np.zeros(X.shape[0])
        N = X.shape[0]
        # Compute pairwise distances between test samples and training samples
        distances = pairwise_distances(X, self.X_train_)
        for n in range(N):
            # Get the indices of the k-nearest neighbors
            nearest_neighbors = np.argsort(distances[n])[:self.n_neighbors]
            k_nearest_labels = self.y_train[nearest_neighbors]

            # Majority voting by taking the most common label
            counts = Counter(k_nearest_labels).most_common(1)
            y_pred[n] = int(counts[0][0])  # Assign the most common label

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
        X = check_array(X)
        check_classification_targets(y)
        score = 0.0
        y_pred = self.predict(X)
        score = accuracy_score(y, y_pred)
        return score


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
        """Return the number of splitting iterations in the cross-validator."""
        # Check and convert time_col to datetime
        if self.time_col == 'index':
            time_data = pd.to_datetime(X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}' not found in X.")
            time_data = pd.to_datetime(X[self.time_col])

        # Sort data by time
        time_data = time_data.sort_values()

        # Calculate the total number of full months
        first_date = time_data.min()
        last_date = time_data.max()
        diff_month = last_date.month - first_date.month
        diff_year = last_date.year - first_date.year
        total_months = diff_year * 12 + diff_month

        return max(total_months, 0)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        # Ensure time column is in datetime format
        if self.time_col == 'index':
            time_data = pd.to_datetime(X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"Column '{self.time_col}' not found in X.")
            time_data = pd.to_datetime(X[self.time_col])

        # Sort data by time
        sorted_indices = time_data.argsort()
        X_sorted = X.iloc[sorted_indices]
        time_sorted = time_data.iloc[sorted_indices]

        # Generate month pairs
        unique_months = time_sorted.dt.to_period('M').unique()

        for i in range(len(unique_months) - 1):
            train_mask = time_sorted.dt.to_period('M') == unique_months[i]
            test_mask = time_sorted.dt.to_period('M') == unique_months[i + 1]

            train_indices = X_sorted.index[train_mask]
            test_indices = X_sorted.index[test_mask]

            yield train_indices, test_indices
