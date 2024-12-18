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
from sklearn.utils.validation import validate_data
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
        # Input validation
        check_array(X)

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        check_classification_targets(y)

        # Fitted attributes
        self.X_ = X
        self.y_ = y
        self.classes_ = sorted(list(set(y)))
        # self.n_features_in_ = X.shape[1]
        X, y = validate_data(self, X, y)

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
        # Check if the estimator has been already fitted
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        X = validate_data(self, X, reset=False)

        # Distance matrix
        distances_array = pairwise_distances(X, self.X_, metric="euclidean")

        # Prediction Vector
        y_pred = np.zeros(X.shape[0], dtype=self.y_.dtype)

        # Iterate over the distance matrix for each obs
        for index_vector, distances_vector in enumerate(distances_array):

            # Save the k-nearest points index
            nearests = np.argsort(distances_vector)[: self.n_neighbors]

            # Get classes from min values
            classes = self.y_[nearests]

            # Save prediction for obs
            y_pred[index_vector] = max(set(classes), key=list(classes).count)

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
        # Call to predict method
        y_pred = self.predict(X)

        # Number of correct predictions
        correct_pred = 0
        # Number of observations
        obs = y.shape[0]

        for pred_index in range(obs):
            # If prediction is equal to observations
            if y_pred[pred_index] == y[pred_index]:
                correct_pred += 1
        return correct_pred / obs


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
        # Column to use for time split
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
        # Use index for time split
        if self.time_col == "index":
            most_old_period = pd.Period(X.index.values.min(), freq="M")
            most_recent_period = pd.Period(X.index.values.max(), freq="M")
            month_delta = (most_recent_period - most_old_period).n
        # Use a table column for time split
        else:
            # Check for column data type
            if pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                # Look for older date
                most_old_period = pd.Period(X[self.time_col].min(), freq="M")
                # Look for most recent date
                most_recent_period = pd.Period(
                    X[self.time_col].max(), freq="M"
                )
                # Delta month
                month_delta = (most_recent_period - most_old_period).n
            # The column data type is not datetime64
            else:
                raise ValueError(
                    "The column provided is not of datetime64[ns] format"
                )
        n_splits = month_delta
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
        # Get the number of split
        n_splits = self.get_n_splits(X, y, groups)
        # Use index for split
        if self.time_col == "index":
            start_date = pd.Period(X.index.values.min(), freq="M")
            time_data = X.index.values
        # Use table column for split
        else:
            if pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                start_date = pd.Period(X[self.time_col].min(), freq="M")
                time_data = X[self.time_col]
        # Loop over each split
        for i in range(n_splits):
            train_start = start_date + i
            train_end = train_start + 1
            test_start = train_end
            test_end = test_start + 1
            idx_train = np.where(
                (time_data >= train_start.start_time)
                & (time_data < train_end.start_time)
            )[0]
            idx_test = np.where(
                (time_data >= test_start.start_time)
                & (time_data < test_end.start_time)
            )[0]
            yield (idx_train, idx_test)
