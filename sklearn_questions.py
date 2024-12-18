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
number of samples correctly classified). You need to implement the `fit`,
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

from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances, accuracy_score

# from sklearn.utils.multiclass import check_classification_targets
# from sklearn.metrics.pairwise import pairwise_distances
from pandas.api.types import is_datetime64_any_dtype


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    # https://github.com/scikit-learn/scikit-learn/blob/15eb9f30c77ec8166a0135ca14b8de7fdfe15b91/sklearn/neighbors/_classification.py#L40
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
        # Check that X and y have correct shape, set n_features_in_, etc.
        X, y = validate_data(self, X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self._idx2classes = {
            classe: idx for idx, classe in enumerate(self.classes_)
        }
        self._classes2idx = {
            idx: classe for idx, classe in enumerate(self.classes_)
        }
        self.X_ = X
        self.y_ = np.array([self._classes2idx.get(label, 0) for label in y])

        # Handle the case where only one class is present
        if len(self.classes_) == 1:
            self.constant_prediction_ = self.classes_[0]
        else:
            self.constant_prediction_ = None

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
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = validate_data(self, X, reset=False)

        # If only one class was present during fit, return that class
        if self.constant_prediction_ is not None:
            return np.full(X.shape[0], self.constant_prediction_)

        y_pred = np.zeros(X.shape[0])
        dist_X_2_train = euclidean_distances(X, self.X_)
        nearest_neighbors_idx = np.argsort(dist_X_2_train, axis=-1)[
            :, : self.n_neighbors
        ]
        nearest_neighbors_pred_idx = self.y_[nearest_neighbors_idx]

        y_pred = np.empty(X.shape[0])
        for k, neighbors in enumerate(nearest_neighbors_pred_idx):
            sorted_neighbors, idx, counts = np.unique(
                neighbors, return_index=True, return_counts=True
            )
            mode_label = sorted_neighbors[np.argmax(counts)]
            y_pred[k] = self._idx2classes.get(mode_label, 0)

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Return the mean accuracy on the given test data and labels.
        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

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
        return accuracy_score(y, self.predict(X))


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
        # number of months in the dataframe - 1
        if self.time_col == "index":
            X_time_col = X.index
        else:
            X_time_col = X[self.time_col]
        start_date = X_time_col.min()
        end_date = X_time_col.max()
        n_splits = (
            pd.Period(end_date, freq="M") - pd.Period(start_date, freq="M")
        ).n
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
        if self.time_col == "index":
            X_time_col = X.index
        else:
            X_time_col = X[self.time_col]

        if not is_datetime64_any_dtype(X_time_col):
            raise ValueError(f"{self.time_col} is not of type datetime")
        n_samples = X.shape[0]
        n_splits = self.get_n_splits(X, y, groups)
        start_date = X_time_col.min().normalize().replace(day=1)
        for i in range(n_splits):
            # i is the index of the starting month
            train_start = start_date + pd.offsets.MonthBegin(i)
            train_end = start_date.replace(
                hour=23, minute=59, second=59
            ) + pd.offsets.MonthEnd(i + 1)
            test_start = start_date + pd.offsets.MonthBegin(i + 1)
            test_end = start_date.replace(
                hour=23, minute=59, second=59
            ) + pd.offsets.MonthEnd(i + 2)
            idx_train = np.arange(n_samples)[
                (X_time_col >= train_start) & (X_time_col <= train_end)
            ]
            idx_test = np.arange(n_samples)[
                (X_time_col >= test_start) & (X_time_col <= test_end)
            ]
            yield (idx_train, idx_test)
