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

from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors classifier.

    This classifier implements a simple K-Nearest Neighbors approach for
    classification. Given a query point, it finds the k nearest neighbors
    from the training set (using Euclidean distances) and performs a majority
    vote among them to determine the class.
    """

    def __init__(self, n_neighbors=1):
        """Initialize the KNearestNeighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=1
            Number of neighbors to use by default for kneighbors queries.
        """
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the K-Nearest Neighbors classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : ndarray of shape (n_samples,)
            Target values. Will be checked to ensure correct type.

        Returns
        -------
        self : KNearestNeighbors
            Returns the classifier itself.
        """
        # Validate input X and y
        X, y = validate_data(self, X, y, reset=False)
        check_classification_targets(y)

        # Check on n_neighbors
        if not isinstance(self.n_neighbors, int) or self.n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")

        # Store training data
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : ndarray of shape (n_test_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_test_samples,)
            Class labels for each data sample.
        """
        # Check if model has been trained
        check_is_fitted(self, ["X_", "y_"])

        # Validate input X and ensure it has the correct number of features
        X = validate_data(self, X, reset=False)

        # Calculate distances between test and training samples
        distances = pairwise_distances(X, self.X_)

        # Find indices of the k-nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        # Retrieve the neighbors' labels
        neighbor_labels = self.y_[neighbor_indices]

        # Majority vote among neighbors for each sample
        if self.n_neighbors == 1:
            y_pred = neighbor_labels.ravel()
        else:
            y_pred = []
            for lbls in neighbor_labels:
                classes, counts = np.unique(lbls, return_counts=True)
                y_pred.append(classes[np.argmax(counts)])
            y_pred = np.array(y_pred)
        return y_pred

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of the self.predict(X) wrt. y.
        """
        # Check if model has been trained
        check_is_fitted(self, ["X_", "y_", "classes_"])
        # Validate input X
        X = validate_data(self, X, reset=False)
        # Compare predicted labels with actual labels
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """Cross-validator that provides month-based splits.

    This cross-validator splits the data based on consecutive months. Given a
    time column or using the DataFrame's index if `time_col='index'`, it will
    group the data by month and create splits where each split trains on the
    data of one month and tests on the data of the next month.

    For example, if the data spans from November 2020 to March 2021, it will
    produce:
    - Train on November 2020, test on December 2020
    - Train on December 2020, test on January 2021
    - Train on January 2021, test on February 2021
    - Train on February 2021, test on March 2021

    Parameters
    ----------
    time_col : str, default='index'
        The column name to use for date-based splitting. If 'index', the
        DataFrame's index is used. This column/index must be of a datetime
        type.
    """

    def __init__(self, time_col="index"):
        """Initialize the monthly split cross-validator.

        Parameters
        ----------
        time_col : str, default='index'
            Column of the input DataFrame that will be used to split the data.
            If 'index', use the DataFrame index. Must be datetime type.
        """
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Data to be split. Must contain a datetime column specified by
            `self.time_col` or have a datetime index if
            `self.time_col='index'`.
        y : None
            Ignored, exists for compatibility.
        groups : None
            Ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Number of month-based splits.
        """
        # Extract and check the datetime series
        if self.time_col == "index":
            datetime_series = pd.Series(X.index, index=X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"'{self.time_col}' column not found in the DataFrame."
                )
            datetime_series = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            raise ValueError(
                f"Time column '{self.time_col}' must be of datetime type."
            )

        # Sort by datetime
        sorted_idx = np.argsort(datetime_series.values)
        sorted_months = datetime_series.iloc[sorted_idx].dt.to_period("M")
        unique_months = np.sort(sorted_months.unique())
        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set by month.

        For each split, the training set is the data from one month and the
        test set is the data from the following month.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            Data to be split.
        y : None
            Ignored, exists for compatibility.
        groups : None
            Ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray of shape (n_train_samples,)
            The training set indices for that split.
        idx_test : ndarray of shape (n_test_samples,)
            The testing set indices for that split.
        """
        # Extract and check the datetime series
        if self.time_col == "index":
            datetime_series = pd.Series(X.index, index=X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"'{self.time_col}' column not found in the DataFrame."
                )
            datetime_series = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(datetime_series):
            raise ValueError(
                f"Time column '{self.time_col}' must be of datetime type."
            )

        # Sort by datetime to ensure chronological order
        sorted_idx = np.argsort(datetime_series.values)
        datetime_sorted = datetime_series.iloc[sorted_idx]
        months = datetime_sorted.dt.to_period("M")
        unique_months = np.sort(months.unique())

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = months == train_month
            test_mask = months == test_month

            train_indices = sorted_idx[train_mask]
            test_indices = sorted_idx[test_mask]

            yield train_indices, test_indices
