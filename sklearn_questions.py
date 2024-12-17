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

from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
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
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        self.X_ = X
        self.y_ = y
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
        check_is_fitted(self, ["X_", "y_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {self.__class__.__name__} "
                f"is expecting {self.n_features_in_} features as input"
            )

        y_pred = []
        for x in X:
            distances = pairwise_distances(x.reshape(1, -1), self.X_)
            nearest_indices = np.argsort(distances,
                                         axis=1)[0][:self.n_neighbors]
            values, counts = np.unique(self.y_[nearest_indices],
                                       return_counts=True)
            y_pred.append(values[np.argmax(counts)])
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
        check_is_fitted(self)
        X = check_array(X)
        return np.mean(self.predict(X) == y)


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

    def __init__(self, time_col='index'):
        """Initialise l'objet.

        Parameters
        ----------
        time_col : string
            Pour le nom de la colonne de temps.
        """
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
        if self.time_col == 'index':
            time_col = pd.Series(X.index, name='time_col').reset_index(
                drop=True)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"{self.time_col} column not found in input data.")
            time_col = X[self.time_col].reset_index(drop=True)

        # Ensure the time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            raise ValueError(
                f"{self.time_col} must be a datetime column or index.")

        # Convert to datetime and find unique months
        time_col = pd.to_datetime(time_col)
        unique_months = time_col.dt.to_period('M').unique()

        return len(unique_months) - 1

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test sets by month.

        Each split corresponds to training on one month and testing on the
        following month.

        Parameters
        ----------
        X : pd.DataFrame or array-like of shape (n_samples, n_features)
            Input data. If array-like is provided, it must have a datetime
            index if time_col='index', or if time_col is another column name,
            be a DataFrame with that column in datetime format.
        y : array-like, optional
            Ignored.
        groups : array-like, optional
            Ignored.

        Yields
        ------
        idx_train : ndarray
            Training set indices for that split.
        idx_test : ndarray
            Testing set indices for that split.
        """
        if self.time_col == 'index':
            time_col = pd.Series(X.index, name='time_col').reset_index(
                drop=True)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"{self.time_col} column not found in input data.")
            time_col = X[self.time_col].reset_index(drop=True)

        # Ensure the time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(time_col):
            raise ValueError(
                f"{self.time_col} must be a datetime column or index.")

        # Sort the data
        if self.time_col == 'index':
            X_sorted = X.sort_index()
        else:
            X_sorted = X.sort_values(by=self.time_col)

        # Convert to periods
        time_col_sorted = pd.Series(
            X_sorted.index if self.time_col == 'index'
            else X_sorted[self.time_col]
        ).reset_index(drop=True)
        time_periods = pd.to_datetime(time_col_sorted).dt.to_period('M')
        unique_months = time_periods.unique()

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            # Masks for train and test
            train_mask = time_periods == train_month
            test_mask = time_periods == test_month

            # Retrieve positional indices
            idx_train = X_sorted.index[train_mask]
            idx_test = X_sorted.index[test_mask]

            yield X.index.get_indexer(idx_train), X.index.get_indexer(idx_test)
