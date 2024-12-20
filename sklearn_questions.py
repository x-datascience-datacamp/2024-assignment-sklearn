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

from sklearn.utils.validation import check_X_y, check_is_fitted, validate_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    A k-nearest neighbors classifier that uses the specified
    number of neighbors to classify data based on training data and labels.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for classification.
    """

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
        self.X_train_ = X
        self.y_train_ = y
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
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
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        distances = pairwise_distances(X, self.X_train_)
        nearest_neighbors = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        unique_classes, y_indices = np.unique(self.y_train_,
                                              return_inverse=True)
        neighbor_labels = y_indices[nearest_neighbors]
        y_pred = np.array([unique_classes[np.bincount(labels).argmax()]
                           for labels in neighbor_labels])
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
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given time_col (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set time_col to 'index'.
    """

    def __init__(self, time_col='index'):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        # Ensure input is a DataFrame or Series for proper handling of
        # time columns
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("X should be a pandas DataFrame or Series.")

        # Extract the time column (or use index) and ensure it's datetime
        if self.time_col == 'index':
            time_data = X.index
            # Count unique months in the data
            unique_months = time_data.to_period('M').unique()
        else:
            time_data = X[self.time_col] if isinstance(X, pd.DataFrame) else X
            # Count unique months in the data
            unique_months = time_data.dt.to_period('M').unique()
        if not pd.api.types.is_datetime64_any_dtype(time_data):
            raise ValueError("Time column must be of datetime type.")
        return len(unique_months) - 1  # Each split uses a month and the next

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
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
        # Ensure input is a DataFrame or Series for proper handling of
        # time columns
        if not isinstance(X, (pd.DataFrame, pd.Series)):
            raise ValueError("X should be a pandas DataFrame or Series.")

        # Extract the time column (or use index) and ensure it's datetime
        if self.time_col == 'index':
            time_data = X.index
        else:
            time_data = X[self.time_col] if isinstance(X, pd.DataFrame) else X

        if not pd.api.types.is_datetime64_any_dtype(time_data):
            raise ValueError("Time column must be of datetime type.")
        if ((isinstance(X, pd.Series) or isinstance(X, pd.DataFrame))
           and self.time_col != 'index'):
            # Sort the data based on the time column
            sorted_indices = X.sort_values(self.time_col).index
        else:
            sorted_indices = X.sort_index().index
        time_data = pd.Series(time_data.values,
                              index=sorted_indices).sort_index()
        unique_months = time_data.dt.to_period('M').unique()
        unique_months = np.sort(unique_months)

        # Generate train-test splits
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]
            idx_train = np.where(time_data.dt.to_period('M') == train_month)[0]
            idx_test = np.where(time_data.dt.to_period('M') == test_month)[0]
            yield idx_train, idx_test
