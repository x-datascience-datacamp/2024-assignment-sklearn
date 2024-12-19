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

from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances

from sklearn.metrics import accuracy_score
from sklearn.utils.validation import validate_data
from sklearn.preprocessing import LabelEncoder


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
        X, y = validate_data(self, X, y, ensure_2d=True, dtype=np.float64)
        check_classification_targets(y)
        self.X_train_ = X
        self.label_encoder_ = LabelEncoder()
        self.y_train_ = self.label_encoder_.fit_transform(y)

        # Save the unique class labels in the classes_ attribute
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
        # Check if the model has been fitted
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, reset=False,
                          ensure_2d=True, dtype=np.float64)

        # Compute distances between X_test and X_train
        distances = pairwise_distances(X, self.X_train_)

        # Find the indices of the nearest neighbors
        neighbors_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Predict labels based on majority voting
        y_pred = np.array([np.bincount(self.y_train_[indices]).argmax()
                           for indices in neighbors_indices])

        return self.label_encoder_.inverse_transform(y_pred)

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
        # Get predictions
        X, y = check_X_y(X, y, estimator=self)
        y_pred = self.predict(X)

        # Calculate accuracy
        return accuracy_score(y, y_pred)


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
        time_data = self._extract_time_data(X)
        unique_months = pd.Series(time_data).dt.to_period('M').unique()
        return max(len(unique_months) - 1, 0)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set."""
        time_data = self._extract_time_data(X)
        unique_months = sorted(pd.Series(time_data).dt.to_period('M').unique())

        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_idx = (time_data.dt.to_period('M')
                         == train_month).to_numpy().nonzero()[0]
            test_idx = (time_data.dt.to_period('M')
                        == test_month).to_numpy().nonzero()[0]

            yield train_idx.tolist(), test_idx.tolist()

    def _extract_time_data(self, X):
        """Extract datetime data from the specified column or index."""
        if self.time_col == 'index':
            if not isinstance(X.index, (pd.DatetimeIndex, pd.RangeIndex)):
                raise TypeError(
                    f"unsupported Type {type(X.index).__name__}")
            if isinstance(X.index, pd.RangeIndex):
                raise TypeError(
                    "Unsupported RangeIndex for 'index' as time_col.")
            return pd.Series(X.index, index=X.index)
        else:
            if self.time_col not in X.columns:
                raise ValueError(
                    f"Column {self.time_col} not found in DataFrame.")
            if not pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError(
                    f"The column {self.time_col} must be of type datetime.")
            return X[self.time_col]

    def __repr__(self):
        """Return a string representation of the class instance."""
        return f"MonthlySplit(time_col='{self.time_col}')"
