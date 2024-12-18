"""Assignment - making a sklearn estimator and CV splitter.

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
The data to split should contain the index or one column in datetime format.
Then the aim is to split the data between train and test sets when for each
pair of successive months, we learn on the first and predict on the following.
For example, if you have data distributed from November 2020 to March 2021,
you have 4 splits. The first split will allow learning on November data and
predicting on December data, the second split to learn December and predict on
January, etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there are no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also call
at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import pairwise_distances
from pandas.api.types import is_datetime64_any_dtype


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1):  # noqa: D107
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the model."""
        X, y = validate_data(self, X, y)
        self.classes_ = unique_labels(y)
        self.label_encoders_ = LabelEncoder()
        self.X_ = X
        self.y_ = self.label_encoders_.fit_transform(y)
        return self

    def predict(self, X):
        """Predict class labels."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        distances = pairwise_distances(X, self.X_)
        nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]
        nearest_labels = self.y_[nearest_indices]

        if self.n_neighbors == 1:
            y_pred = nearest_labels.ravel()
        else:
            y_pred = np.array(
                [np.bincount(labels).argmax() for labels in nearest_labels]
            )

        y_pred = self.label_encoders_.inverse_transform(y_pred)
        return y_pred

    def score(self, X, y):
        """Calculate accuracy score."""
        y = np.asarray(y)
        return accuracy_score(y, self.predict(X))


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split."""

    def __init__(self, time_col="index"):  # noqa: D107
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations."""
        X_df = pd.DataFrame(X)
        if self.time_col == "index":
            X_df = X_df.reset_index()

        if not is_datetime64_any_dtype(X_df[self.time_col]):
            raise ValueError("datetime")

        unique_month_year = (
            X_df[self.time_col].dt.to_period("M")
            .drop_duplicates()
            .sort_values()
            .dt.to_timestamp()
            .tolist()
        )
        return len(unique_month_year) - 1

    def split(self, X, y, groups=None):
        """Generate train/test indices for each split."""
        n_splits = self.get_n_splits(X, y, groups)
        X_df = pd.DataFrame(X).reset_index()

        unique_month_year = (
            X_df[self.time_col].dt.to_period("M")
            .drop_duplicates()
            .sort_values()
            .dt.to_timestamp()
            .tolist()
        )

        for i in range(n_splits):
            month_train = unique_month_year[i]
            month_test = unique_month_year[i + 1]
            train_period = X_df[
                self.time_col
                ].dt.to_period("M") == month_train.to_period("M")
            test_period = X_df[
                self.time_col
                ].dt.to_period("M") == month_test.to_period("M")
            idx_train = X_df[
                train_period
            ].index.tolist()
            idx_test = X_df[
                test_period
            ].index.tolist()

            yield idx_train, idx_test
