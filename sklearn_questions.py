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
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
# check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_distances
from pandas.api.types import is_datetime64_any_dtype


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
        # Validates input arrays X and y, ensuring that:
        # - They have compatible shapes
        # - X has at least one sample
        # - Data is finite
        # X, y = check_X_y(X, y, ensure_min_samples=1, force_all_finite=True)
        X, y = self._validate_data(X, y, ensure_2d=True, force_all_finite=True,
                                   dtype='numeric')

        # Check that the target is suitable for classification tasks
        check_classification_targets(y)

        # Get unique class labels from y
        self.classes_ = unique_labels(y)

        # Convert class labels in y to integer indices for efficient
        # counting later
        # `np.searchsorted` finds indices at which elements of y
        # should be inserted in self.classes_
        # to maintain order. This effectively maps classes to [0, 1, 2, ...].
        self.y_indices_ = np.searchsorted(self.classes_, y)

        # Store training features
        self.X_ = X

        # A required attribute: n_features_in_ denotes how
        # many features are in the input
        # self.n_features_in_ = X.shape[1]

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
        # Check if the estimator has been fitted before prediction
        check_is_fitted(self, ["X_", "y_indices_", "classes_"])

        # Validate the input data for prediction
        # This ensures X has at least one sample and is finite
        # X = check_array(X, ensure_min_samples=1, force_all_finite=True)
        X = self._validate_data(X, reset=False, ensure_2d=True,
                                force_all_finite=True,
                                dtype='numeric')

        # Compute pairwise distances between the input samples
        # (X) and the training samples (self.X_)
        # Here, we use Euclidean distance as a metric.
        dist = pairwise_distances(X, self.X_, metric='euclidean')

        # For each sample in X, find the indices of the
        # k nearest neighbors in the training set
        neighbors_idx = np.argsort(dist, axis=1)[:, :self.n_neighbors]

        # Retrieve the class indices of these k nearest neighbors
        neighbors_labels = self.y_indices_[neighbors_idx]

        # Perform majority voting:
        # For each sample, determine which class index appears
        # most frequently among its neighbors
        pred_indices = [np.argmax(np.bincount(r)) for r in neighbors_labels]
        pred_indices = np.array(pred_indices)

        # Map the predicted class indices back to the original class labels
        predictions = self.classes_[pred_indices]

        return predictions

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
        # Validate input
        X, y = check_X_y(X, y)
        # Predict labels
        y_pred = self.predict(X)
        # Calculate accuracy
        accuracy = np.mean(y_pred == y)

        return accuracy


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

    def _get_time_series(self, X):
        if self.time_col == 'index':
            # 检查 index 是否为 datetime 类型
            if not is_datetime64_any_dtype(X.index):
                raise ValueError("time_col must be datetime")
            time_series = X.index
        else:
            # 检查指定列是否为 datetime 类型
            if self.time_col not in X.columns:
                raise ValueError(f"Column {self.time_col} does not exist in X")
            if not is_datetime64_any_dtype(X[self.time_col]):
                raise ValueError("time_col must be datetime")
            time_series = X[self.time_col]
        return time_series

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
        """Return the number of splitting iterations in the cross-validator."""
        time_series = self._get_time_series(X)
        # 提取 (year, month) 元组
        year_month = [(d.year, d.month) for d in time_series]
        unique_months = sorted(set(year_month))
        # 分割次数 = unique_months 数量 - 1
        return max(len(unique_months) - 1, 0)

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
        time_series = self._get_time_series(X)
        year_month_list = [(d.year, d.month) for d in time_series]
        unique_months = sorted(set(year_month_list))

        # 根据 unique_months 连续拆分
        for i in range(len(unique_months) - 1):
            train_month = unique_months[i]
            test_month = unique_months[i + 1]

            train_mask = [ym == train_month for ym in year_month_list]
            test_mask = [ym == test_month for ym in year_month_list]

            idx_train = np.where(train_mask)[0]
            idx_test = np.where(test_mask)[0]

            yield idx_train, idx_test
