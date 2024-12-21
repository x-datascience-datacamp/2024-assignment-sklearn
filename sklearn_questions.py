class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for predictions.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data stored during fit.

    y_ : ndarray of shape (n_samples,)
        Labels stored during fit.

    n_features_in_ : int
        Number of features in the training data.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fit the KNN classifier on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,)
            Target labels for training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Validate input
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if the classifier has been fitted
        check_is_fitted(self, ["X_", "y_"])

        # Validate input
        X = check_array(X)

        # Compute distances and predict
        distances = pairwise_distances(X, self.X_)
        nearest_indices = np.argmin(distances, axis=1)
        return self.y_[nearest_indices]

    def score(self, X, y):
        """
        Compute the accuracy of the classifier.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data.

        y : ndarray of shape (n_samples,)
            True labels for test data.

        Returns
        -------
        score : float
            Mean accuracy of predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
Updated MonthlySplit Class
python
Copier le code
from sklearn.model_selection import BaseCrossValidator
import numpy as np
import pandas as pd


class MonthlySplit(BaseCrossValidator):
    """
    Cross-validator that splits data based on months.

    Parameters
    ----------
    time_col : str, default='index'
        Column to use for date-based splitting. If 'index', the index of the
        DataFrame is used as the date column.

    Methods
    -------
    get_n_splits(X, y=None, groups=None)
        Return the number of splits.

    split(X, y=None, groups=None)
        Generate indices for training and testing splits.

    Raises
    ------
    ValueError
        If the `time_col` is not found or not a datetime type.
    """

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def get_n_splits(self, X, y=None, groups=None):
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime information.

        y : None
            Ignored, exists for API compatibility.

        groups : None
            Ignored, exists for API compatibility.

        Returns
        -------
        n_splits : int
            Number of month-based splits.
        """
        time_data = self._get_time_data(X)
        return len(time_data.dt.to_period("M").unique()) - 1

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : DataFrame
            Input data with datetime information.

        y : None
            Ignored, exists for API compatibility.

        groups : None
            Ignored, exists for API compatibility.

        Yields
        ------
        train_indices : ndarray
            Indices for training data.

        test_indices : ndarray
            Indices for testing data.
        """
        time_data = self._get_time_data(X)
        months = time_data.dt.to_period("M").unique()

        for i in range(len(months) - 1):
            train_mask = time_data.dt.to_period("M") == months[i]
            test_mask = time_data.dt.to_period("M") == months[i + 1]

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def _get_time_data(self, X):
        """
        Extract the datetime data from the specified column or index.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        time_data : Series
            Series of datetime values.

        Raises
        ------
        ValueError
            If the column is not found or is not datetime-like.
        """
        if self.time_col == 'index':
            if not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex.")
            return X.index
        elif self.time_col in X.columns:
            time_data = X[self.time_col]
            if not np.issubdtype(time_data.dtype, np.datetime64):
                raise ValueError(f"Column {self.time_col} must be of datetime type.")
            return time_data
        else:
            raise ValueError(f"Column {self.time_col} not found in input data.")
