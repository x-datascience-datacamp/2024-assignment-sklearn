from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, validate_data
from sklearn.utils.multiclass import unique_labels
import numpy as np
import pandas as pd


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        X, y = validate_data(X, y, ensure_2d=True)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]  # Required for compatibility
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_", "y_", "classes_"])
        X = validate_data(X, ensure_2d=True, reset=False)

        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_ - x, axis=1)
            nearest_indices = np.argsort(distances)[: self.n_neighbors]
            nearest_labels = self.y_[nearest_indices]
            predictions.append(np.bincount(nearest_labels).argmax())
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MonthlySplit:
    def __init__(self, time_col="index"):
        self.time_col = time_col

    def get_n_splits(self, X, y=None):
        if self.time_col == "index":
            dates = X.index
        else:
            dates = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError("The time column must be of datetime type.")

        return dates.to_period("M").nunique()

    def split(self, X, y=None):
        if self.time_col == "index":
            dates = X.index
        else:
            if self.time_col not in X.columns:
                raise ValueError(f"{self.time_col} column not found in X")
            dates = X[self.time_col]

        if not pd.api.types.is_datetime64_any_dtype(dates):
            raise ValueError("The time column must be of datetime type.")

        if isinstance(dates, pd.Index):
            dates = pd.Series(dates, index=X.index)

        grouped = X.groupby(dates.dt.to_period("M"))

        for _, group in grouped:
            train_idx = X.index.difference(group.index).sort_values()
            test_idx = group.index.sort_values()
            yield train_idx, test_idx

    def __repr__(self):
        return f"MonthlySplit(time_col='{self.time_col}')"
