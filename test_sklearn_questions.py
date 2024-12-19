# ##################################################
# YOU SHOULD NOT TOUCH THIS FILE !
# ##################################################
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier

from sklearn_questions import KNearestNeighbors
from sklearn_questions import MonthlySplit


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_one_nearest_neighbor_match_sklearn(k):
    X, y = make_classification(n_samples=200, n_features=20,
                               random_state=42)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=k)
    y_pred_sk = knn.fit(X_train, y_train).predict(X_test)

    onn = KNearestNeighbors(k)
    y_pred_me = onn.fit(X_train, y_train).predict(X_test)
    assert_array_equal(y_pred_me, y_pred_sk)

    assert onn.score(X_test, y_test) == knn.score(X_test, y_test)


@pytest.mark.parametrize("k", [1, 3, 5, 7])
def test_one_nearest_neighbor_check_estimator(k):
    check_estimator(KNearestNeighbors(n_neighbors=k))


@pytest.mark.parametrize("end_date, expected_splits",
                         [('2021-01-31', 12), ('2020-12-31', 11)])
@pytest.mark.parametrize("shuffle_data", [True, False])
def test_time_split(end_date, expected_splits, shuffle_data):

    date = pd.date_range(start='2020-01-01', end=end_date, freq='D')
    n_samples = len(date)
    X = pd.DataFrame(range(n_samples), index=date, columns=['val'])
    y = pd.DataFrame(
        np.array([i % 2 for i in range(n_samples)]),
        index=date
    )

    if shuffle_data:
        X, y = shuffle(X, y, random_state=0)

    X_1d = X['val']

    cv = MonthlySplit()
    cv_repr = "MonthlySplit(time_col='index')"

    # Test if the repr works without any errors
    assert cv_repr == repr(cv)

    # Test if get_n_splits works correctly
    assert cv.get_n_splits(X, y) == expected_splits

    # Test if the cross-validator works as expected even if
    # the data is 1d
    np.testing.assert_equal(
        list(cv.split(X, y)), list(cv.split(X_1d, y))
    )

    # Test that train, test indices returned are integers and
    # data is correctly ordered
    for train, test in cv.split(X, y):
        assert np.asarray(train).dtype.kind == "i"
        assert np.asarray(test).dtype.kind == "i"

        X_train, X_test = X.iloc[train], X.iloc[test]
        y_train, y_test = y.iloc[train], y.iloc[test]
        assert X_train.index.max() < X_test.index.min()
        assert y_train.index.max() < y_test.index.min()
        assert X.index.equals(y.index)

    with pytest.raises(ValueError, match='datetime'):
        cv = MonthlySplit(time_col='val')
        next(cv.split(X, y))


@pytest.mark.parametrize("end_date", ['2021-01-31', '2020-12-31'])
@pytest.mark.parametrize("shuffle_data", [True, False])
def test_time_split_on_column(end_date, shuffle_data):

    date = pd.date_range(
        start='2020-01-01 00:00', end=end_date, freq='D'
    )
    n_samples = len(date)
    X = pd.DataFrame({'val': range(n_samples), 'date': date})
    y = pd.DataFrame(
        np.array([i % 2 for i in range(n_samples)])
    )

    if shuffle_data:
        X, y = shuffle(X, y, random_state=0)

    cv = MonthlySplit(time_col='date')

    # Test that train, test indices returned are integers and
    # data is correctly ordered
    n_splits = 0
    last_time = None
    for train, test in cv.split(X, y):

        X_train, X_test = X.iloc[train], X.iloc[test]
        assert X_train['date'].max() < X_test['date'].min()
        assert X_train['date'].dt.month.nunique() == 1
        assert X_test['date'].dt.month.nunique() == 1
        assert X_train['date'].dt.year.nunique() == 1
        assert X_test['date'].dt.year.nunique() == 1
        if last_time is not None:
            assert X_test['date'].min() > last_time
        last_time = X_test['date'].max()
        n_splits += 1

    assert 'idx' not in X.columns

    assert n_splits == cv.get_n_splits(X, y)