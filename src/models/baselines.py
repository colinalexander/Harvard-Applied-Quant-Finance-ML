import warnings

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.linear_model import LinearRegression, Ridge


def fit_ols(X, y):
    m = LinearRegression()
    m.fit(X, y)
    return m


def fit_ridge(X, y, alpha=0.01):
    m = Ridge(alpha=alpha)
    m.fit(X, y)
    return m


def predict(m, X):
    return m.predict(X)


def spearman_ic(y_true, y_pred):
    """Calculate a Spearman correlation coefficient."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        s = spearmanr(a=y_true, b=y_pred)
        stat = s.statistic
    return stat
