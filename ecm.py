import pandas as pd
import numpy as np
import statsmodels.api as sm

class ECM:
    """
    Error correction model.

    Parameters
    ----------
    longterm : list of str, optional
        List of variable names to be used in the long-term (cointegration) model.
        If None, all columns of X are used.
    shortterm : list of str, optional
        List of variable names to be used in the short-term (ECM) model.
        If None, all columns of X are used.  

    Attributes
    ----------
    self.model : sm.OLS object
        An ecm model.
    self.longterm_model : sm.OLS object
        A model for the cointegration vector (longterm dependency).
    self.longterm : list of str
        List of variables used in the long-term model.
    self.shortterm : list of str
        List of variables used in the short-term model.

    Methods
    -------
    fit(X, y):
        Fits both the ecm and the cointegraion models.
    predict(X, X_start, y_start):
        Creates a forecast based on the provided dataset and starting values,
        with y_start being the first element of the prediction vector.
        The returned vector begins with y_start as its first element.
    """

    def __init__(
            self,
            longterm = None,
            shortterm = None):
        self.model = None
        self.longterm = longterm
        self.shortterm = shortterm

    def fit(self, X, y):
        """
        Fits both the ecm and the cointegraion models.

        Parameters
        ----------
        X : pandas.DataFrame
            Independent variables used in the long-term as well as the short-term model.        
        y : pandas.DataFrame
            Dependent variable.

        Returns
        -------
        None
        """

        self.longterm = X.columns.tolist() if self.longterm is None else self.longterm
        self.shortterm = X.columns.tolist() if self.shortterm is None else self.shortterm

        if X.isna().any().any() or y.isna().any():
            raise ValueError("Input data contains NaN values.")
        missing_columns = set(self.longterm + self.shortterm).difference(X.columns)
        if missing_columns:
            raise ValueError(f"Variables not found in X: {missing_columns}")

        data = X.copy()

        self.longterm_model = sm.OLS(y, sm.add_constant(data.loc[:,self.longterm])).fit()
        data['coint_vector'] = self.longterm_model.resid.shift(1)

        data['d_y'] = y.diff()
        d_x = [f'd_{col}' for col in self.shortterm]
        data[d_x] = data[self.shortterm].diff()

        self.model = sm.OLS(data['d_y'][1:], data[['coint_vector'] + d_x][1:]).fit()

        return self
    
    def predict(self, X, X_start, y_start):
        """
        Creates a forecast based on the provided dataset and starting values,
        with y_start being the first element of the prediction vector.

        Parameters
        ----------
        X : pandas.DataFrame
            Independent variables used in the long-term as well as the short-term model.
        X_start : pandas.Series
            Starting values of the independent variables used in the long-term as well as the short-term model.
        y_start : float
            Starting value of the dependent variable.

        Returns
        -------
        ndarray
            Array of predictions. The returned vector begins with y_start as its first element.
        """

        if X.isna().any().any() or X_start.isna().any() or np.isnan(y_start):
            raise ValueError("Input data contains NaN values.")

        all_params = list(set(self.shortterm + self.longterm))

        missing_columns_X = set(all_params).difference(X.columns)
        missing_columns_X_start = set(all_params).difference(X_start.index)
        if missing_columns_X:
            raise ValueError(f"X is missing required variables: {missing_columns_X}")
        if missing_columns_X_start:
            raise ValueError(f"X_start is missing required variables: {missing_columns_X_start}")
        
        X_L = X_start[all_params]
        y_L = y_start

        predictions = [y_start]
        for i in range(0, len(X)):
            cointegration_vector = y_L - (self.longterm_model.params['const'] + self.longterm_model.params[self.longterm] @ X_L[self.longterm])
            d_X = X.loc[:, self.shortterm].iloc[i] - X_L[self.shortterm]
            d_X.index = 'd_' + d_X.index.astype(str)
            d_y = (self.model.params[d_X.index]@d_X) + (self.model.params['coint_vector'] * cointegration_vector)
            y_hat = y_L + d_y
            predictions.append(y_hat)
            y_L = y_hat
            X_L = X.loc[:, all_params].iloc[i]

        return np.array(predictions)
