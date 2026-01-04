import pandas as pd
import numpy as np
import statsmodels.api as sm


class TECM:
    """
    Threshold error correction model.

    Parameters
    ----------
    longterm : list of str, optional
        List of variable names to be used in the long-term (cointegration) model.
        If None, all columns of X are used.
    shortterm : list of str, optional
        List of variable names to be used in the short-term (tecm) model.
        If None, all columns of X are used.
    theta : float, default None
        Threshold parameter for cointegration vector.
        If None, the parameter will be estimated using a grid search.

    Attributes
    ----------
    model : sm.OLS object
        An tecm model.
    longterm_model : sm.OLS object
        A model for the cointegration vector (longterm dependency).
    longterm : list of str
        List of variables used in the long-term model.
    shortterm : list of str
        List of variables used in the short-term model.
    thresholds_comparison : pandas.DataFrame
        Table containing the results of fitting the threshold parameter.

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
            shortterm = None,
            theta = None):
        self.model = None
        self.longterm = longterm
        self.shortterm = shortterm
        self.theta = theta


    def fit(self, X, y):
        """
        Fits both the tecm and the cointegraion models.

        Parameters
        ----------
        X : pandas.DataFrame
            Independent variables used in the long-term as well as the tecm model.        
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

        data['d_y'] = y.diff()
        diff = data[self.shortterm].diff()
        d_X_names = []
        for col in self.shortterm:
            data[f'd_{col}+'] = np.where(diff[col] >= 0, diff[col], 0)
            data[f'd_{col}-'] = np.where(diff[col] <  0, diff[col], 0)
            d_X_names.append(f'd_{col}+')
            d_X_names.append(f'd_{col}-')

        coint_vector = self.longterm_model.resid.shift(1)

        coint_quantiles = coint_vector.quantile([0.15, 0.85])
        thetas = coint_vector[
            (coint_vector > coint_quantiles.iloc[0]) & (coint_vector < coint_quantiles.iloc[1])
            ].to_numpy()
        theta_list = np.sort(thetas)

        self.thresholds_comparison = []
        for theta in theta_list:
            data[f'coint_vector+'] = np.where(coint_vector >= theta, coint_vector, 0)
            data[f'coint_vector-'] = np.where(coint_vector <  theta, coint_vector, 0)
            model = sm.OLS(data['d_y'][1:], data[['coint_vector+'] + ['coint_vector-'] + d_X_names][1:]).fit()
            self.thresholds_comparison.append(model.rsquared)

        self.thresholds_comparison = pd.DataFrame(
            {'rsquared':self.thresholds_comparison},
            index = theta_list)
        self.theta = self.thresholds_comparison['rsquared'].idxmax() if self.theta is None else self.theta

        data[f'coint_vector+'] = np.where(coint_vector >= self.theta, coint_vector, 0)
        data[f'coint_vector-'] = np.where(coint_vector <  self.theta, coint_vector, 0)

        self.model = sm.OLS(data['d_y'][1:], data[['coint_vector+'] + ['coint_vector-'] + d_X_names][1:]).fit()

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
        shortterm_params = pd.DataFrame(self.model.params).T

        predictions = []
        for i in range(0, len(X)):
            cointegration_vector = y_L - (self.longterm_model.params['const'] + self.longterm_model.params[self.longterm] @ X_L[self.longterm])

            d_X = pd.DataFrame(X.loc[:, self.shortterm].iloc[i] - X_L[self.shortterm]).T
            d_X_names = []
            for col in self.shortterm:
                d_X[f'd_{col}+'] = np.where(d_X[col] >= 0, d_X[col], 0)
                d_X[f'd_{col}-'] = np.where(d_X[col] <  0, d_X[col], 0)
                d_X_names.append(f'd_{col}+')
                d_X_names.append(f'd_{col}-')
            
            d_y = (
                d_X[d_X_names]@self.model.params[d_X_names]
                + (self.model.params['coint_vector+'] * np.where(cointegration_vector >= self.theta, cointegration_vector, 0))
                + (self.model.params['coint_vector-'] * np.where(cointegration_vector <  self.theta, cointegration_vector, 0))
            )

            y_hat = (y_L + d_y).iloc[0]
            predictions.append(y_hat)
            y_L = y_hat
            X_L = X.loc[:, all_params].iloc[i]

        return predictions