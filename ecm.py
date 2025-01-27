import pandas as pd
import numpy as np
import statsmodels.api as sm

class ECM:
    """
    Error correction model.

    Parameters
    ----------
    None

    Attributes
    ----------
    self.model (sm.OLS object): An ecm model.
    self.longterm_model_ (sm.OLS object): A model for the cointegration vector (longterm dependency).

    Methods
    -------
    fit(X, y):
        Fits both the ecm and the cointegraion models.

    predict(X, X_start, y_start):
        Creates a forecast based on the provided dataset and starting values, with y_start being the first element of the prediction vector.
    """
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        data = X.copy()
        self.longterm_model_ = sm.OLS(y, sm.add_constant(data)).fit()
        data['coint_vector'] = self.longterm_model_.resid.shift(1)

        data['d_y'] = y.diff()
        d_x = [f'd_{col}' for col in X]
        data[d_x] = data[X.columns].diff()
        data = sm.add_constant(data)

        self.model = sm.OLS(data['d_y'][1:], data[['const', 'coint_vector'] + d_x][1:]).fit()

        return self
    
    def predict(self, X, X_start, y_start):

        X_L = X_start[X_start.index.isin(self.longterm_model_.params.index[1:])]
        y_L = y_start

        predictions = [y_start]
        for i in range(0, len(X)):
            cointegration_vector = y_L - (self.longterm_model_.params['const'] + self.longterm_model_.params.iloc[1:]@X_L)
            d_X = X.loc[:, self.longterm_model_.params.index[1:]].iloc[i] - X_L
            d_X.index = 'd_' + d_X.index.astype(str)
            d_y = self.model.params['const'] + (self.model.params[d_X.index]@d_X) + (self.model.params['coint_vector'] * cointegration_vector)
            y_hat = y_L + d_y
            predictions.append(y_hat)
            y_L = y_hat
            X_L = X.loc[:, self.longterm_model_.params.index[1:]].iloc[i]

        return np.array(predictions)