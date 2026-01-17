"""
model_evaluation.py

Evaluates regression models using standard metrics
"""

import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

def evaluate_regression (model, X_test, y_test):
    y_pred= model.predict(X_test)
    mae=mean_absolute_error(y_test, y_pred)
    rmse=np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }