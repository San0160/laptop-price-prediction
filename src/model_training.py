from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    model=LinearRegression()
    model.fit(X_train, y_train)
    
    return model

from sklearn.linear_model import Ridge

def train_ridge_regression(X_train, y_train, alpha=1.0):
    model=Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    return model
    