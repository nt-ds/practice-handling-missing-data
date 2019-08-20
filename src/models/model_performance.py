#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[2]:


def mean_abs_error(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=123)
    model.fit(X_train, y_train)
    training_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    mae_training = mean_absolute_error(y_train, training_preds)
    mae_test = mean_absolute_error(y_test, test_preds)
    print(f"Mean absolute error (training): {mae_training}")
    print(f"Mean absolute error (test):     {mae_test}")
    return mae_test

