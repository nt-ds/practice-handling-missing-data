#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# In[2]:


def mean_abs_error(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(criterion="mae", random_state=123)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)

