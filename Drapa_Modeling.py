# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:51:04 2018

@author: mob3f
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:51:04 2018

@author: mob3f
"""
import pandas as pd
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


label_encoder = preprocessing.LabelEncoder()

df = pd.read_csv('C:/Users/mob3f/Documents/Python Scripts/Darpa/Data/Featurized/iPhone9.1_feat.csv')
print(df.head(20))
print(df.describe())
print(df.groupby('activity').size())
df['activity'].hist()

# Use only one feature
df_X = df[['mean_x','mean_m']]
# Split the data into training/testing sets
X_train, X_test= train_test_split(df_X,test_size=0.33, random_state=42)

# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, X_train['mean_m'])
# Make predictions using the testing set
y_pred = regr.predict(X_test)
len(y_pred)
# The coefficients
print('Coefficients: \n', regr.coef_)
print('Intercept: \n', regr.intercept_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(X_test['mean_m'], y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(X_test['mean_m'], y_pred))

# Plot outputs
plt.scatter(X_test['mean_x'], X_test['mean_m'], color = 'black')
plt.scatter(X_test['mean_x'], y_pred, color = 'blue', linewidth = 1)
plt.xticks(())
plt.yticks(())
plt.show()




input_classes = df.activity.unique()
label_encoder.fit(input_classes)
df.columns
print ("\nClass mapping:")
for i, item in enumerate(label_encoder.classes_):
    print (item, '-->', i)

