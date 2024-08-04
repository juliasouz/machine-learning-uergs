import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=pd.read_csv("kc_house_data.csv")
pd.set_option('display.max_columns',21)
pd.set_option('display.max_rows',21)

missing_data=data.isnull().sum()
missing_data

data.drop('id',axis=1,inplace=True)
data.drop('date',axis=1,inplace=True)
data.drop('zipcode',axis=1,inplace=True)
x = data.drop('price', axis=1)
y = data['price']
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.30, random_state=5)
model= LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

plt.scatter(y_test, y_pred)
plt.xlabel('Real price')
plt.ylabel('Expected price')
plt.show()

r2 = model.score(x_test, y_test)
print('R2 accuracy/value =', r2)