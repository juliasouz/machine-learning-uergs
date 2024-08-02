import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
print(iris.target_names)

parameters = {'n_neighbors': [1, 3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']}
knn = KNeighborsClassifier()

grid = GridSearchCV(knn, parameters, cv=5)
grid.fit(x, y)
print("Best parameters: ",grid.best_params_)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best = grid.best_estimator_
best.fit(x_train, y_train)

y_prev = best.predict(x_test)
confusion = confusion_matrix(y_test, y_prev)
print("Confusion matrix: ",confusion)

plt.imshow(confusion, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1, 2], iris.target_names)
plt.yticks([0, 1, 2], iris.target_names)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion matrix - Iris dataset')

for i in range(3):
    for j in range(3):
        plt.text(j, i, confusion[i, j], ha="center", va="center", color="black")

plt.show()

accuracy = confusion.diagonal().sum() / confusion.sum()
print("KNN model accuracy: ", accuracy)
