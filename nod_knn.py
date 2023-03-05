import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import ast
import time


df1 = pd.read_csv('data/genes_with_onehot.csv', on_bad_lines='skip', nrows=40000)

#extract x and y
X = df1['onehot_seq']
y = df1['nod_relation'].astype(int).values

#convert pandas series to float array
X = X.tolist()
print(X[0])

#split data
X_train, X_test, y_train, y_test = train_test_split(float_X, y, test_size=0.2)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(predictions)
print(y_test)

num_correct = (predictions == y_test).sum()
print(num_correct)
print(num_correct/len(X_train))
