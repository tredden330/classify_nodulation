import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import ast
import time
import matplotlib.pyplot as plt
import numpy as np
import time

#track program runtime
start_time = time.time()

df1 = pd.read_csv('data/genes_with_onehot.csv', on_bad_lines='skip', nrows=100000)

keys = map(str, range(1200))
X = df1[keys]
y = df1['nod_relation'].astype(int).values

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print(y_train)

k_range = range(1,500)
results = []
for num in k_range:
    classifier = KNeighborsClassifier(n_neighbors=num)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    print("num = ", num)

    num_correct = (predictions == y_test).sum()
    print(num_correct)
    print(num_correct/len(X_test))
    results.append(num_correct/len(X_test))
    
fig, ax = plt.subplots()
ax.plot(k_range, results)
plt.title("Knn - correct guessing")
plt.xlabel("k-value")
plt.ylabel("percent correct")
ax.grid()
fig.savefig("knn_percent.png")

print("finished in ", time.time() - start_time, " seconds")