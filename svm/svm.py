import numpy as np
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("breast-cancer-wisconsin.data")

# Replace missing data with some, you don't want to just trash every data when is missing:
df.replace("?", -99999, inplace=True)

# We don't want the id column to be able to classify instances!:
df.drop(["id"], 1, inplace=True)

# Define X and Y:
X = np.array(df.drop(["class"], 1))
y = np.array(df["class"])

# Split the data:
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

# Classify data:
clf = svm.SVC()
clf.fit(X_train, y_train)

# Statistics:
accuracy = clf.score(X_test, y_test)
print("ACCURACY: {}".format(accuracy))

# Test some non existing predictions in the test data:
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])

prediction = clf.predict(example_measures)

print("PREDICTION: {}".format(prediction))