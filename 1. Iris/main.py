from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

iris_dataset = load_iris()

# the object returned by load_iris is a bunch object which is like a dict with keys and values 

# see the keys
print("Keys of Iris_dataset:\n", iris_dataset.keys())

# target_names contains an array of strings specifying the species we want to predict
print("Target names: ", iris_dataset['target_names'])

# feature names is a list of strings giving a description of each feature
print("Feature_names: ", iris_dataset['feature_names'])

# the data itself is contained in the 'target' and 'data' fields. 'data' contains the numeric measurements of features in a NumPy array:
print("Type of data: ", type(iris_dataset['data']))

# the rows in the data each represent an individual flower
# the columns represent the four measurements taken for each flower
print("Shape of the data: ", iris_dataset["data"].shape)

# print the first five rows of the data
print("The first five lines of the data: \n", iris_dataset['data'][:5])

# the species of each row is contained also in a NumPy array in the target array
print("Data type of 'target': ", type(iris_dataset['target']))

# shape of target is a 1 dimensional array with one entry per flower
print("Shape of the target array: ", iris_dataset['target'].shape)

# the species are encoded as integers from 0 to 2
print("Target: \n", iris_dataset['target'])

# the meaning of these integers are given in the target_names array:
print("Target name integer meaning:")
for i in range(len(iris_dataset['target_names'])):
    print(i, ": ", iris_dataset["target_names"][i])

# create a training and test split with the data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)

# by default this creates a 75% v 25% train to test split
print("X train shape: ", X_train.shape)
print("y train shape: ", y_train.shape)
print("X test shape: ", X_test.shape)
print("y test shape: ", y_test.shape)

# create dataframe from data in X_train
# label the columns using teh strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix rom the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
plt.show()