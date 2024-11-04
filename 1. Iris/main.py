from sklearn.datasets import load_iris

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