from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

print("Housing keys: \n", housing.keys())

print("Data shape: \n", housing.data.shape)