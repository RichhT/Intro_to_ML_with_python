import mglearn
import matplotlib.pyplot as plt

# generate synthetic dataset
X, y = mglearn.datasets.make_forge()

#plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)

plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")


mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

print("tes")
