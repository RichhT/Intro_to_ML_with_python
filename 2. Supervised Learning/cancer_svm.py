from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics

cancer = load_breast_cancer()

print(cancer.keys())

print("Feature names: \n", cancer.feature_names)

print("Target names: \n", cancer.target_names)

print("Data shape: \n", cancer.data.shape)

print("First five rows of data: \n", cancer.data[0:5])

print("Target data: \n", cancer.target)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=8)

# create svm classifier
clf = svm.SVC(kernel='poly')

# train the mnodel
clf.fit(X_train, y_train)

# predict 
y_pred = clf.predict(X_test)

# calculate model accuracy
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# calculate precision score
print("Precision: ", metrics.precision_score(y_test, y_pred))

# calculate the recall score
print("Recall: ", metrics.recall_score(y_test, y_pred))