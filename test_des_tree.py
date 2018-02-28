from sklearn.datasets import load_digits, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from des_tree import DecisionTree


X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

tree = DecisionTree(criterion='entropy')
print(accuracy_score(tree.fit(X_train, y_train).predict(X_test), y_test))

tree = DecisionTree(criterion='gini')
print(accuracy_score(tree.fit(X_train, y_train).predict(X_test), y_test))

X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

tree = DecisionTree(criterion='mad_median')
print(mean_squared_error(tree.fit(X_train, y_train).predict(X_test), y_test))

tree = DecisionTree(criterion='variance')
print(mean_squared_error(tree.fit(X_train, y_train).predict(X_test), y_test))
