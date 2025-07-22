from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

iris_model = LogisticRegression(multi_class='ovr', max_iter=1000)
iris_model.fit(X_train_iris, y_train_iris)

y_pred_iris = iris_model.predict(X_test_iris)

print("Accuracy:", accuracy_score(y_test_iris, y_pred_iris))
print("\nClassification Report:\n", classification_report(y_test_iris, y_pred_iris))
