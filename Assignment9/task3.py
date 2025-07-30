# Task 9: AdaBoost or Gradient Boosting
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import time

df = sns.load_dataset('titanic')
df = df[['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']].dropna()
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df.drop('survived', axis=1)
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)


rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


boost_model = GradientBoostingClassifier(random_state=0)

start = time.time()
boost_model.fit(X_train, y_train)
end = time.time()
boost_preds = boost_model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, boost_preds))
print("F1 Score:", f1_score(y_test, boost_preds))
print("Training Time: {:.4f} seconds".format(end - start))


print("\n--- Comparison of All Models ---")
models = {
    'Decision Tree': dt_preds,
    'Random Forest': rf_preds,
    'Gradient Boosting': boost_preds
}

for name, preds in models.items():
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Score:", f1_score(y_test, preds))
