from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
thresholds_to_try = [0.3, 0.5, 0.7]

for threshold in thresholds_to_try:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    print(f"\nThreshold: {threshold}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_threshold))
    print("F1 Score:", classification_report(y_test, y_pred_threshold, output_dict=True)['weighted avg']['f1-score'])


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

youden_index = tpr - fpr
best_idx = np.argmax(youden_index)
best_threshold = thresholds[best_idx]

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label='ROC Curve', color='green')
plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='red', label=f'Best Threshold = {best_threshold:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Best Threshold')
plt.legend()
plt.grid(True)
plt.show()

print(f"Best Threshold by Youden's J Index: {best_threshold}")
