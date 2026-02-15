from sklearn.datasets import load_iris
iris=load_iris()
x=iris.data
y=iris.target
print(iris.feature_names,iris.target_names)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Predictions:",y_pred[:5])
print("True labels:",y_test[:5])
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
import os
import joblib
os.makedirs("outputs", exist_ok=True)
import matplotlib.pyplot as plt

plt.figure()
plt.title("Confusion Matrix")
plt.table(cellText=cm, loc='center')
plt.axis('off')

plt.savefig("outputs/confusion_matrix.png")
plt.close()
joblib.dump(model, "outputs/decision_tree_model.joblib")
