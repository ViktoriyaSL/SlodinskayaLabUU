import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Titanic-Dataset2.csv")
df.head()
df.info()
cols = df.columns
df.columns.tolist()

from sklearn.model_selection import train_test_split

X = df.drop(['Age'], axis='columns')
y = df['Age']
y = [int(label) for label in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.4, random_state=42)

from sklearn.tree import DecisionTreeClassifier

dt_classifier_model = DecisionTreeClassifier(max_depth=4, max_leaf_nodes = 10)

dt_classifier_model.fit(X_train, y_train)
y_proba = dt_classifier_model.predict_proba(X_test)
print(dt_classifier_model.classes_)

print(y_proba)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1], pos_label=1)

plt.figure()
plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC curve')
plt.savefig('ROC curve.png')

from sklearn.metrics import auc
auc_metric = auc(fpr, tpr)


from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import load_iris
from sklearn import tree

dt_regressor_model = DecisionTreeRegressor()
dt_regressor_model.fit(X_train, y_train)
plt.figure()
tree.plot_tree(dt_regressor_model)
plt.show()
plt.savefig('Reg')