import pandas as pd
import numpy

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


print('\nОценка ргрессионной модели')

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred_test)
print('MSE = ',MSE,' - среднеквадратичная ошибка')

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(y_test, y_pred_test)
print('\nRMSE = ',RMSE,' - корень среднеквадратичной ошибки')

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)
print('\nMAE = ',MAE,' - средняя абсолютная ошибка')

#from sklearn.preprocessing import PolynomialFeatures
#n = 3
#poly_features = PolynomialFeatures(n)
#X_train = poly_features.fit_transform(X_train)
#linear_model.fit(X_train, y_train)

print('\nУлучшение ргрессионной модели')

from sklearn.linear_model import Ridge
l2_linear_model = Ridge(alpha=2.0)

l2_linear_model.fit(X_train, y_train)
y_pred_test = l2_linear_model.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred_test)
print('\nMSE = ',MSE,' - среднеквадратичная ошибка')

from sklearn.metrics import root_mean_squared_error
RMSE = root_mean_squared_error(y_test, y_pred_test)
print('\nRMSE = ',RMSE,' - корень среднеквадратичной ошибки')

from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred_test)
print('\nMAE = ',MAE,' - средняя абсолютная ошибка')

print('\nОценка классификационной модели')

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)


from sklearn.linear_model import LogisticRegression
logreg_model = LogisticRegression()

logreg_model.fit(X_train_std, y_train)
y_pred_test = logreg_model.predict(X_test_std)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_test)
print('\nAccuracy = ',accuracy,' - доля правильных классификаций модели')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print('Error = ',cm,' - матрица ошибок')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Graf.png')

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred_test,  zero_division = numpy.nan)
print(report)



