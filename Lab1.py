import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")
df.head()
df.info()
cols = df.columns
df.columns.tolist()

obj = 'Name'
if obj in cols:
	print('Dataset has column Name')
obj = 'Drowned'
if obj not in cols:
	print('Dataset hasn’t column Drowned')

cols_without_age = df.columns.drop('Age') # колонки без Age
cols_without_age_and_name = df.columns.drop(['Age', 'Name']) # колонки без Age и Name

for col in df:
    print(col) #вывод всех названий столбцов

age_column = df['Age']
age_and_name_df = df[['Age','Name']]

numeric_df = df.select_dtypes(include='number') # цифровые данные
not_numeric_df = df.select_dtypes(exclude='number') # текстовые данные

df_without_name = df.drop('Name', axis='columns')

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

nan_matrix = df.isnull()
print(nan_matrix)

sum = nan_matrix.sum()
print(sum)

cabin_mode = df['Cabin'].mode()[0]
age_median = df['Age'].median()

df['Cabin'] = df['Cabin'].fillna(cabin_mode)
df['Age'] = df['Age'].fillna(age_median)

nan_matrix2 = df.isnull()
print(nan_matrix2)

sum2 = nan_matrix2.sum()
print(sum2)

print(df['Age'])




from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()

#scaler.fit(df['Age'])
#scaler.transform(df['Age'])
#scaler.fit_transform(df['Age'])
#print('--------')
#print(df['Age'])

#print(df['Cabin'])
#df = pd.get_dummies(df, columns=['Cabin'], drop_first=True)
#print('--------')
#print(df)

#print(df['Name'])
#df = pd.get_dummies(df, columns=['Name'], drop_first=True)
#print('--------')
#print(df)

print(df['Sex'])
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
print('--------')
print(df)

#print(df['Embarked'])
#df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
#print('--------')
#print(df)


#print(df['Ticket'])
#df = pd.get_dummies(df, columns=['Ticket'], drop_first=True)
#print('--------')
#print(df)
df = df.drop(['Ticket'], axis='columns')
df = df.drop(['Embarked'], axis='columns')
df = df.drop(['Name'], axis='columns')
df = df.drop(['Cabin'], axis='columns')

df.to_csv("Titanic-Dataset2.csv", index=False, sep=',')

