import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("insurance.csv")
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])
df['region'] = le.fit_transform(df['region'])
df['high_charges'] = (df['charges'] > df['charges'].mean()).astype(int)

X = df.drop(['charges', 'high_charges'], axis=1)
y = df['high_charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.3f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
new_data = pd.DataFrame({'age': [28], 'sex': [1], 'bmi': [30], 'children': [2], 'smoker': [1], 'region': [2]})
new_pred = log_reg.predict(new_data)
print(f'Predicted charges: {new_pred[0]:.2f}')