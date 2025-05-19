import pandas as pd
import numpy as np
from word2number import w2n
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

df = pd.read_excel(r"C:\Users\UGANDA\Downloads\Book1.csv.xlsx")
df.replace("?", np.nan, inplace=True)

def saf_work_to_num(x):
    try:
        return w2n.word_to_num(x) if isinstance(x, str) else x
    except:
        return np.nan

df['age'] = df['age'].apply(saf_work_to_num)
df['age'].fillna(df['age'].mean(), inplace=True)
k = df['salary'].mean()
df['gender'].replace(np.nan, "other", inplace=True)
df['salary'].replace(np.nan, k, inplace=True)

print(df)
 # unused, but retained as per your request
df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
# x and y are Series, we need to reshape x for sklearn
df['gender']=df['gender'].astype(str)
df=pd.get_dummies(df,columns=['gender'])
x = df[['age','gender_Male','gender_other']]
y = df['salary']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Removed: x_pred = model.predict(y_test) — this was invalid
# If you want to predict salary using age, x_test is sufficient

print("Predicted salaries:", y_pred)
print("Actual salaries   :", y_test.values)
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
new_data=pd.DataFrame({'age':[28],'gender_Male':[1],'gender_other':[0]})
predected_salary=model.predict(new_data)
print(predected_salary)
