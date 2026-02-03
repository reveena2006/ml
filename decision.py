import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_text


data = {
    'Attendance': ['High','High','High','Medium','Medium','Medium','Low','Low','Low','High'],
    'Study_Hours': ['High','Medium','Low','High','Medium','Low','High','Medium','Low','High'],
    'Result': ['Pass','Pass','Fail','Pass','Pass','Fail','Fail','Fail','Fail','Pass']
}

df = pd.DataFrame(data)

print("ORIGINAL DATASET:\n")
print(df)


le_attendance = LabelEncoder()
le_study = LabelEncoder()
le_result = LabelEncoder()

df['Attendance'] = le_attendance.fit_transform(df['Attendance'])
df['Study_Hours'] = le_study.fit_transform(df['Study_Hours'])
df['Result'] = le_result.fit_transform(df['Result'])

print("\nENCODED DATASET:\n")
print(df)


X = df[['Attendance', 'Study_Hours']]
y = df['Result']


model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)


print("\nDECISION TREE STRUCTURE:\n")
tree_rules = export_text(model, feature_names=['Attendance', 'Study_Hours'])
print(tree_rules)


attendance_input = le_attendance.transform(['High'])[0]
study_input = le_study.transform(['Low'])[0]

new_data = [[attendance_input, study_input]]
prediction = model.predict(new_data)


result = le_result.inverse_transform(prediction)

print("\nNEW INSTANCE PREDICTION:")
print("Attendance = High")
print("Study Hours = Low")
print("Predicted Result =", result[0])
