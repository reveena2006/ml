
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text


data = {
    'id': [1,2,3,4,5,6,7,8,9,10],
    'fever': [1,0,1,0,1,0,1,0,1,0],
    'cough': [1,1,0,0,1,0,1,0,0,1],
    'fatigue': [0,1,1,0,1,0,1,0,1,1],
    'disease': ['Yes','Yes','Yes','Yes','Yes','No','No','No','No','No']
}


df = pd.DataFrame(data)
print("ORIGINAL DATASET:\n")
print(df)


le_disease = LabelEncoder()
df['disease'] = le_disease.fit_transform(df['disease'])

print("\nENCODED DATASET:\n")
print(df)


X = df[['fever', 'cough', 'fatigue']]
y = df['disease']


rf_model = RandomForestClassifier(
    n_estimators=10,     
    criterion='entropy',
    max_features=2,       
    random_state=42
)
rf_model.fit(X, y)

print("\nRandom Forest trained successfully with", len(rf_model.estimators_), "trees.")


for i, tree in enumerate(rf_model.estimators_):
    print(f"\n--- Rules of Tree {i+1} ---\n")
    print(export_text(tree, feature_names=['fever','cough','fatigue']))


new_data = [[1, 0, 1]]  # fever=1, cough=0, fatigue=1
pred = rf_model.predict(new_data)
print("\nPrediction for fever=1, cough=0, fatigue=1:", le_disease.inverse_transform(pred)[0])
