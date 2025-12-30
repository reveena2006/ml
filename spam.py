import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv("spam_dataset.csv")
le=LabelEncoder()
df["offer"]=le.fit_transform(df["offer"])
df["free"]=le.fit_transform(df["free"])
df["class"]=le.fit_transform(df["class"])
X=df[["offer","free"]]
y=df["class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=BernoulliNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy:{accuracy*100:.2f}%")
new_email = [[1, 1]]
prediction = model.predict(new_email)
print("Prediction for new email:", "Spam" if prediction[0]==1 else "Not Spam")
