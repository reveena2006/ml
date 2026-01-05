import pandas as pd
from transformers import pipeline

df = pd.read_csv("spam_dataset.csv")


df.columns = df.columns.str.strip().str.capitalize()
print("Columns detected:", df.columns)

def feature_to_text(row):
    offer_text = "contains offer" if row["Offer"].strip().lower() == "yes" else "no offer"
    free_text = "contains free" if row["Free"].strip().lower() == "yes" else "no free"
    return f"This email {offer_text} and {free_text}."

df["text"] = df.apply(feature_to_text, axis=1)

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


predictions = []
for idx, row in df.iterrows():
    text = row["text"]
    result = classifier(text)[0]
    predictions.append(result)
    print(f'Text: "{text}"')
    print(f'Prediction: {result["label"]} (Score: {result["score"]:.2f})\n')

df["prediction"] = [p["label"] for p in predictions]
df["score"] = [p["score"] for p in predictions]

print(df[["Offer","Free","Class","text","prediction","score"]])
