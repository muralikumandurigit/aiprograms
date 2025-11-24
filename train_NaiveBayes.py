import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import re


# -----------------------------------
# 1. CLEAN TEXT FUNCTION
# -----------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------------
# 2. LOAD DATA
# -----------------------------------
df = pd.read_csv("data.csv")  # columns: text, intent
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["intent"]


# -----------------------------------
# 3. TRAIN-TEST SPLIT
# -----------------------------------
# test_size automatically adjusts based on number of classes
test_size = max(0.3, len(set(y)) / len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)


# -----------------------------------
# 4. PIPELINE (TF-IDF + MultinomialNB)
# -----------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=1,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", MultinomialNB())
])


# -----------------------------------
# 5. TRAIN MODEL
# -----------------------------------
model.fit(X_train, y_train)


# -----------------------------------
# 6. EVALUATE MODEL
# -----------------------------------
preds = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))


# -----------------------------------
# 7. SAVE MODEL
# -----------------------------------
os.makedirs("model", exist_ok=True)

with open("model/intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model/intent_model.pkl")
