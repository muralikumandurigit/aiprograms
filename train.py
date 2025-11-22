# train.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------------------
# Example training dataset (in real production, load from DB or CSV)
# -----------------------------------

emails = [
    "I want to check the price of your product",             # Sales
    "Please send me a quote for 100 units",                  # Sales
    "My order is not delivered yet",                         # Support
    "The application is showing an error",                   # Support
    "I need to apply for leave tomorrow",                    # HR
    "Can you share my salary slip?",                         # HR
]

labels = [
    "sales",
    "sales",
    "support",
    "support",
    "hr",
    "hr"
]

# -----------------------------------
# Split data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    emails, labels, test_size=0.2, random_state=42
)

# -----------------------------------
# Production pipeline: TF-IDF + Logistic Regression
# -----------------------------------

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluation
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))

# -----------------------------------
# Save model + vectorizer in production-friendly format
# -----------------------------------

joblib.dump(pipeline, "model/intent_model.pkl")

print("Model training completed. Saved to model/intent_model.pkl")
