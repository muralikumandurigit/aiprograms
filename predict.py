# predict.py
import joblib

# Load model
model = joblib.load("model/intent_model.pkl")

def classify_email(text: str):
    """
    Predicts intent using trained ML model.
    This is production-ready: deterministic, fast, scalable.
    """
    prediction = model.predict([text])[0]
    return prediction

# Example usage
if __name__ == "__main__":
    email = "Can I get my experience certificate?"
    print("Predicted Intent:", classify_email(email))
