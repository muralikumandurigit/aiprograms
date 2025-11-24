import pickle

with open("model/intent_model.pkl", "rb") as f:
    model = pickle.load(f)

while True:
    text = input("\nEnter query: ")
    pred = model.predict([text])
    print("Predicted Intent:", pred[0])
