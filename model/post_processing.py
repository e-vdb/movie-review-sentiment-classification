import pickle
import pandas as pd
def load_model():
    """Load the model from disk."""
    filename = 'finalized_model.sav'
    return pickle.load(open(filename, 'rb'))


def make_prediction(model, text):
    return "The review is " + "".join(model.predict([text])) +"."

def details_proba(model, text):
    details = pd.DataFrame(model.predict_proba([entry]), columns=['Negative', 'Positive'], index=['Review'])
    print(f"Probabilities")
    print(details)
