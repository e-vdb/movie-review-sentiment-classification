import pickle
import pandas as pd


def load_model():
    """Load the model from disk."""
    filename = 'model/finalized_model.sav'
    return pickle.load(open(filename, 'rb'))


def make_prediction(model, text):
    pred = model.predict([text])
    output = "The review is " + "".join(pred) +"."
    return pred, output

def details_proba(model, text):
    return pd.DataFrame(model.predict_proba([text]), columns=['Negative', 'Positive'], index=['Review'])
