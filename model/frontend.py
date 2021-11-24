import streamlit as st
from post_processing import load_model, make_prediction, details_proba

ML_model = load_model()
# Set the app title
st.title("Movie review: Sentiment Analysis App")
st.write(
    "A simple machine learning app to predict the sentiment of a movie's review"
)
# Declare a form to receive a movie's review
form = st.form(key="my_form")
review = form.text_input(label="Enter the text of your movie review")
submit = form.form_submit_button(label="Make Prediction")
show = form.checkbox('Show details')
if submit:
    result = make_prediction(ML_model, review)
    # Display results of the NLP task
    st.header("Results")
    st.write(result)

if show:
    st.write("Probabilities")
    st.write(details_proba(ML_model, review))