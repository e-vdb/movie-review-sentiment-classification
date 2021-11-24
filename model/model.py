import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from data_processing import setup_datatrain
import pickle
my_filepath="IMDB Dataset.csv"
reviews = pd.read_csv(my_filepath)

X_train, y_train = setup_datatrain(reviews)
model = Pipeline([('tvec',TfidfVectorizer(min_df=5,stop_words='english')),('lg',LogisticRegression(C=0.1,max_iter=500))])
model.fit(X_train,y_train)
print("Accuracy score on training set {}".format(model.score(X_train,y_train)))

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
print('Model saved')