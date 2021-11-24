# movie-review-sentiment-classification

![1200px-IMDB_Logo_2016Rescaled](https://user-images.githubusercontent.com/82372483/124572982-67c6e900-de49-11eb-9e45-ee0e3a973ae1.png)


## Summary

Sentiment analysis of movie reviews from Internet Movies Database (IMDb) 

## Dataset

Dataset from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

50000 reviews : 25000 positive - 25000 negative

## Streamlit interface



## Repository content 
* model/ folder containing different Python scripts
* moviereview-sentimentanalysis.ipynb: Jupyter Notebook 

## Notebook

* Natural Language Processing (NLP)
* Machine learning model : Logistic Regression
* Evaluation of the accuracy score

## Visualization of the most discriminating features

### Count Vectorizer (unigrams only)

![featuresCV](https://user-images.githubusercontent.com/82372483/125919019-1e3410ec-6e2c-45f9-ab8c-9c66672b0b94.png)

### Count Vectorizer (unigrams and bigrams)

![featuresCV2](https://user-images.githubusercontent.com/82372483/125951592-7e1856ab-985a-41b0-a589-9d702e81c8b5.png)


### Count Vectorizer with TFIDF rescaling (unigrams only)

![featuresTFIDF](https://user-images.githubusercontent.com/82372483/125919113-619e01fe-7463-43bd-8733-ef00b4a1ee37.png)

### Count Vectorizer with TFIDF rescaling (unigrams and bigrams)

![featuresTFIDF2](https://user-images.githubusercontent.com/82372483/126063600-36c883d8-8b56-427e-83bd-234b3a631508.png)


## TASK LIST
- [x] Implement machine learning algorithm using Scikit-learn
- [ ] Implement deep learning algorithm using Keras
- [ ] Deploy model

