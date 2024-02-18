from datasets import load_dataset
import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import codecs
from datasets import load_dataset
import os
df = pd.read_csv('results/trump_tweets.csv')
df.columns = ["id", "text", "is_retweet", "is_deleted", "device", "favorites", "retweets", "datetime", "is_flagged", "date"]
df.drop("id", axis = 1, inplace = True)
df.drop("is_retweet", axis = 1, inplace = True)
df.drop("is_deleted", axis = 1, inplace = True)
df.drop("device", axis = 1, inplace = True)
df.drop("favorites", axis = 1, inplace = True)
df.drop("retweets", axis = 1, inplace = True)
df.drop("datetime", axis = 1, inplace = True)
df.drop("is_flagged", axis = 1, inplace = True)
df.drop("date", axis = 1, inplace = True)
print(df)
model1_path = os.path.join('pickle', 'model1.p')
with open(model1_path, 'rb') as file:
    model = pickle.load(file)
vectorizer_path = os.path.join('pickle', 'vectorizer.pk1')
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)
X_new = vectorizer.transform(df['text'])
predictions = model.predict(X_new)
df['predictions'] = predictions
df.to_csv('results/df_with_predictions.csv', index=False)