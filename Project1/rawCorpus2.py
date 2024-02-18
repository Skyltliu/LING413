from datasets import load_dataset
from datasets import load_dataset
import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import codecs
import os
dataset = load_dataset("mteb/toxic_conversations_50k")
train_dataset = dataset['train']
test_dataset = dataset['test']
train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()
consolidated_df = pd.concat([train_df, test_df], ignore_index=True)
consolidated_df = consolidated_df.sample(frac=1).reset_index(drop=True)
consolidated_df.columns = ["text", "label", "label_text"]
consolidated_df.drop("label", axis = 1, inplace=True)
consolidated_df.drop("label_text", axis = 1, inplace=True)
print(consolidated_df)
model1_path = os.path.join('pickle', 'model1.p')
with open(model1_path, 'rb') as fw:
    model = pickle.load(fw)
vectorizer_path = os.path.join('pickle', 'vectorizer.pk1')
with open(vectorizer_path, 'rb') as fw2:
    vectorizer = pickle.load(fw2)
X_new = vectorizer.transform(consolidated_df['text'])
predictions = model.predict(X_new)
rawCorpus_predictions_df = pd.DataFrame(predictions)
rawCorpus_predictions_df.to_csv('rawCorpus_predictions.csv', index=False)
consolidated_df['predictions'] = predictions
full_path = os.path.join('results', 'toxicEval.csv')
consolidated_df.to_csv(full_path, index=False)
