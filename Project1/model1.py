from datasets import load_dataset
import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle
import codecs
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
print(dataset['train']['human_annotation'])
train_dataset = dataset['train']
df = pd.DataFrame(train_dataset)
print(df.info())
print(df.describe())
df['toxicity'].value_counts().plot(kind='bar')
df['user_input'] = df['user_input'].str.lower()
df['user_input'] = df['user_input'].str.replace(r'[^a-z\s]', '', regex=True)
X_train, X_test, y_train, y_test = train_test_split(df['user_input'], df['toxicity'], test_size=0.2, random_state=42)
vectorizer = feature_extraction.text.CountVectorizer(stop_words='english', max_features=10000)
print("Train shape", X_train.shape, y_train.shape)
print("Test shape", X_test.shape, y_test.shape)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
vocab = vectorizer.vocabulary_
with codecs.open("Project1/results/model1_vocab_info.txt", "w", encoding = "utf-8") as f:
    for k, v in sorted(vocab.items()):
        f.write(k+ "," + str(v) + '\n')
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
predictions = model.predict(X_test_vec)
model1_test_df = pd.DataFrame(predictions)
model1_test_df.to_csv('Project1/results/model1_test_predictions.csv', index=False)
report = classification_report(y_test, predictions)
print(report)
with codecs.open("Project1/results/model1_results.txt", "w", encoding = "utf-8") as fw:
    fw.write(report)
with open("Project1/pickle/model1.p", "wb") as handle:
    pickle.dump(model, handle)
with open('Project1/pickle/vectorizer.pk1', 'wb') as file:
    pickle.dump(vectorizer, file)
