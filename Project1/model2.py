from datasets import load_dataset
import pandas as pd
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import codecs
dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
train_dataset = dataset['train']
df = pd.DataFrame(train_dataset)
df['user_input'] = df['user_input'].str.lower()
df['user_input'] = df['user_input'].str.replace(r'[^a-z\s]', '', regex=True)
X_train, X_test, y_train, y_test = train_test_split(df['user_input'], df['toxicity'], test_size=0.2, random_state=42)
vectorizer = feature_extraction.text.CountVectorizer(stop_words='english', max_features=10000)
print("Train shape", X_train.shape, y_train.shape)
print("Test shape", X_test.shape, y_test.shape)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
vocab = vectorizer.vocabulary_
with codecs.open("Project1/results/model2_vocab_info.txt", "w", encoding = "utf-8") as f:
    for k, v in sorted(vocab.items()):
        f.write(k+ "," + str(v) + '\n')
param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_vec)
report = classification_report(y_test, predictions)
print(report)
with codecs.open("Project1/results/model2_results.txt", "w", encoding = "utf-8") as fw:
    fw.write(report)
with open("Project1/pickle/model2.p", "wb") as handle:
    pickle.dump(best_model, handle)