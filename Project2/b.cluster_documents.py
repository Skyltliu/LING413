import random
import pandas as pd
from sklearn import feature_extraction
from sklearn.cluster import MiniBatchKMeans
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import os
#--------------------------------------------------------------------------------
def cluster(stopwords, data_df, cluster_prefix = "Main"):

    features = feature_extraction.text.TfidfVectorizer(input='content', 
                                                    encoding='utf-8', 
                                                    decode_error='ignore', 
                                                    lowercase=True,
                                                    stop_words = stopwords,
                                                    tokenizer = None,
                                                    ngram_range=(1, 1), 
                                                    analyzer='word', 
                                                    max_features=10000,
                                                    )
    features.fit(data_df.loc[:,"comment_text"].values)

    x = features.transform(data_df.loc[:,"comment_text"].values)
    print(x)
    print(x.shape)

    cluster = MiniBatchKMeans(n_clusters = 10, 
                        init = "k-means++", 
                        n_init = "auto", 
                        max_iter = 30000,
                        )
    cluster.fit(x)

    data_df.loc[:,"Topic"] = [cluster_prefix+"_"+str(x) for x in cluster.labels_]

    data_df.sort_values("Topic", inplace = True)
    print(data_df)
    print(data_df.value_counts("Topic"))

    most_frequent = data_df.value_counts("Topic").index[0]
    most_frequent_count = data_df.value_counts("Topic").iloc[0]
    print("Most frequent: ", most_frequent)

    main_topic = data_df[data_df.loc[:,"Topic"] == most_frequent]
    other_topics = data_df[data_df.loc[:,"Topic"] != most_frequent]
    return main_topic, other_topics, most_frequent
#--------------------------------------------------------------------------------

file = "test_preprocessed.csv"
data_df = pd.read_csv(file)
print(data_df.columns)
data_df['comment_text'] = data_df['comment_text'].fillna('')

phrase_model = Phrases([doc.split() for doc in data_df.loc[:,"comment_text"].values], 
                        min_count = 2, 
                        threshold = 0.7, 
                        connector_words = ENGLISH_CONNECTOR_WORDS, scoring = "npmi"
                        )

print(phrase_model.export_phrases().keys())
print("ABOVE: Learned phrases")

data_df.loc[:,"comment_text"] = [" ".join(phrase_model[sentence.split()]) for sentence in data_df.loc[:,"comment_text"]]

features = feature_extraction.text.CountVectorizer(input='content', 
                                                encoding='utf-8', 
                                                decode_error='ignore', 
                                                lowercase=True, 
                                                tokenizer = None,
                                                ngram_range=(1, 1), 
                                                analyzer='word', 
                                                max_features=500,  
                                                )

features.fit(data_df.loc[:,"comment_text"].values)
stopwords = list(features.vocabulary_.keys())
print(stopwords)
print("ABOVE: Frequent words to exclude")

main_topic = data_df    
cluster_prefix = "Topic"    
holder = []
starting_length = len(data_df)
counter = 0

while True:

    counter += 1
    main_topic, other_topics, most_frequent = cluster(stopwords, main_topic, cluster_prefix)
    cluster_prefix = str(most_frequent)


    if len(main_topic)/len(data_df) < 0.20: #or counter >= 10:
        holder.append(other_topics)
        holder.append(main_topic)
        break


    else:
        holder.append(other_topics)
        print("Continuing after round " + str(counter), "Current: ", len(main_topic), "Total: ", starting_length)
        

data_df = pd.concat(holder)
data_df.sort_values("Topic", inplace = True)

#Save
data_df.to_csv("organized_commentsB.csv", index=False)
sample_size = 15 
cluster_samples = {}


folder_name = 'modelB'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
for cluster_label in sorted(data_df['Topic'].unique()):
    cluster_df = data_df[data_df['Topic'] == cluster_label]
    filename = os.path.join(folder_name, f"cluster_{cluster_label}.csv")
    cluster_df.to_csv(filename, index=False)
    print(f"Saved {len(cluster_df)} comments to {filename}")
