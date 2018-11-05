import sys
import pandas as pd
import numpy as np
import re
import nltk
from nltk import FreqDist, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_table(sys.argv[1], header=None, encoding = 'utf-8')
nq = pd.read_table(sys.argv[2], header=None, encoding = 'utf-8')
char_list = pd.read_table('recursos/list_characters.txt', sep="\n", header=None, encoding = 'utf-8')
company_list = pd.read_table('recursos/list_companies.txt', header=None, encoding = 'utf-8')
genre_list = pd.read_table('recursos/list_genres.txt', header=None, encoding = 'utf-8')
job_list = pd.read_table('recursos/list_jobs.txt', header=None, encoding = 'utf-8')
keyword_list = pd.read_table('recursos/list_keywords.txt', header=None, encoding = 'utf-8')
movie_list = pd.read_table('recursos/list_movies.txt', header=None, encoding = 'utf-8')
people_list = pd.read_table('recursos/list_people.txt', sep="\n", header=None, encoding = 'utf-8')


Y = df[0]
X = df[1]
nq = nq[0]

all_words = []
key_list = [char_list, company_list, genre_list, job_list, keyword_list,
            movie_list, people_list]

for key in key_list:
    key = key[0]
    
key_dict = {"movie_key ":movie_list, "character_key":char_list, "company_key ":company_list, "genre_key ":genre_list, 
            "job_key ":job_list, "people_key ":people_list}

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
tokenizer = PunktSentenceTokenizer()

def remove_spaces(text):
    return text.strip(' ')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    
    return text

def remove_stopwords(text):
    filtered_text = []
    for word in text:
        if word not in stop_words:
            filtered_text.append(word)
    return filtered_text

def lemmatize_words(text):
    text = [lemmatizer.lemmatize(word) for word in text]
    text = ' '.join(text)
    return text


def gather_all_words(text):
    for words in text:
        words = word_tokenize(words)
        for word in words:
            all_words.append(word)
    return all_words

def find_features(text):
    features = {}
    for word in word_features:
        features[word] = (word in text)
    return features

def find_feature_set(data):
    feature_set = []
    for text, clss in data:
        feature_set.append((find_features(text),clss))
    return feature_set

X = X.apply(lambda text : clean_text(text))
X = X.apply(lambda text : word_tokenize(text))
X = X.apply(lambda text : remove_stopwords(text))
X = X.apply(lambda text : lemmatize_words(text))

gather_all_words(X)
        
all_words_fd = FreqDist(all_words)
word_features = list(all_words_fd.keys())
quest = list(zip(X, Y))

np.random.seed = 1       #seed para repetição
np.random.shuffle(quest) #shuffling para garantir que as classes não ficam agrupadas

feature_set = find_feature_set(quest)

nq = nq.apply(lambda text : clean_text(text))
nq = nq.apply(lambda text : word_tokenize(text))
nq = nq.apply(lambda text : remove_stopwords(text))
nq = nq.apply(lambda text : lemmatize_words(text))

nq_feature_set = []
for text in nq:
    nq_feature_set.append(find_features(text))

nltk_classifier = SklearnClassifier(LogisticRegression())
nltk_classifier.train(feature_set)
class_set = nltk_classifier.classify_many(nq_feature_set)

    