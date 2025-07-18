import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    data = pd.read_csv(path)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    return train_test_split(X, y, test_size=0.2), vectorizer
