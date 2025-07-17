import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):
    # Load CSV file
    data = pd.read_csv(path)

    # Convert text to numerical features
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['text'])

    # Labels
    y = data['label']

    # Split into train and test sets
    return train_test_split(X, y, test_size=0.2), vectorizer
