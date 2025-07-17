from preprocess import load_and_preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load and preprocess data
(X_train, X_test, y_train, y_test), vectorizer = load_and_preprocess('data/reviews.csv')

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
