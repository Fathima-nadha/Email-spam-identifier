# spam_identifier_beginner.py
# Beginner-friendly Email/SMS Spam Identifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1) Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[['v1','v2']]
data.columns = ['label', 'text']

# 2) Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)

# 3) Convert text to numbers (Bag of Words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 4) Train a simple model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5) Evaluate
pred = model.predict(X_test_vec)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print()
print("Breakdown:\n", classification_report(y_test, pred))

# 6) Try your own message
your_msg = input("\nType a message to check: ")
your_vec = vectorizer.transform([your_msg])
print("Prediction:", model.predict(your_vec)[0])
