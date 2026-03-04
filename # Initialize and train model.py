# Initialize and train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Predict
y_pred = model.predict(X_test_tfidf)
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load data
df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=["label", "text"])
# Map labels to binary
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
# Show class distribution
print(df['label'].value_counts())
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.show()
# Split data
X = df['text']
y = df['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
import joblib
# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
import joblib
# Save model and vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
# Predict new message
def predict_message(msg):
    vect_msg = vectorizer.transform([msg])
    pred = model.predict(vect_msg)
    return "Spam" if pred[0] == 1 else "Ham"

# Example
print(predict_message("You won a free prize, click here!"))
print(predict_message("Hey, are you coming to the meeting?"))