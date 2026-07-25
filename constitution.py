import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load and preprocess the training data
data = pd.read_csv('propositions.csv')

# Feature extraction (using TF-IDF)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Adjust max_features as needed
X_train_tfidf = tfidf_vectorizer.fit_transform(data['text'])
y_train = data['label']

# Train the sentiment analysis model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Define a function to preprocess new phrases and make predictions
def predict_sentiment(new_phrase):
    # Preprocess the new phrase
    new_phrase_tfidf = tfidf_vectorizer.transform([new_phrase])
    # Make prediction
    prediction = svm_model.predict(new_phrase_tfidf)
    return prediction[0]

# Example usage
new_phrase = "I love this product! It works amazingly well."
prediction = predict_sentiment(new_phrase)
print("Prediction:", prediction)

new_phrase = "I hate waiting in long lines. It's so frustrating."
prediction = predict_sentiment(new_phrase)
print("Prediction:", prediction)
