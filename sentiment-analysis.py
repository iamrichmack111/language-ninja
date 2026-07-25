import sys
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def load_model(model_path):
    with open(model_path, 'rb') as file:
        classifier, vectorizer = pickle.load(file)
    return classifier, vectorizer

def analyze_sentiment(text, classifier, vectorizer):
    X = vectorizer.transform([text])
    prediction = classifier.predict(X)
    return prediction[0]

def main():
    # Check if the path to the model and input text/file are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python sentiment_analysis.py <model_path> <input_text_or_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    input_text_or_file = sys.argv[2]

    # Load the trained model
    classifier, vectorizer = load_model(model_path)

    # Determine if the input is a file or text
    if input_text_or_file.endswith('.txt'):
        with open(input_text_or_file, 'r', encoding='utf-8') as file:
            input_text = file.read()
    else:
        input_text = input_text_or_file

    # Analyze sentiment
    sentiment = analyze_sentiment(input_text, classifier, vectorizer)
    print("Sentiment:", sentiment)

if __name__ == "__main__":
    main()
