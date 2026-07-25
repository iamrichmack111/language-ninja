import sys
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

def load_data(filename):
    texts = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            texts.append(row[0])
            labels.append(row[1])
    return {'texts': texts, 'labels': labels}

def train_model(data):
    # Vectorize the text data
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['texts'])

    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X, data['labels'])

    # Get words associated with positive, negative, and neutral sentiments
    features = vectorizer.get_feature_names_out()
    word_counts = X.toarray()
    word_sentiments = {'positive': {}, 'negative': {}, 'neutral': {}}

    for i, label in enumerate(data['labels']):
        for j, count in enumerate(word_counts[i]):
            word = features[j]
            if word not in word_sentiments[label]:
                word_sentiments[label][word] = 0
            word_sentiments[label][word] += count

    # Calculate percentage of occurrence of each word for each sentiment
    total_word_counts = {sentiment: sum(word_sentiments[sentiment].values()) for sentiment in ['positive', 'negative', 'neutral']}
    word_percentages = {sentiment: {word: (count / total_word_counts[sentiment]) * 100 for word, count in word_sentiments[sentiment].items()} for sentiment in ['positive', 'negative', 'neutral']}

    return classifier, vectorizer, word_percentages

def main():
    # Check if the path to the CSV file is provided as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python train_sentiment_model.py <path_to_csv>")
        sys.exit(1)

    filename = sys.argv[1]

    # Load data from the CSV file
    data = load_data(filename)

    # Train the model
    classifier, vectorizer, word_percentages = train_model(data)

    # Write statistics about words and their association with sentiments to a text file
    with open('decision_logic.txt', 'w') as file:
        file.write("Words associated with positive sentiment:\n")
        for word, percentage in sorted(word_percentages['positive'].items(), key=lambda x: x[1], reverse=True)[:10]:
            file.write(f"{word}: {percentage:.2f}%\n")

        file.write("\nWords associated with negative sentiment:\n")
        for word, percentage in sorted(word_percentages['negative'].items(), key=lambda x: x[1], reverse=True)[:10]:
            file.write(f"{word}: {percentage:.2f}%\n")

        file.write("\nWords associated with neutral sentiment:\n")
        for word, percentage in sorted(word_percentages['neutral'].items(), key=lambda x: x[1], reverse=True)[:10]:
            file.write(f"{word}: {percentage:.2f}%\n")

    # Save the trained model to disk using pickle
    with open('sentiment_model.pkl', 'wb') as file:
        pickle.dump((classifier, vectorizer), file)

if __name__ == "__main__":
    main()