import tkinter as tk
from tkinter import filedialog, messagebox
import csv
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

class SentimentAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis App")
        
        self.label = tk.Label(self, text="Upload a CSV file:")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.upload_button = tk.Button(self, text="Upload", command=self.upload_file)
        self.upload_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.text = tk.Text(self, height=10, width=50)
        self.text.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

        self.analyze_button = tk.Button(self, text="Analyze", command=self.analyze_sentiment)
        self.analyze_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self.save_model_button = tk.Button(self, text="Save Model", command=self.save_model)
        self.save_model_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.bind("<Return>", lambda event: self.analyze_sentiment())

    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.text.delete("1.0", tk.END)
            self.text.insert(tk.END, f"File uploaded: {file_path}\n")
            self.filename = file_path

    def load_data(self):
        texts = []
        labels = []
        with open(self.filename, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                texts.append(row[0])
                labels.append(row[1])
        return {'texts': texts, 'labels': labels}

    def train_model(self, data):
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data['texts'])
        
        classifier = MultinomialNB()
        classifier.fit(X, data['labels'])

        features = vectorizer.get_feature_names_out()
        word_counts = X.toarray()
        word_sentiments = {'positive': {}, 'negative': {}, 'neutral': {}}

        for i, label in enumerate(data['labels']):
            for j, count in enumerate(word_counts[i]):
                word = features[j]
                if word not in word_sentiments[label]:
                    word_sentiments[label][word] = 0
                word_sentiments[label][word] += count

        total_word_counts = {sentiment: sum(word_sentiments[sentiment].values()) for sentiment in ['positive', 'negative', 'neutral']}
        word_percentages = {sentiment: {word: (count / total_word_counts[sentiment]) * 100 for word, count in word_sentiments[sentiment].items()} for sentiment in ['positive', 'negative', 'neutral']}
        
        return classifier, vectorizer, word_percentages

    def analyze_sentiment(self):
        try:
            data = self.load_data()
            classifier, vectorizer, word_percentages = self.train_model(data)
            
            output = ""
            output += "Words associated with positive sentiment:\n"
            for word, percentage in sorted(word_percentages['positive'].items(), key=lambda x: x[1], reverse=True)[:10]:
                output += f"{word}: {percentage:.2f}%\n"
            
            output += "\nWords associated with negative sentiment:\n"
            for word, percentage in sorted(word_percentages['negative'].items(), key=lambda x: x[1], reverse=True)[:10]:
                output += f"{word}: {percentage:.2f}%\n"

            output += "\nWords associated with neutral sentiment:\n"
            for word, percentage in sorted(word_percentages['neutral'].items(), key=lambda x: x[1], reverse=True)[:10]:
                output += f"{word}: {percentage:.2f}%\n"
            
            self.text.insert(tk.END, output)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_model(self):
        try:
            data = self.load_data()
            classifier, vectorizer, _ = self.train_model(data)
            filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
            if filename:
                with open(filename, 'wb') as file:
                    pickle.dump((classifier, vectorizer), file)
                messagebox.showinfo("Success", "Model saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

app = SentimentAnalysisApp()
app.mainloop()
