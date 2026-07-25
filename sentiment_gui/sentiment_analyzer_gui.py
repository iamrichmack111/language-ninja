import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer

class SentimentAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis App")
        
        self.label_model = tk.Label(self, text="Select Model File:")
        self.label_model.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.button_model = tk.Button(self, text="Choose", command=self.choose_model)
        self.button_model.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        self.label_input = tk.Label(self, text="Enter Text or Choose File:")
        self.label_input.grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.entry_input = tk.Entry(self, width=50)
        self.entry_input.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.button_file = tk.Button(self, text="Choose File", command=self.choose_file)
        self.button_file.grid(row=1, column=2, padx=5, pady=5, sticky="w")

        self.button_analyze = tk.Button(self, text="Analyze Sentiment", command=self.analyze_sentiment)
        self.button_analyze.grid(row=2, column=0, columnspan=3, padx=5, pady=5)

        # Database connection
        self.connection = sqlite3.connect("sentiment_results.db")
        self.create_table()

    def create_table(self):
        cursor = self.connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS results
                          (timestamp TEXT, model TEXT, text TEXT, sentiment TEXT)''')
        self.connection.commit()

    def choose_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if model_path:
            self.model_path = model_path
            messagebox.showinfo("Info", "Model selected successfully.")

    def choose_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.entry_input.delete(0, tk.END)
                self.entry_input.insert(tk.END, file.read())

    def analyze_sentiment(self):
        try:
            input_text = self.entry_input.get()
            if not input_text:
                messagebox.showerror("Error", "Please enter text or choose a file.")
                return

            with open(self.model_path, 'rb') as file:
                classifier, vectorizer = pickle.load(file)

            X = vectorizer.transform([input_text])
            prediction = classifier.predict(X)
            sentiment = prediction[0]

            # Insert results into database
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_name = self.model_path.split('/')[-1]  # Extract model file name
            cursor = self.connection.cursor()
            cursor.execute('''INSERT INTO results (timestamp, model, text, sentiment) 
                              VALUES (?, ?, ?, ?)''', (timestamp, model_name, input_text, sentiment))
            self.connection.commit()

            messagebox.showinfo("Sentiment Analysis Result", f"Sentiment: {sentiment}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

app = SentimentAnalysisApp()
app.mainloop()
