Sure, here's an updated README with instructions for installing pickle and additional details about the sentiment analysis model and decision logic text:

---

# Sentiment Analysis and SQLite Query Tool

This project consists of three scripts: 
1. `sentiment_trainer.py`: A script to train a sentiment analysis model using a provided dataset and export the trained model as a `.pkl` file.
2. `sentiment_analyzer.py`: A script to load a trained sentiment analysis model from a `.pkl` file and analyze sentiment for user-provided text inputs or files.
3. `sqlite_query_tool.py`: A graphical user interface (GUI) tool to execute SQL queries on SQLite databases, view database schema, and export query results.

## Dependencies

The scripts require the following dependencies:

- Python 3.x
- tkinter (for GUI in `sqlite_query_tool.py`)
- scikit-learn (for machine learning operations)
- pandas (for data manipulation)
- numpy (for numerical computations)
- pickle (for serialization)

## Installation

You can install the dependencies using pip:

```bash
pip install scikit-learn pandas numpy
```

On Ubuntu or Debian-based systems, you can install Python 3 and tkinter using:

```bash
sudo apt update
sudo apt install python3 python3-tk
```

## How to Use

### 1. Sentiment Trainer (`sentiment_trainer.py`)

#### Instructions:
1. Place your dataset in CSV format with columns `text` and `label` (e.g., `"I love this product! It works amazingly well.",positive`).
2. Run the script with Python:

```bash
python sentiment_trainer.py your_dataset.csv
```

3. The script will train the model and save it as a `.pkl` file.

#### Decision Logic Text:
The `decision_logic.txt` file contains statistics about word usage in the dataset, including percentages of positive, negative, and neutral words. Each line of the file corresponds to a word, followed by its usage statistics.

### 2. Sentiment Analyzer (`sentiment_analyzer.py`)

#### Instructions:
1. Provide the path to the trained model `.pkl` file and input text or file to analyze sentiment.
2. Run the script with Python:

```bash
python sentiment_analyzer.py path_to_model.pkl "Your text goes here"
```

3. The script will load the model and display the sentiment analysis results.

### 3. SQLite Query Tool (`sqlite_query_tool.py`)

#### Instructions:
1. Run the script with Python:

```bash
python sqlite_query_tool.py
```

2. Choose a SQLite database file (.db).
3. Execute SQL queries, view database schema, and export query results using the GUI.

## Additional Notes

- For `sentiment_trainer.py`, ensure your dataset is properly formatted with text and corresponding labels.
- For `sqlite_query_tool.py`, SQLite database files (.db) are supported.
- Make sure to have appropriate permissions to read and write files.
- The sentiment analysis model is trained using scikit-learn's `CountVectorizer` and `MultinomialNB` classifier. It learns to classify text into positive, negative, or neutral sentiment categories based on the provided dataset.
- The `decision_logic.txt` file contains statistics about word usage in the dataset, which helps understand how the model makes its predictions.

---

Feel free to customize the README according to your project's specific details and requirements!