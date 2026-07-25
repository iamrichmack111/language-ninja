# 🥷 Language Ninja

<p align="center">
  <strong>Terminal-Based NLP Sentiment Analysis, Training & Model Evaluation Workstation</strong>
</p>

<p align="center">
  🧠 Machine Learning • 🖥️ Textual TUI • 📊 Model Evaluation • 🗃️ SQLite • 📦 PyPI
</p>

---

## 🚀 Overview

**Language Ninja** is a keyboard-first terminal application for training, testing, analyzing, and evaluating NLP sentiment models.

Instead of hiding machine-learning behavior behind a simple prediction API, Language Ninja exposes the entire workflow inside an interactive **Textual TUI**.

The application can:

- 🧠 Train sentiment-analysis models
- 🔍 Analyze natural-language input
- 📊 Display class probabilities
- 🟢 Detect positive sentiment
- ⚪ Detect neutral sentiment
- 🔴 Detect negative sentiment
- 📚 Browse the training dataset
- 📝 Store research and prediction notes
- 🕘 Maintain prediction history
- 🗃️ Query the SQLite database
- 📤 Export data to CSV and JSON
- 📈 Display model evaluation metrics
- 💾 Persist prediction results
- 🔄 Retrain models from the terminal
- 📦 Install directly from PyPI

Language Ninja is designed as both a practical NLP utility and an educational environment for understanding how classical machine-learning pipelines behave on real language.

---

# ✨ Features

## 🧠 Sentiment Analysis

Enter natural-language text directly into the terminal:

```text
I absolutely love this application
```

Language Ninja analyzes the sentence and returns:

```text
Prediction: POSITIVE

Positive    98.4%
Neutral      0.8%
Negative     0.8%

Confidence: HIGH
```

Instead of showing only the winning class, Language Ninja exposes the model's probability distribution.

This makes ambiguous predictions easier to understand.

---

## 🎯 Three Sentiment Classes

Language Ninja currently classifies text into:

```text
POSITIVE
NEUTRAL
NEGATIVE
```

Examples:

```text
"This application is fantastic."
→ POSITIVE

"The application opened at nine this morning."
→ NEUTRAL

"This application is frustrating to use."
→ NEGATIVE
```

---

# 🤖 Machine Learning Pipeline

Language Ninja v2.2 uses a classical NLP pipeline built with **scikit-learn**.

The architecture combines word-level and character-level language features.

```text
                    ┌──────────────────┐
                    │    Input Text    │
                    └────────┬─────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
      ┌─────────────────┐         ┌─────────────────┐
      │ Word TF-IDF     │         │ Character TF-IDF│
      │ 1–3 grams       │         │ 3–5 grams       │
      └────────┬────────┘         └────────┬────────┘
               │                           │
               └─────────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Feature Union    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Linear SVM     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Calibration    │
                    └────────┬─────────┘
                             │
                             ▼
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
           POSITIVE        NEUTRAL       NEGATIVE
```

---

## 🔤 Word TF-IDF

Word TF-IDF learns useful words and phrases.

For example:

```text
love
really love
absolutely love

hate
really hate
absolutely hate

not good
not bad
don't like
```

Language Ninja uses word n-grams so the model can learn relationships between neighboring words rather than evaluating every word independently.

---

## 🔡 Character TF-IDF

Character-level TF-IDF helps the model recognize patterns inside words.

This improves robustness for:

- contractions
- spelling variations
- related word forms
- punctuation differences
- informal language

For example:

```text
like
liked
likes
liking
```

share character-level patterns.

---

## ⚔️ Linear SVM

The primary classifier is a **Linear Support Vector Machine**.

Linear SVMs work particularly well with high-dimensional sparse NLP features such as TF-IDF vectors.

The classifier learns decision boundaries separating:

```text
positive
neutral
negative
```

language patterns.

---

## 🎚️ Probability Calibration

Raw SVM output is not naturally a probability.

Language Ninja therefore uses probability calibration so the TUI can display:

```text
POS  82.4%
NEU  10.7%
NEG   6.9%
```

rather than only:

```text
positive
```

This provides much more information about model uncertainty.

---

# 📚 Training Corpus

Language Ninja v2.2 includes a starter sentiment corpus containing approximately:

```text
1,500 training examples
```

balanced across:

```text
500 positive
500 neutral
500 negative
```

The corpus includes:

- direct sentiment
- conversational language
- negation
- mild sentiment
- strong sentiment
- neutral statements
- informal phrases
- mixed linguistic constructions

Example training data:

```csv
text,sentiment
I absolutely love this application,positive
This software works extremely well,positive
The application opened this morning,neutral
I am not sure how I feel about this,neutral
This interface is frustrating,negative
I absolutely hate this application,negative
```

> ⚠️ The bundled corpus is intended as a strong starter and demonstration dataset. Internal holdout performance should not be interpreted as guaranteed real-world accuracy.

---

# 🖥️ Terminal Interface

Language Ninja uses the Python **Textual** framework.

The application is organized into several interactive workspaces:

```text
┌─────────────────────────────────────────────────────────────┐
│ 🥷 LANGUAGE TRAINER NINJA                                  │
├─────────────────────────────────────────────────────────────┤
│ Analyze │ Train │ Dataset │ Notes │ History │ Database │ Export
└─────────────────────────────────────────────────────────────┘
```

Every workspace contains instructions explaining its purpose and workflow.

---

# 🔍 Analyze

The **Analyze** page is the primary prediction interface.

Enter text:

```text
I love this application
```

Then select:

```text
ANALYZE
```

Language Ninja displays:

```text
Prediction
──────────
POSITIVE

Confidence
──────────
98.4%

Probabilities
─────────────
Positive    98.4%
Neutral      0.8%
Negative     0.8%
```

Predictions are automatically stored in the history database.

---

# 🧪 Train

The **Train** workspace retrains the NLP model.

The default dataset is:

```text
data/propositions.csv
```

The trained model is saved to:

```text
models/sentiment_model.pkl
```

Select:

```text
TRAIN MODEL
```

Language Ninja performs:

```text
Load Dataset
      ↓
Validate Labels
      ↓
Stratified Train/Test Split
      ↓
TF-IDF Feature Extraction
      ↓
Linear SVM Training
      ↓
Probability Calibration
      ↓
Model Evaluation
      ↓
Save Model
```

---

# 📊 Training Metrics

After training, Language Ninja reports metrics such as:

```text
Examples
Accuracy
Macro F1
Precision
Recall
Feature Count
Per-Class F1
```

For example:

```text
Training Complete

Examples:        1500
Positive:         500
Neutral:          500
Negative:         500

Algorithm:
Calibrated Linear SVM

Word Features:
TF-IDF 1–3 grams

Character Features:
TF-IDF 3–5 grams

Accuracy:        99.x%
Macro F1:        99.x%
```

Actual metrics depend on the dataset used for training.

---

# 📖 Dataset Browser

The **Dataset** page provides a terminal-native view of training examples.

Example:

```text
┌──────┬────────────────────────────────────────────┬──────────┐
│ ID   │ Text                                       │ Label    │
├──────┼────────────────────────────────────────────┼──────────┤
│ 001  │ I absolutely love this                     │ positive │
│ 002  │ The application opened this morning        │ neutral  │
│ 003  │ This interface is terrible                 │ negative │
└──────┴────────────────────────────────────────────┴──────────┘
```

This makes it possible to inspect what the model is actually learning from.

---

# 🧠 Improving the Model

One of the most useful ways to improve Language Ninja is to collect difficult examples.

Suppose the model incorrectly classifies:

```text
I don't know if I like this app
```

as:

```text
POSITIVE
```

You can add the corrected example to:

```text
data/propositions.csv
```

as:

```csv
I don't know if I like this app,neutral
```

Then retrain the model.

This creates a simple human-guided improvement loop:

```text
Predict
   ↓
Review
   ↓
Find Failure
   ↓
Correct Label
   ↓
Add Training Example
   ↓
Retrain
   ↓
Evaluate
   ↓
Repeat
```

---

# 🧩 Hard Language Examples

Useful training examples include ambiguous and negated language:

```text
I don't know if I like it
I'm not sure I like it
I guess it's okay
I kind of like it
I don't hate it
It's not good
It's not bad
I wanted to like it
I like it but it's frustrating
I hate the UI but love the features
```

These examples are especially useful because simple sentiment models often struggle with negation and mixed language.

---

# 📝 Notes

Language Ninja includes a persistent notes system.

Notes can be used for:

- model observations
- difficult predictions
- training ideas
- dataset issues
- NLP experiments
- evaluation findings

A note can also be associated with the most recent prediction.

Example:

```text
Title:
Negation Test

Note:
The model classified "I don't hate it" as negative.
Add additional negation examples before the next training run.
```

Notes are stored in SQLite.

---

# 🕘 Prediction History

Every analysis can be recorded in the History workspace.

The newer database format stores:

```text
Text
Prediction
Confidence
Positive Probability
Neutral Probability
Negative Probability
Model
Timestamp
```

Example:

```text
TEXT                    RESULT      POS     NEU     NEG     LEVEL
───────────────────────────────────────────────────────────────
I love this app         positive    98.8%    0.7%    0.5%   HIGH
The app opened today    neutral      1.2%   97.9%    0.9%   HIGH
I hate this app         negative     0.4%    0.7%   98.9%   HIGH
```

This makes the History screen useful for model evaluation rather than functioning only as a log.

---

# 🗃️ SQLite Database

Language Ninja uses SQLite for persistent application data.

The database can store:

- predictions
- probability distributions
- notes
- model information
- timestamps

The **Database** workspace also provides a read-only SQL interface for exploring application data.

Example:

```sql
SELECT sentiment, COUNT(*)
FROM predictions
GROUP BY sentiment;
```

Possible result:

```text
positive    42
neutral     31
negative    27
```

---

# 📤 Export

Language Ninja supports exporting application data.

Supported formats include:

```text
CSV
JSON
```

Exportable data includes:

- predictions
- notes
- datasets
- SQL query results

This makes Language Ninja data easy to move into:

- pandas
- Jupyter
- spreadsheets
- visualization tools
- external ML workflows

---

# 📦 Installation

## Install from PyPI

Language Ninja is available on PyPI.

```bash
python3 -m pip install language-ninja
```

Launch it with:

```bash
language-ninja
```

---

## 🐍 Recommended Virtual Environment

On macOS and modern Linux distributions, using a virtual environment is recommended.

```bash
python3 -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

Install:

```bash
python -m pip install --upgrade pip
python -m pip install language-ninja
```

Run:

```bash
language-ninja
```

---

# 🛠️ Install From Source

Clone the repository:

```bash
git clone https://github.com/iamrichmack111/language-ninja.git
```

Enter it:

```bash
cd language-ninja
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Run:

```bash
language-ninja
```

---

# 🧪 Development Installation

Install development/testing dependencies as needed:

```bash
python -m pip install pytest build twine
```

Run tests:

```bash
pytest -v
```

Build the package:

```bash
python -m build
```

Validate distributions:

```bash
python -m twine check dist/*
```

---

# 📁 Project Architecture

```text
language-ninja/
│
├── .github/
│   └── workflows/
│       └── publish.yml
│
├── data/
│   └── propositions.csv
│
├── models/
│   └── sentiment_model.pkl
│
├── language_ninja/
│   ├── __init__.py
│   ├── app.py
│   ├── database.py
│   ├── config.py
│   │
│   ├── ml/
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   ├── metrics.py
│   │   └── explain.py
│   │
│   └── screens/
│       ├── analyze.py
│       ├── train.py
│       ├── dataset.py
│       ├── notes.py
│       ├── history.py
│       ├── database.py
│       └── export.py
│
├── tests/
│
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

The exact source layout may evolve as the project grows.

---

# 🏗️ Architecture Philosophy

Language Ninja separates the application into several layers:

```text
┌──────────────────────────────┐
│        Textual TUI           │
├──────────────────────────────┤
│ Application / Workflow Layer │
├──────────────────────────────┤
│       NLP / ML Layer         │
├──────────────────────────────┤
│     Persistence Layer        │
├──────────────────────────────┤
│ SQLite │ CSV │ Model Files   │
└──────────────────────────────┘
```

The goal is to keep machine-learning logic independent from presentation logic.

This makes it easier to:

- test models
- replace classifiers
- add new datasets
- build APIs later
- create alternate interfaces
- benchmark algorithms

---

# ⌨️ Keyboard-First Design

Language Ninja is designed to work naturally from the keyboard.

Common navigation includes:

```text
A       Analyze
T       Train
D       Dataset
H       History
/       Search
Tab     Move between controls
Enter   Activate
Q       Quit
```

Available bindings may vary by screen and version.

The Textual footer displays active shortcuts.

---

# 🔬 Why Classical NLP?

Language Ninja intentionally uses classical machine learning rather than requiring a large language model.

This provides several advantages:

```text
⚡ Fast inference
💾 Small models
🖥️ CPU friendly
🌐 No cloud API required
🔒 Local processing
📚 Easier model inspection
🧪 Fast retraining
📦 Simple deployment
```

The application can run entirely on a local machine.

No GPU is required.

---

# 🧠 Model Confidence

Confidence does **not** mean overall model accuracy.

For example:

```text
Prediction: POSITIVE
Confidence: 72%
```

means the classifier assigned approximately 72% of its calibrated probability to the positive class for that particular prediction.

It does not mean:

```text
The model is 72% accurate.
```

Language Ninja exposes all class probabilities so uncertainty remains visible.

Example:

```text
Positive    45%
Neutral     42%
Negative    13%
```

The winning prediction is positive, but the model is clearly uncertain between positive and neutral.

---

# ⚠️ Model Limitations

Sentiment analysis is inherently ambiguous.

Statements can contain:

- sarcasm
- irony
- mixed sentiment
- cultural context
- implied meaning
- negation
- domain-specific vocabulary

For example:

```text
Great. Another update that broke everything.
```

contains the positive word `Great`, while the actual sentiment is likely negative.

Language Ninja should therefore be treated as an experimental NLP classifier rather than an authority on human intent.

---

# 🔮 Roadmap

Potential future improvements include:

- 🧑‍🏫 Human-in-the-loop correction directly in the TUI
- ➕ Add training examples from Analyze
- 🔄 One-key correction and retraining
- 🟣 Mixed sentiment class
- 📊 Confusion matrix visualization
- 📈 Training-run comparison
- 🧠 Model selection
- 🧪 Benchmark multiple classifiers
- 🔎 Prediction explanation
- 🧬 Feature importance inspection
- 📚 Dataset editor
- 🏷️ Custom classification labels
- 🌍 Additional languages
- 🔌 REST API
- 🐳 Docker support
- 📊 Experiment tracking
- 🧾 Model metadata/versioning

---

# 🔄 CI/CD

Language Ninja uses GitHub Actions for package builds and PyPI publication.

The release pipeline is:

```text
Code
  ↓
Git Commit
  ↓
GitHub
  ↓
Version Tag
  ↓
GitHub Release
  ↓
GitHub Actions
  ↓
Build Wheel + Source Distribution
  ↓
PyPI Trusted Publishing
  ↓
PyPI
  ↓
pip install language-ninja
```

PyPI publication uses **Trusted Publishing / OIDC**, avoiding long-lived PyPI API tokens in the repository.

---

# 🏷️ Version

Current release:

```text
v2.2.0
```

Check installed versions:

```bash
python3 -m pip index versions language-ninja
```

Upgrade:

```bash
python3 -m pip install --upgrade language-ninja
```

---

# 📦 PyPI

Install:

```bash
pip install language-ninja
```

Package:

```text
language-ninja
```

CLI:

```text
language-ninja
```

---

# 🌐 GitHub

Repository:

```text
iamrichmack111/language-ninja
```

Clone:

```bash
git clone https://github.com/iamrichmack111/language-ninja.git
```

---

# 🛡️ Privacy

Language Ninja is designed around local processing.

The classical NLP model does not require sending prediction text to an external AI service.

Your local deployment can therefore perform:

```text
Text
 ↓
Local Vectorizer
 ↓
Local Model
 ↓
Local Prediction
 ↓
Local SQLite
```

without requiring a remote inference API.

---

# 🤝 Contributing

Contributions, experiments, bug reports, and dataset improvements are welcome.

A typical development workflow:

```bash
git clone https://github.com/iamrichmack111/language-ninja.git
cd language-ninja

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -e .
pytest -v
```

Create a branch:

```bash
git checkout -b feature/my-improvement
```

Make changes, test, and submit a pull request.

---

# 📜 License

See the repository's `LICENSE` file for licensing information.

---

# 🥷 Language Ninja

**Train language. Test assumptions. Inspect uncertainty. Improve the model.**

```text
INPUT
  ↓
LEARN
  ↓
PREDICT
  ↓
MEASURE
  ↓
CORRECT
  ↓
RETRAIN
  ↓
IMPROVE
```

Built with:

**Python • Textual • scikit-learn • SQLite • TF-IDF • Linear SVM**

