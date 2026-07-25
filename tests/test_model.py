from pathlib import Path

from language_ninja.ml import load_dataset, load_model, predict


def test_bundled_dataset_is_balanced():
    texts, labels = load_dataset(Path('data/propositions.csv'))
    assert len(texts) == 1500
    assert labels.count('positive') == 500
    assert labels.count('neutral') == 500
    assert labels.count('negative') == 500


def test_bundled_model_handles_baselines():
    classifier, vectorizer = load_model(Path('models/sentiment_model.pkl'))
    cases = [
        ('I absolutely love this application', 'positive'),
        ('I absolutely hate this application', 'negative'),
        ('The application opened at nine this morning', 'neutral'),
        ('You should speak more like a human instead of a robot', 'negative'),
        ('You did well, your little business is alright', 'positive'),
    ]
    for text, expected in cases:
        result = predict(text, classifier, vectorizer)
        assert result.label == expected
        assert result.confidence >= 0.75
