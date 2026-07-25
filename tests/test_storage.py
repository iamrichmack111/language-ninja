from pathlib import Path

from language_ninja.storage import Storage, export_records


def test_prediction_note_and_export(tmp_path: Path):
    db = Storage(tmp_path / "test.db")
    prediction_id = db.add_prediction("demo.pkl", "great", "positive", 0.9)
    db.add_note("review", "Looks correct", prediction_id)
    assert len(db.recent_predictions()) == 1
    assert len(db.recent_notes()) == 1
    out = export_records(db.recent_predictions(), tmp_path / "predictions.json", "json")
    assert out.exists()
