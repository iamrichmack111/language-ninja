from pathlib import Path

import language_ninja.paths as paths


def test_bootstrap_copies_packaged_assets(tmp_path, monkeypatch):
    app_dir = tmp_path / "user-data"
    monkeypatch.setattr(paths, "APP_DIR", app_dir)
    monkeypatch.setattr(paths, "DATA_DIR", app_dir / "data")
    monkeypatch.setattr(paths, "MODEL_DIR", app_dir / "models")
    monkeypatch.setattr(paths, "EXPORT_DIR", app_dir / "exports")
    monkeypatch.setattr(paths, "DB_PATH", app_dir / "language_ninja.db")
    monkeypatch.setattr(paths, "DEFAULT_DATASET", app_dir / "data" / "propositions.csv")
    monkeypatch.setattr(paths, "DEFAULT_MODEL", app_dir / "models" / "sentiment_model.pkl")

    legacy = tmp_path / "empty-launch-directory"
    legacy.mkdir()
    paths.bootstrap_user_files(legacy)

    assert paths.DEFAULT_DATASET.is_file()
    assert paths.DEFAULT_MODEL.is_file()
    assert paths.EXPORT_DIR.is_dir()
