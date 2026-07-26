from __future__ import annotations

import shutil
from importlib.resources import as_file, files
from pathlib import Path

from platformdirs import user_data_path


APP_NAME = "language-ninja"
APP_DIR = Path(user_data_path(APP_NAME, appauthor=False))
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
EXPORT_DIR = APP_DIR / "exports"
DB_PATH = APP_DIR / "language_ninja.db"
DEFAULT_DATASET = DATA_DIR / "propositions.csv"
DEFAULT_MODEL = MODEL_DIR / "sentiment_model.pkl"


def _copy_resource(resource_name: str, destination: Path) -> None:
    resource = files("language_ninja").joinpath("assets", resource_name)
    with as_file(resource) as source:
        shutil.copy2(source, destination)


def bootstrap_user_files(legacy_root: str | Path | None = None) -> None:
    """Create persistent per-user files without depending on the launch directory.

    Existing user files are never overwritten. On first run we prefer an older
    project-local file when one exists, which makes upgrades from v2.2 smoother;
    otherwise the packaged starter asset is copied into the user data directory.
    """
    APP_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    legacy = Path(legacy_root or Path.cwd())

    if not DEFAULT_DATASET.exists():
        old_dataset = legacy / "data" / "propositions.csv"
        if old_dataset.is_file() and old_dataset.resolve() != DEFAULT_DATASET.resolve():
            shutil.copy2(old_dataset, DEFAULT_DATASET)
        else:
            _copy_resource("propositions.csv", DEFAULT_DATASET)

    if not DEFAULT_MODEL.exists():
        old_model = legacy / "models" / "sentiment_model.pkl"
        if old_model.is_file() and old_model.resolve() != DEFAULT_MODEL.resolve():
            shutil.copy2(old_model, DEFAULT_MODEL)
        else:
            _copy_resource("sentiment_model.pkl", DEFAULT_MODEL)

    if not DB_PATH.exists():
        old_db = legacy / "language_ninja.db"
        if old_db.is_file() and old_db.resolve() != DB_PATH.resolve():
            shutil.copy2(old_db, DB_PATH)
