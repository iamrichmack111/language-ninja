from __future__ import annotations

import csv
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)

from .ml import load_dataset, load_model, predict, save_model, train_and_evaluate
from .storage import Storage, export_records, export_table


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "exports"
DB_PATH = ROOT / "language_ninja.db"
DEFAULT_DATASET = DATA_DIR / "propositions.csv"
DEFAULT_MODEL = MODEL_DIR / "sentiment_model.pkl"


class LanguageNinja(App):
    TITLE = "🥷 Language Trainer Ninja"
    SUB_TITLE = "NLP Training • Analysis • Notes • Export"

    CSS = """
    Screen { layout: vertical; }
    TabPane { padding: 1 2; }
    .section { height: auto; margin-bottom: 1; }
    .panel { border: round $primary; padding: 1; height: auto; }
    .instructions { border: round $secondary; padding: 1; margin-bottom: 1; height: auto; }
    .grow { height: 1fr; }
    .status { height: 7; border: round $accent; padding: 1; }
    #train_status { height: 14; }
    .compact { width: 1fr; }
    Button { margin-right: 1; }
    DataTable { height: 1fr; }
    TextArea { height: 10; }
    #analysis_input { height: 8; }
    #note_body { height: 12; }
    #sql_input { height: 5; }
    """

    BINDINGS = [
        ("a", "tab('analyze')", "Analyze"),
        ("t", "tab('train')", "Train"),
        ("d", "tab('dataset')", "Dataset"),
        ("n", "tab('notes')", "Notes"),
        ("h", "tab('history')", "History"),
        ("e", "tab('export')", "Export"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.storage = Storage(DB_PATH)
        self.classifier = None
        self.vectorizer = None
        self.active_model = DEFAULT_MODEL
        self.last_prediction_id: int | None = None
        self.query_columns: list[str] = []
        self.query_rows: list[tuple] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="analyze", id="tabs"):
            with TabPane("Analyze", id="analyze"):
                yield Static(
                    "HOW TO USE • Type or paste a sentence below, then choose Analyze. "
                    "The highest calibrated probability becomes the prediction. HIGH means 75%+, MODERATE 55–74%, and LOW below 55%. "
                    "All three probabilities are saved to History so uncertainty stays visible.",
                    classes="instructions",
                )
                yield Label("Text to analyze")
                yield TextArea(id="analysis_input")
                with Horizontal(classes="section"):
                    yield Button("Analyze", id="analyze_btn", variant="primary")
                    yield Button("Clear", id="clear_analysis")
                yield Static("Ready. Train or load a model to begin.", id="analysis_result", classes="status")

            with TabPane("Train", id="train"):
                yield Static(
                    "HOW TO USE • Choose a CSV containing text,label columns and select Train Model. "
                    "The v2.2 trainer uses word/phrase TF-IDF + character TF-IDF feeding a calibrated Linear SVM. "
                    "It evaluates on a stratified holdout set, reports per-class quality, then retrains the active model on all rows.",
                    classes="instructions",
                )
                yield Label("Dataset path")
                yield Input(str(DEFAULT_DATASET), id="dataset_path")
                yield Label("Model output path")
                yield Input(str(DEFAULT_MODEL), id="model_path")
                with Horizontal(classes="section"):
                    yield Button("Train Model", id="train_btn", variant="primary")
                    yield Button("Load Model", id="load_model_btn")
                yield Static("No training run yet.", id="train_status", classes="status")

            with TabPane("Dataset", id="dataset"):
                yield Static(
                    "HOW TO USE • Review the examples that teach the model. Each row needs text and a sentiment label. "
                    "The bundled corpus contains 1,500 balanced conversational examples, including negation and mixed-language cases. "
                    "Refresh after editing the CSV externally, or export a copy for backup.",
                    classes="instructions",
                )
                with Horizontal(classes="section"):
                    yield Button("Refresh", id="dataset_refresh", variant="primary")
                    yield Button("Export CSV", id="dataset_export_csv")
                    yield Button("Export JSON", id="dataset_export_json")
                yield DataTable(id="dataset_table", zebra_stripes=True)

            with TabPane("Notes", id="notes"):
                yield Static(
                    "HOW TO USE • Record observations, experiments, or corrections. Save Note stores a general note. "
                    "Link to Last Prediction attaches the note to the most recent analysis so you can document mistakes or edge cases.",
                    classes="instructions",
                )
                yield Label("Title")
                yield Input(placeholder="Experiment, correction, observation...", id="note_title")
                yield Label("Note")
                yield TextArea(id="note_body")
                with Horizontal(classes="section"):
                    yield Button("Save Note", id="save_note", variant="primary")
                    yield Button("Link to Last Prediction", id="link_note")
                    yield Button("Export Notes", id="notes_export")
                yield DataTable(id="notes_table", zebra_stripes=True)

            with TabPane("History", id="history"):
                yield Static(
                    "HOW TO USE • Review past predictions, confidence scores, and the model used. "
                    "Use this page to spot repeated low-confidence cases and export prediction history for deeper analysis.",
                    classes="instructions",
                )
                with Horizontal(classes="section"):
                    yield Button("Refresh", id="history_refresh", variant="primary")
                    yield Button("Export CSV", id="history_export_csv")
                    yield Button("Export JSON", id="history_export_json")
                yield DataTable(id="history_table", zebra_stripes=True)

            with TabPane("Database", id="database"):
                yield Static(
                    "HOW TO USE • Run read-only SELECT queries against the app database. "
                    "Use this for advanced inspection of predictions and notes. Query results can be exported to CSV or JSON.",
                    classes="instructions",
                )
                yield Label("Read-only SQL")
                yield TextArea("SELECT * FROM predictions ORDER BY id DESC LIMIT 25;", id="sql_input")
                with Horizontal(classes="section"):
                    yield Button("Run Query", id="query_run", variant="primary")
                    yield Button("Export Query CSV", id="query_export_csv")
                    yield Button("Export Query JSON", id="query_export_json")
                yield DataTable(id="query_table", zebra_stripes=True)

            with TabPane("Export", id="export"):
                yield Static(
                    "HOW TO USE • Choose what you want to export and select CSV or JSON, then choose Export Now. "
                    "Files are written to the project's ./exports/ directory.",
                    classes="instructions",
                )
                yield Static(
                    "Exports are written to ./exports/\n\n"
                    "Available exports:\n"
                    "• Prediction history → CSV / JSON\n"
                    "• Notes → JSON\n"
                    "• Dataset → CSV / JSON\n"
                    "• Database query results → CSV / JSON\n",
                    classes="panel",
                )
                with Horizontal(classes="section"):
                    yield Select(
                        [("Predictions", "predictions"), ("Notes", "notes"), ("Dataset", "dataset")],
                        value="predictions",
                        id="export_source",
                    )
                    yield Select([("CSV", "csv"), ("JSON", "json")], value="csv", id="export_format")
                    yield Button("Export Now", id="export_now", variant="primary")
                yield Static("Nothing exported yet.", id="export_status", classes="status")
        yield Footer()

    def on_mount(self) -> None:
        self._setup_tables()
        self.refresh_dataset()
        self.refresh_history()
        self.refresh_notes()
        if self.active_model.exists():
            self._load_model(self.active_model)

    def _setup_tables(self) -> None:
        dataset = self.query_one("#dataset_table", DataTable)
        dataset.add_columns("#", "Text", "Label")
        history = self.query_one("#history_table", DataTable)
        history.add_columns("ID", "Time", "Text", "Result", "POS", "NEU", "NEG", "Level")
        notes = self.query_one("#notes_table", DataTable)
        notes.add_columns("ID", "Time", "Title", "Body", "Prediction")

    def action_tab(self, tab_id: str) -> None:
        self.query_one("#tabs", TabbedContent).active = tab_id

    def _notify_error(self, message: str) -> None:
        self.notify(message, severity="error", timeout=5)

    def _load_model(self, path: Path) -> None:
        try:
            self.classifier, self.vectorizer = load_model(path)
            self.active_model = path
            self.query_one("#train_status", Static).update(f"Loaded model: {path}")
        except Exception as exc:
            self._notify_error(f"Could not load model: {exc}")

    def refresh_dataset(self) -> None:
        table = self.query_one("#dataset_table", DataTable)
        table.clear()
        path = Path(self.query_one("#dataset_path", Input).value or DEFAULT_DATASET)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                next(reader, None)
                for index, row in enumerate(reader, start=1):
                    if len(row) >= 2:
                        table.add_row(str(index), row[0], row[1])
        except Exception as exc:
            self._notify_error(f"Dataset: {exc}")

    def refresh_history(self) -> None:
        table = self.query_one("#history_table", DataTable)
        table.clear()
        for row in self.storage.recent_predictions():
            table.add_row(
                str(row["id"]),
                row["timestamp"],
                row["text"],
                row["sentiment"].upper(),
                f'{row["positive_probability"]:.1%}',
                f'{row["neutral_probability"]:.1%}',
                f'{row["negative_probability"]:.1%}',
                row["confidence_level"],
            )

    def refresh_notes(self) -> None:
        table = self.query_one("#notes_table", DataTable)
        table.clear()
        for row in self.storage.recent_notes():
            table.add_row(
                str(row["id"]), row["timestamp"], row["title"], row["body"], str(row["linked_prediction_id"] or "")
            )

    def _export_path(self, stem: str, fmt: str) -> Path:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        return EXPORT_DIR / f"{stem}.{fmt}"

    def export_predictions(self, fmt: str) -> Path:
        return export_records(self.storage.recent_predictions(100000), self._export_path("predictions", fmt), fmt)

    def export_notes(self, fmt: str) -> Path:
        return export_records(self.storage.recent_notes(100000), self._export_path("notes", fmt), fmt)

    def export_dataset(self, fmt: str) -> Path:
        path = Path(self.query_one("#dataset_path", Input).value or DEFAULT_DATASET)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            records = list(reader)
        return export_records(records, self._export_path("dataset", fmt), fmt)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        try:
            if button_id == "analyze_btn":
                self.analyze_text()
            elif button_id == "clear_analysis":
                self.query_one("#analysis_input", TextArea).text = ""
                self.query_one("#analysis_result", Static).update("Ready.")
            elif button_id == "train_btn":
                self.train_active_dataset()
            elif button_id == "load_model_btn":
                self._load_model(Path(self.query_one("#model_path", Input).value))
            elif button_id == "dataset_refresh":
                self.refresh_dataset()
            elif button_id == "history_refresh":
                self.refresh_history()
            elif button_id == "save_note":
                self.save_note(link=False)
            elif button_id == "link_note":
                self.save_note(link=True)
            elif button_id == "notes_export":
                path = self.export_notes("json")
                self.notify(f"Exported {path}")
            elif button_id == "dataset_export_csv":
                self.notify(f"Exported {self.export_dataset('csv')}")
            elif button_id == "dataset_export_json":
                self.notify(f"Exported {self.export_dataset('json')}")
            elif button_id == "history_export_csv":
                self.notify(f"Exported {self.export_predictions('csv')}")
            elif button_id == "history_export_json":
                self.notify(f"Exported {self.export_predictions('json')}")
            elif button_id == "query_run":
                self.run_query()
            elif button_id == "query_export_csv":
                self.export_query("csv")
            elif button_id == "query_export_json":
                self.export_query("json")
            elif button_id == "export_now":
                self.export_selected()
        except Exception as exc:
            self._notify_error(str(exc))

    def train_active_dataset(self) -> None:
        dataset_path = Path(self.query_one("#dataset_path", Input).value)
        model_path = Path(self.query_one("#model_path", Input).value)
        texts, labels = load_dataset(dataset_path)
        classifier, vectorizer, report = train_and_evaluate(texts, labels)
        save_model(model_path, classifier, vectorizer)
        self.classifier, self.vectorizer = classifier, vectorizer
        self.active_model = model_path
        if report.accuracy is None:
            metrics = "Evaluation: add more examples per class for a reliable holdout score"
            per_class = ""
        else:
            metrics = f"Holdout accuracy: {report.accuracy:.1%} • Macro F1: {report.macro_f1:.1%}"
            per_class = "\n".join(
                f"{label.upper():8} P {values['precision']:.1%}  R {values['recall']:.1%}  F1 {values['f1']:.1%}"
                for label, values in report.class_metrics.items()
            )
        self.query_one("#train_status", Static).update(
            "TRAINING COMPLETE\n"
            f"Examples: {report.rows} • {report.labels}\n"
            f"Model: {report.algorithm}\n"
            f"Features: {report.feature_count:,} • Word TF-IDF 1–3 grams • Char TF-IDF 3–5 grams\n"
            f"{metrics}\n{per_class}\n"
            f"Saved: {model_path}"
        )
        self.notify("Model trained and activated.")

    def analyze_text(self) -> None:
        text = self.query_one("#analysis_input", TextArea).text.strip()
        if not text:
            raise ValueError("Enter text to analyze.")
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Train or load a model first.")
        result = predict(text, self.classifier, self.vectorizer)
        probabilities = result.probabilities
        self.query_one("#analysis_result", Static).update(
            f"Prediction: {result.label.upper()} • {result.confidence_level} CONFIDENCE\n"
            f"Winning probability: {result.confidence:.1%}\n"
            f"POS {probabilities.get('positive', 0.0):.1%}   "
            f"NEU {probabilities.get('neutral', 0.0):.1%}   "
            f"NEG {probabilities.get('negative', 0.0):.1%}"
        )
        self.last_prediction_id = self.storage.add_prediction(
            self.active_model.name,
            text,
            result.label,
            result.confidence,
            probabilities=result.probabilities,
            confidence_level=result.confidence_level,
        )
        self.refresh_history()

    def save_note(self, link: bool) -> None:
        title = self.query_one("#note_title", Input).value.strip() or "Untitled note"
        body = self.query_one("#note_body", TextArea).text.strip()
        if not body:
            raise ValueError("Write a note before saving.")
        linked = self.last_prediction_id if link else None
        if link and linked is None:
            raise ValueError("There is no prediction to link this note to yet.")
        self.storage.add_note(title, body, linked)
        self.query_one("#note_title", Input).value = ""
        self.query_one("#note_body", TextArea).text = ""
        self.refresh_notes()
        self.notify("Note saved.")

    def run_query(self) -> None:
        query = self.query_one("#sql_input", TextArea).text.strip()
        columns, rows = self.storage.execute_read_query(query)
        self.query_columns = list(columns)
        self.query_rows = [tuple(row) for row in rows]
        table = self.query_one("#query_table", DataTable)
        table.clear(columns=True)
        if columns:
            table.add_columns(*columns)
        for row in rows:
            table.add_row(*[str(value) for value in row])
        self.notify(f"Query returned {len(rows)} row(s).")

    def export_query(self, fmt: str) -> None:
        if not self.query_columns:
            self.run_query()
        path = export_table(self.query_columns, self.query_rows, self._export_path("query-results", fmt), fmt)
        self.notify(f"Exported {path}")

    def export_selected(self) -> None:
        source = self.query_one("#export_source", Select).value
        fmt = self.query_one("#export_format", Select).value
        if source == "predictions":
            path = self.export_predictions(str(fmt))
        elif source == "notes":
            path = self.export_notes(str(fmt))
        else:
            path = self.export_dataset(str(fmt))
        self.query_one("#export_status", Static).update(f"Export complete:\n{path}")
        self.notify(f"Exported {path}")


def main() -> None:
    LanguageNinja().run()


if __name__ == "__main__":
    main()
