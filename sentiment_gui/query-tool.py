import tkinter as tk
from tkinter import filedialog, messagebox
import sqlite3
from tkinter.scrolledtext import ScrolledText
import csv
import json

class QueryTool(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SQLite Query Tool")

        self.db_label_var = tk.StringVar()
        self.db_label_var.set("No database selected")
        self.label_selected_db = tk.Label(self, textvariable=self.db_label_var)
        self.label_selected_db.grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        self.button_select_db = tk.Button(self, text="Choose Database", command=self.choose_database)
        self.button_select_db.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        self.label_schema = tk.Label(self, text="Schema:")
        self.label_schema.grid(row=2, column=0, padx=5, pady=5, sticky="w")

        self.text_schema = ScrolledText(self, height=10, width=50)
        self.text_schema.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.label_query = tk.Label(self, text="Enter Query:")
        self.label_query.grid(row=4, column=0, padx=5, pady=5, sticky="w")

        self.entry_query = tk.Entry(self, width=50)
        self.entry_query.grid(row=4, column=1, columnspan=2, padx=5, pady=5, sticky="w")

        self.button_execute = tk.Button(self, text="Execute Query", command=self.execute_query)
        self.button_execute.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

        self.button_export = tk.Button(self, text="Export Results", command=self.export_results)
        self.button_export.grid(row=6, column=0, columnspan=3, padx=5, pady=5)

        self.export_format = tk.StringVar()
        self.export_format.set("csv")

        self.radio_csv = tk.Radiobutton(self, text="CSV", variable=self.export_format, value="csv")
        self.radio_csv.grid(row=7, column=0, padx=5, pady=5)

        self.radio_json = tk.Radiobutton(self, text="JSON", variable=self.export_format, value="json")
        self.radio_json.grid(row=7, column=1, padx=5, pady=5)

        self.radio_tabbed = tk.Radiobutton(self, text="Tabbed View", variable=self.export_format, value="tabbed")
        self.radio_tabbed.grid(row=7, column=2, padx=5, pady=5)

        self.label_results = tk.Label(self, text="Query Results:")
        self.label_results.grid(row=8, column=0, columnspan=3, padx=5, pady=5, sticky="w")

        self.text_results = ScrolledText(self, height=10, width=80)
        self.text_results.grid(row=9, column=0, columnspan=3, padx=5, pady=5)

        self.selected_db_file = None

    def choose_database(self):
        db_file = filedialog.askopenfilename(filetypes=[("SQLite databases", "*.db")])
        if db_file:
            self.selected_db_file = db_file
            self.db_label_var.set(f"Selected Database: {db_file}")
            self.get_schema()

    def get_schema(self):
        try:
            self.text_schema.delete("1.0", tk.END)
            self.connection = sqlite3.connect(self.selected_db_file)
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                self.text_schema.insert(tk.END, f"Table: {table_name}\n")
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                for column in columns:
                    column_name = column[1]
                    column_type = column[2]
                    self.text_schema.insert(tk.END, f"  {column_name}: {column_type}\n")
                self.text_schema.insert(tk.END, "\n")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def execute_query(self):
        try:
            query = self.entry_query.get()
            if not query:
                messagebox.showwarning("Warning", "Please enter a query.")
                return

            cursor = self.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            self.display_results(rows)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_results(self):
        try:
            query = self.entry_query.get()
            if not query:
                messagebox.showwarning("Warning", "Please execute a query first.")
                return

            cursor = self.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()

            if self.export_format.get() == "csv":
                self.export_to_csv(rows)
            elif self.export_format.get() == "json":
                self.export_to_json(rows)
            elif self.export_format.get() == "tabbed":
                self.display_tabbed_view(rows)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_to_csv(self, rows):
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as file:
                csv_writer = csv.writer(file)
                csv_writer.writerows(rows)
            messagebox.showinfo("Export Successful", f"Data exported to {file_path}")

    def export_to_json(self, rows):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = [dict(zip([description[0] for description in cursor.description], row)) for row in rows]
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            messagebox.showinfo("Export Successful", f"Data exported to {file_path}")

    def display_tabbed_view(self, rows):
        top = tk.Toplevel(self)
        top.title("Tabbed View")
        text_results = ScrolledText(top, height=20, width=80)
        text_results.pack()
        for row in rows:
            text_results.insert(tk.END, "\t".join(map(str, row)) + "\n")

    def display_results(self, rows):
        self.text_results.delete("1.0", tk.END)
        for row in rows:
            self.text_results.insert(tk.END, f"{row}\n")

app = QueryTool()
app.mainloop()
