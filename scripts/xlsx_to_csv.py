"""Convert the course-provided Bamboogle workbook to data/bamboogle.csv.

Excel stores some answers as dates or floats, so we format them back to text
(e.g. 1932-05-29 -> "May 29, 1932", 1999.0 -> "1999").
"""
import csv
import datetime

import openpyxl


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, datetime.datetime):
        return f"{value:%B} {value.day}, {value.year}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


rows = openpyxl.load_workbook("data/Bamboogle Prerelease.xlsx").active.iter_rows(min_row=2, values_only=True)
with open("data/bamboogle.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["question", "answer", "answer_2"])
    for row in rows:
        if row[0]:  # columns: Question, -, -, -, Answer, Answer 2
            writer.writerow([as_text(row[0]), as_text(row[4]), as_text(row[5])])
