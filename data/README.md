# Bamboogle evaluation dataset

- `Bamboogle Prerelease.xlsx`: the course-provided workbook (source of truth).
- `bamboogle.csv`: 125 question/answer pairs generated from it with `python scripts/xlsx_to_csv.py`, including the optional alternate answer column (`answer_2`).
- Excel stores some answers as dates or floats; the script formats them back to the text shown in the spreadsheet (e.g. `May 29, 1932`, `1999`).
- **Do not train on these evaluation questions**: use separate training data for Flow-GRPO.
