# Bamboogle evaluation dataset

- `bamboogle.csv`: 125 question/answer pairs converted from the course-provided `Bamboogle Prerelease.xlsx` workbook, including the optional alternate answer column.
- The original `.xlsx` file is not yet committed; CSV is the ready-to-use dataset in this repository.
- The CSV preserves the spreadsheet's stored cell values. Some original date cells are Excel serial numbers rather than formatted dates; we should normalize those when implementing evaluation, without altering source ground truth silently.
- **Do not train on these evaluation questions**: use separate training data for Flow-GRPO.
