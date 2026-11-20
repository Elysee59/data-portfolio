# AI invoice extraction with Claude

**Business problem:** a small business receives dozens of supplier invoices a week, in French and English, and someone retypes them into the accounting tool. This takes hours and errors slip through.

**Solution:** a small Python tool that reads each invoice, extracts the key fields with an LLM (Claude), **checks the numbers** and produces one CSV for accounting. Invoices that don't add up are flagged "to review" instead of being silently accepted.

```
invoices (.txt) ──► extractor ──► validation rules ──► invoices.csv
                    Claude API      totals, dates       status: ok / to review / failed
                    or regex
```

## Design choices

- **Structured output:** Claude is called with a tool whose input schema is the `Invoice` Pydantic model, and `tool_choice` forces it to use it. The answer is therefore always valid JSON with the expected fields.
- **Trust but verify:** the LLM extracts, but deterministic business rules decide (excl. tax + VAT = incl. tax, sum of lines = total excl. tax, due date after invoice date). A model mistake or an incorrect invoice ends up in front of a human.
- **Works without an API key:** a rule-based extractor with the same interface is used for tests, CI and as a fallback.
- **Production basics:** packaged with `pyproject.toml`, unit tests with `pytest`, a `Dockerfile`, and a GitHub Actions workflow that runs the tests on every push.

## Example

The three sample invoices in `samples/` are fictional. The third one has been made inconsistent on purpose (wrong total and due date before the invoice date):

```
$ python -m invoice_extractor.cli samples --out output/invoices.csv --offline
3 invoices processed: 2 ok, 1 to review or failed -> output/invoices.csv
```

| file | supplier | number | total incl. tax | status | warnings |
|---|---|---|---|---|---|
| invoice_001.txt | Atelier Vélo Lumière SARL | FA-2025-0412 | 772.80 | ok | |
| invoice_002.txt | Northwind Outdoor Supplies Ltd | NW-88213 | 2,637.00 | ok | |
| invoice_003.txt | Glisse & Neige Distribution | GN/2025/1107 | 1,785.00 | to review | totals don't add up; due date before invoice date |

## Run it

```bash
pip install -e ".[llm,dev]"
pytest                                   # 9 tests, no API key needed

# Offline (rules only)
python -m invoice_extractor.cli samples --offline

# With Claude
export ANTHROPIC_API_KEY=...             # and optionally ANTHROPIC_MODEL
python -m invoice_extractor.cli samples

# With Docker
docker build -t invoice-extractor .
docker run --rm -e ANTHROPIC_API_KEY invoice-extractor
```

## Possible extensions

PDF and scanned invoices (PDF text extraction, or sending the PDF directly to Claude), export to an accounting tool, a small web interface for the "to review" queue, and a test set of real invoices to measure field-level accuracy of the LLM against the rules.

> Note: the tests cover the rule-based path and the validation logic. The Claude path needs an API key and has not been benchmarked on real invoices yet.
