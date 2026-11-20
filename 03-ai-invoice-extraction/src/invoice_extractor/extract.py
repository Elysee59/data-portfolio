"""Two extraction back-ends behind the same interface.

- ``ClaudeExtractor`` sends the invoice text to the Claude API and forces a
  structured answer through tool use, so the output always matches the schema.
- ``RuleBasedExtractor`` uses regular expressions. It works offline, is used
  in the tests and CI, and is a fallback when no API key is configured.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Protocol

from .schema import Invoice, LineItem


class Extractor(Protocol):
    def extract(self, text: str) -> Invoice: ...


# --------------------------------------------------------------------------- LLM
class ClaudeExtractor:
    """Extract invoice fields with Claude (requires `pip install anthropic` and ANTHROPIC_API_KEY)."""

    def __init__(self, model: str | None = None):
        import anthropic  # imported lazily so the rest of the package works without it

        self.client = anthropic.Anthropic()
        # Set ANTHROPIC_MODEL to the model you want to use (see docs.claude.com for current model names).
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def extract(self, text: str) -> Invoice:
        schema = Invoice.model_json_schema()
        schema["properties"].pop("warnings", None)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=(
                "You extract data from supplier invoices. Use only information present in the document. "
                "Dates in ISO format (YYYY-MM-DD). Amounts as numbers without currency symbols. "
                "If a field is missing, omit it rather than guessing."
            ),
            tools=[{"name": "record_invoice", "description": "Record the extracted invoice.", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "record_invoice"},
            messages=[{"role": "user", "content": f"<invoice>\n{text}\n</invoice>"}],
        )
        data = next(block.input for block in response.content if block.type == "tool_use")
        return Invoice.model_validate(data)


# ----------------------------------------------------------------- rule-based
_AMOUNT = r"([0-9][0-9\s.,]*[0-9])"
_DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d", "%d.%m.%Y", "%d-%m-%Y")


def _to_float(raw: str) -> float:
    s = raw.replace(" ", "").replace(" ", "")
    if "," in s and "." in s:  # 1.234,56 or 1,234.56
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    return float(s)


def _to_date(raw: str):
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {raw!r}")


def _find(pattern: str, text: str, required: bool = True):
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        if required:
            raise ValueError(f"Pattern not found: {pattern}")
        return None
    return m.group(1).strip()


class RuleBasedExtractor:
    """Regex extractor for French/English invoices with labelled fields."""

    def extract(self, text: str) -> Invoice:
        supplier = text.strip().splitlines()[0].strip()
        number = _find(r"(?:invoice\s*(?:no|number|#)|facture\s*n[°o]?)\s*[:.]?\s*([A-Z0-9\-/]+)", text)
        inv_date = _to_date(_find(r"(?:invoice date|date de facture|date)\s*:\s*([0-9./\-]+)", text))
        due_raw = _find(r"(?:due date|échéance|date d'échéance)\s*:\s*([0-9./\-]+)", text, required=False)
        currency = "EUR" if re.search(r"€|EUR", text) else (_find(r"\b(USD|GBP|SGD|JPY|CHF)\b", text, False) or "EUR")

        items = []
        for m in re.finditer(rf"^\s*-\s*(.+?)\s*\|\s*{_AMOUNT}\s*\|\s*{_AMOUNT}\s*\|\s*{_AMOUNT}", text, re.MULTILINE):
            items.append(LineItem(description=m.group(1), quantity=_to_float(m.group(2)),
                                  unit_price=_to_float(m.group(3)), total=_to_float(m.group(4))))

        excl = _to_float(_find(rf"(?:total\s*(?:excl\.?\s*tax|ht|net))\s*:?\s*[€$£]?\s*{_AMOUNT}", text))
        tax = _to_float(_find(rf"^\s*(?:vat|tva|tax)(?:\s*\(?[0-9.,]+\s*%\)?)?\s*:\s*[€$£]?\s*{_AMOUNT}", text))
        incl = _to_float(_find(rf"(?:total\s*(?:incl\.?\s*tax|ttc|due)|amount due)\s*:?\s*[€$£]?\s*{_AMOUNT}", text))

        return Invoice(
            supplier_name=supplier, invoice_number=number, invoice_date=inv_date,
            due_date=_to_date(due_raw) if due_raw else None, currency=currency, line_items=items,
            total_excl_tax=excl, tax_amount=tax, total_incl_tax=incl,
        )


def get_extractor(prefer_llm: bool = True) -> Extractor:
    if prefer_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return ClaudeExtractor()
        except ImportError:
            pass
    return RuleBasedExtractor()


def to_json(invoice: Invoice) -> str:
    return json.dumps(invoice.model_dump(mode="json"), ensure_ascii=False, indent=2)
