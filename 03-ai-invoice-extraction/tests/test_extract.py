from datetime import date
from pathlib import Path

import pytest

from invoice_extractor import Invoice, RuleBasedExtractor
from invoice_extractor.cli import run
from invoice_extractor.extract import _to_float

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


@pytest.mark.parametrize("raw,expected", [("1 465,00", 1465.0), ("2,637.00", 2637.0), ("1.234,56", 1234.56), ("4,50", 4.5)])
def test_amount_parsing(raw, expected):
    assert _to_float(raw) == pytest.approx(expected)


def test_french_invoice():
    inv = RuleBasedExtractor().extract((SAMPLES / "invoice_001.txt").read_text())
    assert inv.invoice_number == "FA-2025-0412"
    assert inv.invoice_date == date(2025, 11, 3)
    assert inv.total_incl_tax == pytest.approx(772.80)
    assert len(inv.line_items) == 3
    assert not inv.needs_review


def test_english_invoice():
    inv = RuleBasedExtractor().extract((SAMPLES / "invoice_002.txt").read_text())
    assert inv.invoice_number == "NW-88213"
    assert inv.total_excl_tax == pytest.approx(2197.50)
    assert not inv.needs_review


def test_inconsistent_invoice_is_flagged():
    inv = RuleBasedExtractor().extract((SAMPLES / "invoice_003.txt").read_text())
    assert inv.needs_review
    assert any("total_incl_tax" in w for w in inv.warnings)
    assert any("due_date" in w for w in inv.warnings)


def test_schema_flags_bad_totals():
    inv = Invoice(supplier_name="X", invoice_number="1", invoice_date=date(2025, 1, 1),
                  total_excl_tax=100, tax_amount=20, total_incl_tax=130)
    assert inv.needs_review


def test_cli_writes_csv(tmp_path):
    rows = run(SAMPLES, tmp_path / "out.csv", offline=True)
    assert len(rows) == 3
    assert [r["status"] for r in rows] == ["ok", "ok", "to review"]
    assert (tmp_path / "out.csv").exists()
