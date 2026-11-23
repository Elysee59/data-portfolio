"""Smoke tests: every page of the Streamlit app renders without an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Home.py", "pages/1_HR_Analytics.py", "pages/2_Demand_Forecasting.py", "pages/3_AI_Invoice_Extraction.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception


def test_invoice_flags_inconsistent_sample():
    at = AppTest.from_file(str(ROOT / "pages/3_AI_Invoice_Extraction.py"), default_timeout=60).run()
    at.radio[0].set_value("invoice_003.txt").run()
    at.button[0].click().run()
    assert any("To review" in e.value for e in at.error)
