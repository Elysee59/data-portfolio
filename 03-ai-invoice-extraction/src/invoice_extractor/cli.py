"""Process a folder of invoices and write one CSV ready for accounting.

Usage:
    python -m invoice_extractor.cli samples/ --out output/invoices.csv [--offline]
"""
import argparse
import csv
import sys
from pathlib import Path

from .extract import RuleBasedExtractor, get_extractor

FIELDS = ["file", "supplier_name", "invoice_number", "invoice_date", "due_date", "currency",
          "total_excl_tax", "tax_amount", "total_incl_tax", "n_lines", "status", "warnings"]


def run(folder: Path, out: Path, offline: bool = False) -> list[dict]:
    extractor = RuleBasedExtractor() if offline else get_extractor()
    rows = []
    for path in sorted(folder.glob("*.txt")):
        try:
            inv = extractor.extract(path.read_text(encoding="utf-8"))
            rows.append({
                "file": path.name, **inv.model_dump(mode="json", exclude={"line_items", "warnings"}),
                "n_lines": len(inv.line_items),
                "status": "to review" if inv.needs_review else "ok",
                "warnings": "; ".join(inv.warnings),
            })
        except Exception as exc:  # keep going, but record the failure for a human
            rows.append({"file": path.name, "status": "failed", "warnings": str(exc)})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path)
    p.add_argument("--out", type=Path, default=Path("output/invoices.csv"))
    p.add_argument("--offline", action="store_true", help="use the rule-based extractor, no API call")
    args = p.parse_args(argv)
    rows = run(args.folder, args.out, args.offline)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{len(rows)} invoices processed: {ok} ok, {len(rows) - ok} to review or failed -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
