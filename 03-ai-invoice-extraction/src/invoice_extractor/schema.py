"""Data model for an extracted invoice, with business-rule validation."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LineItem(BaseModel):
    description: str
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)


class Invoice(BaseModel):
    supplier_name: str
    invoice_number: str
    invoice_date: date
    due_date: Optional[date] = None
    currency: str = Field(default="EUR", min_length=3, max_length=3)
    line_items: list[LineItem] = Field(default_factory=list)
    total_excl_tax: float = Field(ge=0)
    tax_amount: float = Field(ge=0)
    total_incl_tax: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_totals(self) -> "Invoice":
        """Flag (rather than reject) inconsistencies so a human can review them."""
        tol = 0.02
        if abs(self.total_excl_tax + self.tax_amount - self.total_incl_tax) > tol:
            self.warnings.append("total_excl_tax + tax_amount != total_incl_tax")
        if self.line_items:
            lines = round(sum(li.total for li in self.line_items), 2)
            if abs(lines - self.total_excl_tax) > tol:
                self.warnings.append(f"sum of line items ({lines}) != total_excl_tax ({self.total_excl_tax})")
        if self.due_date and self.due_date < self.invoice_date:
            self.warnings.append("due_date is before invoice_date")
        return self

    @property
    def needs_review(self) -> bool:
        return bool(self.warnings)
