FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[llm]"
COPY samples ./samples
# Pass ANTHROPIC_API_KEY at run time; without it the rule-based extractor is used.
ENTRYPOINT ["python", "-m", "invoice_extractor.cli"]
CMD ["samples", "--out", "output/invoices.csv"]
