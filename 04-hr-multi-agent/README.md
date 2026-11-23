# HR Agent Team: a multi-agent People Analytics assistant

**Business question:** an HR director asks a question in plain language ("Where are we losing people and what should we do first?"). Today an analyst needs a day to pull the numbers and write a note. Can a team of AI agents produce a reliable, sourced memo in about a minute?

> Built on the HR dataset of [project 01](../01-hr-analytics/) (**synthetic** data). Try it in the Streamlit app: page **HR Agent Team**.

## Architecture

```
            question from the HR director
                         │
                         ▼
          ┌──────────────────────────────┐
          │ Coordinator agent             │  plans, delegates, writes the memo
          │ (Head of People Analytics)    │  only tool: delegate(specialist, task)
          └──────┬──────────┬──────────┬─┘
     in parallel │          │          │
                 ▼          ▼          ▼
         Turnover &     Pay Equity   Retention Risk     each: own prompt, own tools,
          Absence                                        own context, own tool loop
                 │          │          │
                 ▼          ▼          ▼
     ┌────────────────────────────────────────────┐
     │ Data tools (Python): aggregates only        │  workforce_overview, turnover_breakdown,
     │ groups < 10 people suppressed               │  absence_overview, pay_gap_analysis,
     └────────────────────────────────────────────┘  attrition_drivers, risk_by_segment
```

- **Coordinator / specialists.** The coordinator cannot touch the data. It decides which specialists a question needs and briefs them through a `delegate` tool. Each specialist is a separate agent loop on the Claude Messages API with its own system prompt, its own subset of tools and its own context, so the coordinator only reads short reports.
- **Parallelism.** When the coordinator delegates to several specialists in the same turn, they run in parallel threads.
- **Grounded answers.** Every figure must come from a tool result. Tools are ordinary, tested Python functions (pandas, a log-salary regression for the adjusted pay gap, the scikit-learn attrition model of project 01).
- **Privacy by design.** Tools return aggregates only and suppress groups under 10 people: an agent cannot be asked "who is going to resign?", only "which groups are at risk?". The memo always reminds that no automated individual decision should be taken (GDPR).
- **Observable.** Every step (delegation, tool call, result, report, memo) is emitted as an event; the Streamlit page streams them live, like an agent console.
- **Cost control.** The coordinator runs on Claude Sonnet; the specialists, which only pick tools and summarise figures, run on Claude Haiku (about 20× cheaper per token). Reports and memo are kept short (output tokens cost the most), tool results are sent as compact JSON, and the app reuses the saved answer when the exact same question is asked again. The page shows the cost of each answer.
- **Offline mode.** Without an API key, the same team runs a fixed tool plan per specialist and fills a memo template. The public demo works for free and is clearly labelled as generated without AI.

## Example (offline mode, real figures from the tools)

> **Key findings**
> - Turnover was **17.8 %** in 2025 (395 leavers, 487 hires).
> - **Retail Operations** loses the most people (19.6 % of staff left) vs Finance (8.4 %); 73 % of departures are resignations.
> - Adjusted for level, department, country and age, women earn **4.2 % less** than comparable men (95 % CI 3.3 to 5.2 %).
> - The risk model puts **40.6 %** of leavers in its top 20 % risk group; main drivers: engagement, overtime, pay vs. market.

The adjusted pay-gap tool recovers the 4 % gap built into the synthetic data, which is a useful end-to-end check of the method.

## Tests

`pytest` (15 tests, no API key, no network):

- every tool returns JSON, rejects bad arguments with an error message instead of crashing, and never exposes an employee ID;
- the offline router and memo;
- the agent loop with a **fake Claude client**: the coordinator delegates to two specialists in one turn, each specialist only sees its own tools, reports flow back into the memo, token usage and cost are counted per model, specialists run on the cheaper model, and an unknown specialist is returned to the model as an error.

The Streamlit page is also tested in live mode with a fake SDK (`tests/test_app.py` at the repository root).

## Run it

```bash
pip install -e ".[llm]"
pytest

# From the repository root, in the app (offline mode without a key)
streamlit run Home.py
```

Live mode: add `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL` for the coordinator and `ANTHROPIC_SPECIALIST_MODEL` for the specialists) to `.streamlit/secrets.toml` locally, or to the app's secrets on Streamlit Community Cloud. Never commit the key.

```python
from hr_agents import HRAgentTeam
result = HRAgentTeam(on_event=print).run("Which departments are most at risk of losing people, and why?")
print(result.memo, result.input_tokens, result.output_tokens)
```

## Next steps

Evaluation set of HR questions with expected figures (LLM-as-judge + exact checks on numbers), a memory of past memos so the team can track changes over time, and an MCP server exposing the data tools to other agents.

## Credits

The coordinator / sub-agent pattern follows the multi-agent approach taught in Anthropic's *Code with Claude* workshop materials (Apache-2.0). The code in this folder was written for this project.
