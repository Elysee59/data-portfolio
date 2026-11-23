"""Portfolio home page. Run locally with:  streamlit run Home.py"""
import streamlit as st

from app_utils import chips, hero, setup

setup("Data & AI Portfolio", "📊")

hero(
    "Elysée Mugabire · Data & AI Portfolio",
    "I build data products that answer a business question end to end: from raw data to a model or a "
    "dashboard, and a recommendation someone can act on.",
    ["Data Science & AI engineer (ESIEE Paris)", "Master's in Data Analysis (IAE Paris-Est)",
     "HR data @ Decathlon", "ML forecasting @ Taiwan", "French · English"],
)

st.markdown("#### Explore the projects")
c1, c2, c3 = st.columns(3)
projects = [
    (c1, "👥 HR Analytics", "Where are we losing people, is pay fair, and who is likely to leave next?",
     "Interactive HR dashboard, equal-pay check, attrition risk model with a what-if simulator.",
     "pages/1_HR_Analytics.py"),
    (c2, "📈 Demand Forecasting", "How much should each store order 4 weeks ahead?",
     "Backtested ML forecast per store and product family: error cut by ~22 % and bias removed.",
     "pages/2_Demand_Forecasting.py"),
    (c3, "🧾 AI Invoice Extraction", "Can we stop retyping supplier invoices?",
     "Claude-powered extraction with structured output and automatic consistency checks.",
     "pages/3_AI_Invoice_Extraction.py"),
]
for col, title, question, what, page in projects:
    with col, st.container(border=True, height=250):
        st.markdown(f"**{title}**")
        st.markdown(f"*{question}*")
        st.markdown(what)
        st.page_link(page, label="Open the project", icon="➡️")

st.markdown("#### What I can help with")
a, b = st.columns(2)
with a:
    st.markdown(
        "- **Dashboards & reporting** (Power BI, Tableau, Streamlit) on clean, automated pipelines\n"
        "- **HR data & people analytics**: workforce KPIs, turnover, equal pay, Workday reports"
    )
with b:
    st.markdown(
        "- **Forecasting & predictive models**, tested the way they will be used\n"
        "- **AI automation**: turning documents and repetitive tasks into structured data with LLMs"
    )

st.markdown("#### Stack")
chips(["Python", "pandas", "scikit-learn", "Keras", "SQL", "Spark / Databricks", "R", "Power BI", "Tableau",
       "Streamlit", "Plotly", "Docker", "Kubernetes", "CI/CD", "Workday reporting", "Claude API"])
