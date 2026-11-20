# Results (synthetic data)

- Headcount on 2025-12-31: **2260**
- 2025 turnover rate: **17.8%**
- Highest-turnover department: **Retail Operations** (19.6%); lowest: Finance (8.4%)
- Median pay gap (men vs. women, same level): Associate 5.1%, Professional 9.3%, Manager 14.9%, Director -4.5%

## Attrition model (who leaves in 2025, from a 1 January snapshot)

Base leaving rate: 15.5%

| Model | ROC AUC | PR AUC |
|---|---|---|
| Logistic regression | 0.675 | 0.307 |
| Gradient boosting | 0.634 | 0.240 |

The 20 % of employees with the highest risk score account for **41%** of actual leavers (test set).

Top drivers: engagement_score, salary_to_market_ratio, avg_monthly_overtime_h, department
