-- HR KPIs on the employees table (Databricks SQL / Spark SQL syntax).
-- Load data/employees.csv as a table named `hr.employees` first.

-- 1. Month-end headcount, hires and leavers
WITH months AS (
  SELECT explode(sequence(DATE'2022-01-31', DATE'2025-12-31', INTERVAL 1 MONTH)) AS month_start
),
month_ends AS (
  SELECT last_day(month_start) AS month_end, trunc(month_start, 'MM') AS month_first FROM months
)
SELECT
  m.month_end,
  COUNT_IF(e.hire_date <= m.month_end AND (e.leave_date IS NULL OR e.leave_date > m.month_end)) AS headcount,
  COUNT_IF(e.hire_date BETWEEN m.month_first AND m.month_end)                                AS hires,
  COUNT_IF(e.leave_date BETWEEN m.month_first AND m.month_end)                               AS leavers
FROM month_ends m
CROSS JOIN hr.employees e
GROUP BY m.month_end
ORDER BY m.month_end;

-- 2. 2025 turnover rate by department (leavers / employees present during the year)
SELECT
  department,
  COUNT(*)                                                      AS employees_present,
  COUNT_IF(leave_date BETWEEN DATE'2025-01-01' AND DATE'2025-12-31') AS leavers,
  ROUND(100 * COUNT_IF(leave_date BETWEEN DATE'2025-01-01' AND DATE'2025-12-31') / COUNT(*), 1) AS turnover_pct
FROM hr.employees
WHERE hire_date <= DATE'2025-12-31'
  AND (leave_date IS NULL OR leave_date >= DATE'2025-01-01')
GROUP BY department
ORDER BY turnover_pct DESC;

-- 3. Equal-pay check: median salary by level and gender, active employees
SELECT
  job_level,
  percentile_approx(CASE WHEN gender = 'F' THEN annual_salary_eur END, 0.5) AS median_women,
  percentile_approx(CASE WHEN gender = 'M' THEN annual_salary_eur END, 0.5) AS median_men,
  ROUND(100 * (1 - percentile_approx(CASE WHEN gender = 'F' THEN annual_salary_eur END, 0.5)
                 / percentile_approx(CASE WHEN gender = 'M' THEN annual_salary_eur END, 0.5)), 1) AS gap_pct
FROM hr.employees
WHERE leave_date IS NULL
GROUP BY job_level;

-- 4. Absenteeism: average absence days and share of employees above 15 days
SELECT
  department,
  ROUND(AVG(absence_days_last_12m), 1)                      AS avg_absence_days,
  ROUND(100 * AVG(CASE WHEN absence_days_last_12m > 15 THEN 1 ELSE 0 END), 1) AS pct_above_15_days
FROM hr.employees
WHERE leave_date IS NULL
GROUP BY department
ORDER BY avg_absence_days DESC;
