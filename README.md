# Do Oil Prices Affect Inflation?

Time series project in Python on monthly US data from FRED (2000–2026). It asks three questions:

1. **How strong?** How much does inflation rise when oil gets 10% more expensive?
2. **How fast?** How many months does it take for the effect to show?
3. **Best model?** Can a neural network (GRU) predict inflation better than a regression?

## Data
- US consumer price index, all items (`CPIAUCSL`)
- WTI crude oil price (`MCOILWTICO`)

Both series come from FRED (Federal Reserve Bank of St. Louis), downloaded in November 2026 and saved as CSV files so the results are reproducible. The one missing CPI value (October 2025, never published because of the US government shutdown) is filled by linear interpolation.

## Method
- Both series converted to yearly % changes, to compare like with like and avoid spurious correlation from shared trends
- Lag analysis on the training years (2000–2018) to find when oil affects inflation
- **Model 1:** simple regression on last month's oil price change
- **Model 2:** distributed-lag regression with lags of 1, 3, 6 and 12 months, HAC (Newey–West) standard errors and VIF checks
- **Model 3:** GRU recurrent neural network in PyTorch, size chosen with time-series cross-validation, averaged over 5 random starts
- All models evaluated on unseen test years (2019–2026) against a naive baseline

## Results

| Model | Test RMSE | Better than baseline |
|---|---|---|
| Baseline (training mean) | 2.64 | – |
| Model 1 (lag 1) | 2.28 | 13.5% |
| Model 2 (lags 1, 3, 6, 12) | 2.09 | 20.9% |
| Model 3 (GRU) | 1.98 | 24.8% |

- Inflation reacts most strongly to oil price changes **one month later**, with a smaller delayed effect over a year.
- A lasting **10% oil price rise** goes with about **0.4 percentage points** higher inflation over a year.
- The GRU predicts best, while the regressions are the easiest to interpret. No model captures the 2021–2023 inflation surge, which oil alone cannot explain.

## Files
- `Project Oil & Inflation.ipynb` – full analysis with code, plots and comments
- `cpi_fred.csv`, `oil_fred.csv` – the data used
- `Oil Prices & Inflation 3D.html` – interactive 3D visualization

## Tools
Python · pandas · statsmodels · PyTorch · seaborn · matplotlib
