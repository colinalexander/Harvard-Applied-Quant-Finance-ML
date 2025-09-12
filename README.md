# Harvard Applied Quantitative Finance & Machine Learning (Fall 2025)

This repository contains my **study notes, Jupyter notebooks, and supporting code** for  
**CSCI E-278: Applied Quantitative Finance and Machine Learning**, offered through Harvard Extension School in Fall 2025.

---

## 📘 Course Overview
The course explores the application of **quantitative finance and modern machine learning techniques** to investment management, portfolio construction, and risk modeling. It is structured into four major modules:

1. **Data Management**  
   - Visualization, preprocessing, and transformation of financial data  
   - Data curation and cleaning  
   - Temporal structure of financial time series  
   - Feature engineering for predictive modeling  

2. **Quantitative Investment Strategies**  
   - Backtesting methodologies  
   - Statistical arbitrage and mean reversion strategies  
   - Momentum and trend-following approaches  

3. **Portfolio Management**  
   - Asset allocation and portfolio construction  
   - Portfolio optimization and rebalancing  
   - Reinforcement learning for dynamic allocation  

4. **Risk Management**  
   - Portfolio, trading, and factor risk exposure  
   - Hedging techniques  
   - Value-at-Risk (VaR) and volatility modeling  

---

## 📂 Repository Structure
```text
Harvard-Applied-Quant-Finance-ML/
├── docs/                # Markdown lecture notes and study guides
├── notebooks/           # Jupyter notebooks for code demonstrations
│   ├── 01_data_mgmt/
│   ├── 02_quant_strats/
│   ├── 03_portfolio_mgmt/
│   └── 04_risk_mgmt/
├── src/aqfml/           # Reusable helper functions (data, viz, features, models)
├── data/                # Local data (not versioned in Git)
│   ├── raw/
│   ├── interim/
│   └── processed/
├── tests/               # Optional unit tests for reusable modules
├── LICENSE
├── pyproject.toml       # Project metadata and dependencies (uv-based)
└── README.md
````

---

## ⚙️ Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

1. Clone the repository:

   ```bash
   git clone https://github.com/colinalexander/Harvard-Applied-Quant-Finance-ML.git
   cd Harvard-Applied-Quant-Finance-ML
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Launch JupyterLab:

   ```bash
   uv run jupyter lab
   ```

---

## 🛠️ Optional Setup for a Clean Git Workflow

### Strip Notebook Outputs

To avoid committing large notebook outputs and execution counts (which clutter diffs):

```bash
uv add nbstripout
uv run nbstripout --install
```

This installs a Git filter so that only code and markdown are stored in Git, not outputs. Locally, you’ll still see your outputs.

### Pre-commit Hooks

To automatically lint/format code and strip notebooks before committing:

```bash
brew install pre-commit   # or install via uv: uv run pre-commit install
pre-commit install
```

This enables `pre-commit` hooks configured in `.pre-commit-config.yaml`.

---

## 📦 Core Dependencies

* Python 3.13+
* NumPy, Pandas, SciPy
* Statsmodels, Arch, scikit-learn
* Matplotlib, Seaborn, Plotly
* Jupyter, JupyterLab, ipywidgets
* yfinance, pandas-datareader (for market data)
* Ruff, mypy, pre-commit (for code quality)
* Pytest + coverage (for testing)

---

## ✍️ Notes

* **Notes** are written in Markdown (`docs/`).
* **Notebooks** reproduce code demos and exercises from each module.
* **Helper functions** (in `src/aqfml/`) are meant to avoid copy-paste in notebooks.

---

## 📅 Term

**Fall 2025 – Harvard Extension School (CSCI E-278)**
*Applied Quantitative Finance and Machine Learning*

---

## 📜 License

This repository is provided for educational purposes. See [LICENSE](LICENSE) for details.
