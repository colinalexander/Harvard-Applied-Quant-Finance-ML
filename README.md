# Harvard Applied Quantitative Finance & Machine Learning (Fall 2025)

This repository contains **coursework, research notebooks, and final project** for
**[CSCI E-278: Applied Quantitative Finance and Machine Learning](https://coursebrowser.dce.harvard.edu/course/applied-quantitative-finance-and-machine-learning-2/)**, offered through Harvard Extension School in Fall 2025.

The centerpiece of this repository is the **final applied research project**, which studies how **data scale, model choice, and portfolio implementation interact in practice** when deploying machine learning signals in equity factor portfolios.

---

## Final Project (Primary Focus)

### **Scaling Machine Learning Signals for Equity Factor Portfolios**

**An applied study using 35 years of daily Fama–French 100 portfolios**

**Location:**

* Notebook / report: `notebooks/final_project/`
* Supporting code: `src/harvard/`

### Project Summary

This final project examines whether **richer data environments**—specifically **daily frequency and expanded cross-sections**—meaningfully change the behavior and usability of machine-learning signals in equity factor portfolios.

Using **35 years of daily returns** on the **Fama–French 100 Size × Book-to-Market portfolios**, the analysis compares **linear and nonlinear models** under a **strictly leakage-aware walk-forward framework**, evaluating both:

* **Signal quality** via cross-sectional ranking metrics (Information Coefficient), and
* **Economic viability** via long–short portfolio implementations with realistic turnover and transaction costs.

### Key Research Questions

* Does scaling from monthly to daily data improve signal stability?
* Do nonlinear models materially outperform linear baselines once evaluated fairly?
* How much do portfolio construction and transaction costs change conclusions drawn from predictive metrics alone?

### Models Evaluated

* **Ridge Regression** (linear baseline)
* **Random Forest**
* **Shallow Neural Network (MLP)**
* **Boosted Trees (XGBoost)** – robustness extension

All models are:

* Trained using **rolling walk-forward cross-validation** with purge buffers
* Evaluated on identical out-of-sample windows
* Compared using **ranking metrics first**, then **portfolio outcomes**

### Core Findings

* **Daily Information Coefficients are modest (≈0.5–1.0%) but consistently positive**, indicating economically meaningful yet noisy cross-sectional structure.
* **Nonlinear models capture incremental interaction effects**, but **do not dominate linear baselines** once evaluated under the same protocol.
* **Portfolio implementation is decisive**: models with similar ranking skill diverge sharply once turnover and transaction costs are applied.
* The **linear Ridge baseline** combines moderate signal strength with **lower turnover**, resulting in more robust net outcomes than more flexible models.

### Central Takeaway

> Improving predictive metrics, adding data, or increasing model flexibility in isolation does not guarantee usable results.
> **Model choice, data scale, and portfolio implementation must be considered jointly.**

This framing reflects a **practitioner-oriented perspective**, prioritizing deployability and robustness over raw in-sample performance.

---

## Course Overview

The course explores the application of **quantitative finance and modern machine learning techniques** to investment management, portfolio construction, and risk modeling. It is structured into four major modules:

1. **Data Management**

   * Visualization, preprocessing, and transformation of financial data
   * Data curation and cleaning
   * Temporal structure of financial time series
   * Feature engineering for predictive modeling

2. **Quantitative Investment Strategies**

   * Backtesting methodologies
   * Statistical arbitrage and mean reversion strategies
   * Momentum and trend-following approaches

3. **Portfolio Management**

   * Asset allocation and portfolio construction
   * Portfolio optimization and rebalancing
   * Reinforcement learning for dynamic allocation

4. **Risk Management**

   * Portfolio, trading, and factor risk exposure
   * Hedging techniques
   * Value-at-Risk (VaR) and volatility modeling

---

## Repository Structure

```text
Harvard-Applied-Quant-Finance-ML/
├── README.md
├── data/
│   ├── raw/                # Ignored (source data)
│   ├── interim/            # Ignored (intermediate artifacts)
│   ├── processed/          # Versioned, analysis-ready datasets
│   └── 100_Portfolios_10x10_Daily.CSV
├── notebooks/
│   └── aqfml-final-project.ipynb   # Final project notebook
├── src/
│   ├── models/             # Model implementations (Ridge, RF, NN)
│   ├── pipeline.py         # Walk-forward training orchestration
│   ├── signals.py          # Ranking & IC diagnostics
│   ├── portfolio.py        # Portfolio construction & costs
│   └── plots.py            # Figures used in analysis
├── results/                # Generated outputs used in the report
├── docs/                   # Course notes and reference material
├── pyproject.toml
├── uv.lock
└── LICENSE

---

### Notes on Scope

This repository is intentionally scoped around the **final applied project**.

Only the final project notebook is versioned in Git; exploratory and intermediate
notebooks used during development are excluded to keep the repository focused,
reviewable, and reproducible.

---

## Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/colinalexander/Harvard-Applied-Quant-Finance-ML.git
cd Harvard-Applied-Quant-Finance-ML
uv sync
uv run jupyter lab
```

---

## Development & Reproducibility Notes

* **Notebook outputs are stripped** before commit using `nbstripout`
* Walk-forward pipelines are implemented in reusable modules to avoid copy-paste logic
* All preprocessing and scaling steps are recomputed **within each training window** to prevent leakage
* Portfolio simulations explicitly track **turnover and transaction costs**

---

## Core Dependencies

* Python 3.13+
* NumPy, Pandas, SciPy
* scikit-learn, statsmodels, arch
* Matplotlib, Plotly
* Jupyter / JupyterLab
* Ruff, mypy, pytest, pre-commit

---

## Term

**Fall 2025 – Harvard Extension School**
**CSCI E-278: Applied Quantitative Finance and Machine Learning**

---

## License

This repository is provided for educational and research purposes.
See [LICENSE](LICENSE) for details.
