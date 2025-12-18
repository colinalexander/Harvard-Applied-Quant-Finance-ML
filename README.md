````markdown
# Harvard Applied Quantitative Finance & Machine Learning (Fall 2025)

This repository contains my **final project** and supporting code for  
**[CSCI E-278: Applied Quantitative Finance and Machine Learning](https://coursebrowser.dce.harvard.edu/course/applied-quantitative-finance-and-machine-learning-2/)** (Harvard Extension School, Fall 2025).

The repository is intentionally **final-project-centric** (with a small amount of supporting scaffolding). All exploratory material and non-essential artifacts are excluded to keep the repo focused, reviewable, and reproducible.

---

## Final Project

### **Scaling Machine Learning Signals for Equity Factor Portfolios**
**An applied study using 35 years of daily Fama–French 100 portfolios**

**Primary notebook:**  
- [`notebooks/aqfml-final-project.ipynb`](notebooks/aqfml-final-project.ipynb)

**Key supporting code (reusable pipeline):**  
- `src/` (feature construction, walk-forward training, signal diagnostics, portfolio implementation, plotting)

**Key reproducibility artifacts:**  
- `data/processed/` (analysis-ready datasets used by the notebook)
- `results/` (saved intermediate outputs used in appendices / diagnostics)

### Project Summary

This case study examines how increasing data richness affects the behavior and usability of machine-learning signals in equity factor portfolios. Using **Fama–French 100 Size × Book-to-Market portfolios** at **daily frequency** over ~35 years, it compares linear and nonlinear models under a **leakage-aware walk-forward framework**, evaluating:

- **Cross-sectional ranking quality** via the Information Coefficient (Spearman rank correlation)
- **Economic outcomes** via long–short portfolio implementations with turnover and transaction costs

### Models Evaluated

- **Ridge Regression** (linear baseline)
- **Random Forest**
- **Shallow Neural Network (MLP)**
- **Boosted Trees (XGBoost)** (robustness extension)

### Core Takeaway

Improving predictive metrics, adding data, or increasing model flexibility **in isolation** does not guarantee deployable results. **Model choice, data scale, and portfolio implementation must be considered jointly**, because turnover, stability, and transaction costs materially change conclusions.

---

## Repository Structure

```text
Harvard-Applied-Quant-Finance-ML/
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── data/
│   ├── raw/                 # Ignored (source inputs kept local)
│   ├── interim/             # Ignored (intermediate artifacts kept local)
│   └── processed/           # Versioned, analysis-ready datasets
├── notebooks/
│   ├── aqfml-final-project.ipynb
│   └── static/              # Figures/images used by the notebook
├── results/                 # Generated outputs used in appendices/diagnostics
├── src/                     # Reusable code (features, models, walk-forward, portfolio, plots)
├── scripts/                 # Optional utilities
└── tests/                   # Optional tests
````

> **Notes:** Course notes/slides are kept locally (not versioned) to keep the repository focused on the final project deliverable.

---

## Data Versioning & Reproducibility

This repository is structured to be reproducible without committing bulky or redundant artifacts:

* **Raw data (`data/raw/`)** and **intermediate artifacts (`data/interim/`)** are intentionally not versioned
* **Processed datasets (`data/processed/`) are versioned** and represent:

  * fully cleaned
  * leakage-safe
  * analysis-ready inputs used by the final notebook

This design allows results and figures to be reproduced directly from the repository without re-downloading or re-processing raw inputs.

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

## Development Notes

* Pre-commit hooks are used for formatting/linting Python code.
* Notebook linting is intentionally excluded (notebooks are treated as deliverables, not library modules).
* The final notebook is designed to read top-to-bottom as an applied workflow:
  data → features → walk-forward training → signal diagnostics → portfolio implementation → cost sensitivity.

---

## Term

**Fall 2025 — Harvard Extension School**
**CSCI E-278: Applied Quantitative Finance and Machine Learning**

---

## License

This repository is provided for educational and research purposes.
See [LICENSE](LICENSE) for details.
