# Vehicle-to-Grid Revenue Maximization Model

This repository contains a Mixed-Integer Linear Programming (MILP) optimization framework for Vehicle-to-Grid (V2G) operations using Python and Google OR-Tools.

The project focuses on maximizing net revenue from energy market participation while respecting EV battery, availability, and grid constraints.

## Project Structure

```text
.
├── LICENSE
├── README.md
├── data/
│   └── (input datasets and scenario files)
├── notebooks/
│   ├── Model_Implementation.ipynb
│   └── Optimization_Model.ipynb
├── results/
│   └── (outputs, figures, and exported summaries)
└── src/
    └── v2g_model.py
```

## Components

- `notebooks/Model_Implementation.ipynb`: Full implementation including base model, extensions, and sensitivity analysis.
- `notebooks/Optimization_Model.ipynb`: Compact, end-to-end optimization workflow.
- `src/v2g_model.py`: Modular Python implementation with reusable functions:
  - `generate_default_data()`
  - `create_model()`
  - `solve_model()`
  - `extract_results()`
  - `run_pipeline()`

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run the Notebook

1. Open `notebooks/Optimization_Model.ipynb` or `notebooks/Model_Implementation.ipynb` in VS Code or Jupyter.
2. Select a Python kernel with required packages installed.
3. Run all cells from top to bottom.

## How to Run the Python Module

Run the modular optimization script from the repository root:

```bash
python -m src.v2g_model
```

This executes the full pipeline (data generation, model build, solve, result extraction, and export). Output files are saved in `results/`.

## How to Run Tests

Run automated tests from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Continuous Integration

This repository includes a GitHub Actions workflow that runs on each push and pull request to:

- Install dependencies from `requirements.txt`
- Run the test suite
- Execute the optimization pipeline (`python -m src.v2g_model`) as a smoke check

## Portfolio Readiness Notes

- Clear separation between exploratory analysis (`notebooks/`) and reusable code (`src/`).
- Dedicated `data/` and `results/` directories for analytics workflow hygiene.
- Modular model functions to support extension, testing, and integration into larger analytics pipelines.
