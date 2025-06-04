# Wordle Solver

A Python implementation of two Wordle solvers: an entropy-based solver and a baseline solver. The project includes performance analysis, visualization tools, and detailed evaluation metrics.

## Overview

This project implements and compares two different approaches to solving Wordle:

1. **Entropy Solver**: Uses information theory and heuristics to maximize information gain
2. **Baseline Solver**: Uses a simple fixed first word and greedy approach

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Solvers
```bash
python src/visualize.py
```

### Generating Analysis
```bash
python src/error_analysis.py
```

## Project Structure

```
wordle-solver/
├── data/
│   ├── solutions.txt    # List of solution words
│   └── guesses.txt      # List of allowed guess words
├── src/
│   ├── entropy_solver.py    # Entropy-based solver implementation
│   ├── baseline_solver.py   # Baseline solver implementation
│   ├── visualize.py         # Visualization tools
│   ├── error_analysis.py    # Error analysis tools
│   └── feedback.py          # Wordle feedback implementation
└── plots/                  # Generated visualizations

