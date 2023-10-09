# Quantum Computing 191

## Overview

This repository is dedicated to exploring and understanding the fascinating world of Quantum Computing. With the resources and code snippets provided here, we aim to provide insights into the principles and applications of Quantum Computing.

## Contents

- `Code.py`: A Python script that contains code related to Quantum Computing algorithms and implementations.
- `Notebook_QuantumComputing.ipynb`: A Jupyter Notebook that provides an interactive environment to run and explore Quantum Computing code and theories.
- `Quantum_Computing_191.pdf`: A research and popularization document that provides theoretical information, explanations, and research findings related to Quantum Computing.

## Pricer Class Overview

The `Pricer` class, found within the repository, is designed to calculate option prices using various methods and models, including the Cox-Ross-Rubinstein (CRR) model, American option pricing, Partial Differential Equation (PDE) method, Longstaff-Schwartz Monte Carlo (LSM) method, and Neural Network pricing. The class is implemented in Python and utilizes libraries such as NumPy, SciPy, Matplotlib, TensorFlow, and others.

### Key Features

1. **CRR Model:**
   - Implements the binomial CRR model for option pricing.
   - Handles both Call and Put options.
   
2. **American Option Pricing:**
   - Prices American options using a binomial tree model.
   - Considers early exercise opportunities.

3. **PDE Option Pricing:**
   - Utilizes the PDE method for option pricing, considering European and American exercise styles.
   - Supports different solvers like "spsolve" and "splu".

4. **LSM (Polynomial Regression):**
   - Implements the Longstaff-Schwartz method using polynomial regression for American option pricing.

5. **Neural Network Pricing:**
   - Implements a neural network model for option pricing.
   - Utilizes TensorFlow for model building and prediction.

### Usage Example

```python
# Import necessary libraries
import numpy as np
import scipy as scp
import scipy.stats as ss
from scipy import sparse
import matplotlib.pyplot as plt
import tensorflow as tf

# Instantiate the Pricer class
pricer_example = Pricer(S0=100, K=110, r=0.05, sigma=0.3, T=2.221918, num_steps=5000, O_type="Call")

# Use various pricing methods
call_price_crr = pricer_example.CRR_model()
call_price_american = pricer_example.American_Price()
call_price_pde, exec_time_pde = pricer_example.PDE_price(Time=True, solver="spsolve")
call_price_lsm, exec_time_lsm = pricer_example.LSM(order=3)
```

## Methods Overview

- `CRR_model()`: Computes option price using the CRR model.
- `American_Price()`: Computes the price of American options.
- `PDE_price(Nspace=10000, exercise="American", Time=False, solver="splu")`: Prices options using the PDE method.
- `LSM(paths=10000, order=2)`: Prices options using the LSM method.
- `Neural_network()`: Prices options using a neural network model.

### Note

Ensure to install all necessary Python libraries and handle potential exceptions for robust application. The class is designed to handle various edge cases and provides a comprehensive toolkit for option pricing using different methods.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- [Qiskit](https://qiskit.org/) or any relevant Quantum Computing library

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RostaneF/Quantum_computing-191.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Quantum_computing-191
   ```
### Usage

- Open and run the Python script `Code.py` using your preferred IDE.
- Launch Jupyter Notebook and explore `Notebook_QuantumComputing.ipynb` for interactive learning and experimentation.
- Read through `Quantum_Computing_191.pdf` for theoretical insights and additional information.

## Collaboration Note

This work has been conducted in collaboration with a team of quant traders from BNY Mellon, London, aiming to explore, understand, and potentially leverage the capabilities of Quantum Computing in the realm of trading and financial analysis.

## Contact

- **Rostane F** - [GitHub Profile](https://github.com/RostaneF)
