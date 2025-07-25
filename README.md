# Gaussian Mixture EM Demonstration

This project contains a simple implementation of the Expectation--Maximization (EM) algorithm for Gaussian mixture models. The Python script and accompanying Jupyter notebooks generate data from a mixture of Gaussians and walk through the parameter estimation process.

## Contents
- `gauss_gen.py` – EM algorithm implementation and simulation utilities.
- `Gauss-Gen.ipynb` and `project_GM.ipynb` – notebooks that visualize and explain the steps of the algorithm.

## Requirements
- Python 3
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)

Install the dependencies with `pip`:

```bash
pip install numpy scipy matplotlib scikit-learn
```

## Usage
### Running the script
Execute the EM demonstration directly from the command line:

```bash
python gauss_gen.py
```

The script generates synthetic samples and iteratively fits the Gaussian mixture, displaying diagnostic plots along the way.

### Using the notebooks
Open the Jupyter notebooks to interactively explore the code and plots:

```bash
jupyter notebook
```

Then open `Gauss-Gen.ipynb` or `project_GM.ipynb` from the browser interface. The notebooks reproduce the steps of the script with additional explanations and visualization.
