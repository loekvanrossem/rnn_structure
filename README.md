# RNN Automaton Development Tool

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This repository contains a Jupyter notebook demonstrating a tool for analyzing the development of the automaton within a Recurrent Neural Network (RNN) during training. The tool provides insights into how the RNN learns and evolves its internal structure, particularly focusing on tasks involving sequential data processing.

## Features

- **Automaton Development Analysis**: Determine the evolution of the automaton within an RNN during training, enabling insights into the model's learning process and internal dynamics.
- **Example: Parity Computation Task**: An example task is provided where the RNN is trained to compute the parity of sequences of 0s and 1s. This task demonstrates the generalization capabilities of the RNN to longer sequence lengths as it learns a finite automaton representation.
- **Generalization Evaluation**: Assess the model's ability to generalize to sequences of varying lengths, providing insights into its capacity to learn and represent patterns in sequential data.

## Usage

1. **Installation**: Clone the repository and create a conda environment using the provided `environment.yaml` file.

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
conda env create -f environment.yaml
conda activate rnn_structure

    Run the Jupyter Notebook: Launch Jupyter Notebook and open the provided notebook file (automaton_analysis.ipynb).

bash

jupyter notebook automaton_analysis.ipynb

    Execute Cells: Run each cell in the notebook sequentially to load settings, generate data, instantiate the model, set up the compiler, perform training, visualize automaton dynamics, and evaluate generalization.

    Explore Results: Explore the generated visualizations and analyze the model's behavior and performance across different tasks and dataset characteristics.

Requirements

    Python 3.10+
    PyTorch 1.7.1+
    Additional dependencies listed in environment.yaml
