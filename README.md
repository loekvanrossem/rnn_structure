# RNN Automaton Learning Visualization Tool

This repository contains a Jupyter notebook demonstrating a tool for analyzing the development of the automaton within a Recurrent Neural Network (RNN) during training. The tool provides insights into how the RNN learns and evolves its internal structure, by showing how an automaton is constructed. An example is provided for sequence length generalization in the parity computation task.

## Contents

- **source:** This folder contains source code and utilities used for conducting experiments.
- **example:** This folder contains an example of an application to sequence length generalization in parity computation.
- **environment.yaml:** This file lists the required packages and dependencies needed to run the experiments.

## Usage

1. Ensure you have Python 3.10 or higher installed.
2. Set up the environment using the provided `environment.yaml` file.

```bash
    conda env create -f environment.yaml
    conda activate rnn_structure
```

3. Navigate to the "example" folder and run "main.ipynb".
