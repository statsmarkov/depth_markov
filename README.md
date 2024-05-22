# Depth markov

This repo contains the experiments for the paper "Anomaly Detection based on Markov Data: A Statistical Depth Approach".

## How to the install dependencies ?

1. `pip install --upgrade pip-tools`
2. `pip-compile --generate-hashes --output-file=requirements.txt requirements.in --allow-unsafe`
3. `pip-sync requirements.txt`

## How to run the experiments

The experiments that appear in the paper can be found in the relevant Jupyter notebook. Each notebook indicates
where the experiments are located in the main text. 

To run the experiments, you can use the command `jupyter notebook <name_of_the_notebook>`.
