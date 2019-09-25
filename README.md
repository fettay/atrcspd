# Detecting Change in Seasonal Pattern via Autoencoder and Temporal Regularization

### Description

This repository contains the code accompanying the paper: Detecting Change in Seasonal Pattern via Autoencoder and Temporal Regularization. It is implemented using PyTorch. 

### Setup

To set up just create a virtual environment with **python3** and run:

    pip install -r requirements.txt


### Available

The code helps running ATR-CSPD, KCpE and RDR on both the generated dataset (section 4.2) and the NYC taxi timeseries (section 4.4).


### Run an experiment ###

Run the file *run.py*

###### Example:

    python3 run.py --dataset generated --model atrcspd

###### Usage:

    usage: run.py [-h] [--model MODEL] [--dataset DATASET]

    optional arguments:
    -h, --help         show this help message and exit
    --model MODEL      Model to run can be either: atrcspd, rulsif (RDR in the
                        paper) or kcpe.
    --dataset DATASET  Dataset to run on, can be either: generated or nyc
