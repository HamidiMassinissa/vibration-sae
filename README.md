# Monitoring of a Dynamic System Based on Autoencoders

This repository contains code to reproduce experimental resutls of our submission to IJCAI 2019.

# Considered application
Here is the dashboard of the monitoring system:
<p align="center">
  <img src="https://user-images.githubusercontent.com/8298445/56567828-07c03100-65b6-11e9-96df-8f277771476a.JPG"
       height="400px"/>
</p>

A complete interactive `dash`-based application to visualize the dataset can be found [here](https://lipn.univ-paris13.fr/~hamidi/vibration/). Here is a preview of the signals being monitored:
<p align="center">
  <img src="https://user-images.githubusercontent.com/8298445/56568488-69cd6600-65b7-11e9-8f99-a79ddeab1068.png"
       height="350px"/>
</p>

# Requirements
* `numpy`
* `pyTorch`
* `scikit-multiflow` (patched version available [here](https://github.com/HamidiMassinissa/scikit-multiflow))
* `scikit-optimize`
* `fanova` to install, please follow the steps [here](https://automl.github.io/fanova/install.html)

If you are using `pip` package manager, you can simply install all requirements via the following command:

    python -m virtualenv .env -p python3 [optional]
    source .env/bin/activate [optional]
    pip3 install -r requirements.txt

# How to run
Take a look inside the corresponding folder of each section. Further instructions are provided there.

## Autoencoder

## Continuous monitoring

## Bayesian optimization

## functional analysis of variance
