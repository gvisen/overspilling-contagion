# Overspilling contagion

This repository contains the code for the paper "A default system with overspilling contagion" by Coculescu and Visentin.

To get started, you can follow the brief tutorial in the Jupyter notebook `demo.ipynb`, where we guide you through the  simulation of paths for the process $\ell^{S|\emptyset}$ (i.e. Algorithm 2 in the paper).

## Code overview

The main module is `processes`, which contains the following classes:

* Process: base class for stochastic processes
* Sample: base class for path samples of stochastic processes
* basic_affine_process: implementation of the basic affine process of Duffie and Garleanu
* l: implementation of the l process in Theorem 4.1 using Algorihtms 1 and 2.

## Dependencies

* scipy = 1.7.3
* matplotlib = 3.5.3
* numpy = 1.21.5