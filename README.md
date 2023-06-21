# Urban-Architects
2023 Complex Systems Simulation Group 11

## Introduction:  

The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp

## Initial model:  

The code generates a grid of shape NXN. 
There are 2 example rules
Visualization is based on vector of some parameter
The cells colonize their neighbor with probability P.
Initialization is random

## Visualization:  

plots generated with numpy and matplotlib

## Files:  

`PercolationModel.py` contains the PercolationModel2D object which holds the automaton and the rules by which it evolves.

`config.py` contains the variables needed to run the model.

`run_model.py` is a run script to generate percolation models.

(More details be added with the development of the mathematical model.

