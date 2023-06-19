# Urban-Architects
2023 Complex Systems Simulation Group 11

## Introduction:  

The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp

## Initial model:  

The code generates a grid of shape NXN. 
The cells colonize their neighbor with probability P.
Below a critical probability Pc, the points will fail to occupy the entire box, and regions of space will be left empty.
Well above Pc, the majority of the box is colonized.  Around the critical probability, arbitrarily large regions can be occupied or left empty.
Initialization:

A block of size 2X2 is generalized in the center as the initial starting point.
The river is added into the grid where the cells can never be occupied or colonized.

## Visualization:  

plots generated with numpy and matplotlib

## Files:  

`PercolationModel2D.py` contains the PercolationModel2D object which holds the automaton and the rules by which it evolves.

`PercolationModelPatterns.py` contains a set of basic patterns to add to the grid initially.

`config.py` contains the variables needed to run the model.

`run_model.py` is a run script to generate percolation models.

(More details be added with the development of the mathematical model.

