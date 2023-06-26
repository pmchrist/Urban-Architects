# Urban-Architects
2023 Complex Systems Simulation Group 11

## Introduction:  

This project looks into the insides of the interactions between regional geography, population, climate, and energy.  
A CA (cellular automata) model is used to simulate this complex system.  
The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp  

## Model:   

### Initialization:  
The initial map is applied at step = 0;  
Grid of population, energy, cell type of size NXN are generated accordingly.    
The initial parameters are:  
    migration_threshold,  
    burnrate,   
    growthrate,      
    emissions,         
    emmigration_size,     
    energy_replenish_chance,     
    energy_replenish_size,     
    energy_barrier,     
    view_distance.  


### Logical insides:  

The population of the cell grows over time with a certain amount of energy consuming.  
The initial energy grid is randomized with a constant replenish chance and size.  
When the energy in a cell is drained out by the population, or the density of population is too high (too crowded), people migrate to find a cell with higher fitness score to live in.  
The fitness is decided by the population, energy and number of water cells in the neighborhood.  
While people consume energy in the region, they also emmit CO2 etc. that lead to the regional climate change such as green house effect.  
The cell type change according to the climate change: rise of sea level...  

## Visualization:  

The following plots generated with numpy and matplotlib:    
1. Energy, population and cell type 2D plate plots over n time steps.    
2. Line plots of the climate change, mean population, mean energy and etc.   
3. Hist plots of the climate change, mean population, mean energy and etc.   

## Files:  

`PercolationModel.py` contains the PercolationModel2D object which holds the automaton and the rules by which it evolves.

`config.py` contains the variables needed to run the model.

`init_map2.py` and `init_water_map.py` provides the initial map of the field we are looking into.

`run_model.py` is a run script to generate percolation models.

`visualization.ipynb` provides the visualization of the results.
