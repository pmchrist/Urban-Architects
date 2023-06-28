# Urban-Architects
2023 Complex Systems Simulation Group 11
Isabel Klennert, Christos Perchanidis, Frenk Klein Schiphorst, Chang Lin
## Introduction:  

This project looks into the dynamics of the interactions between regional geography, population, climate, and energy.  
A CA (cellular automata) model is used to simulate this complex system, and the Bak-Sneppen model is used as an inspiration.  
The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp  

## Model:   

### Initialization:  
The initial map is applied at step = 0;  
Grid of population, energy, and cell type of size NXN are generated accordingly.    
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

The population of the cell grows over time with a certain amount of energy consumption.  
The initial energy grid is randomized and the energy in the cell is replenished at a certain replenishment chance and size
When the energy in a cell is drained out by the population, or the density of the population is too high (too crowded), people migrate to find a cell with higher fitness score to live in.  
The fitness is decided by the population, energy and number of water cells in the neighborhood.  
While people consume energy in the region, they also emit CO2, etc. which leads to regional climate change such as the greenhouse effect.  
The cell type change according to climate change: the rise of sea level...  

## Visualization:  

The following plots are generated with numpy and matplotlib:    
1. Energy, population, and cell type 2D plate plots over n time steps.    
2. Line plots of climate change, mean population, mean energy and etc.   
3. Hist plots of climate change, mean population, mean energy and etc.   

## Files:  

`PercolationModel.py` contains the PercolationModel2D object which holds the automaton and the rules by which it evolves.

`config.py` contains the variables needed to run the model.

`init_map2.py` and `init_water_map.py` provide the initial maps of the field we are looking into.

`run_model.py` is a run script to generate percolation models.

`visualization.ipynb` provides the visualization of the results.


## Bak-Sneppen Simple

Each cell in the grid is initialized with a random value that represents the fitness of the species in that cell.

In each iteration of the simulation, the cell with the lowest fitness value (and its immediate neighbors) is identified and then given a new random fitness value. 

*plot_system*  generates a heat map of the fitness values in the grid at a given iteration. In addition, the minimum fitness of the system after each iteration of the simulation is plotted. 

## Bak-Sneppen Population Dynamics

Modified simulation of Bak-Sneppen model where fitness calculation based on the local and neighboring (Moore) densities of species. 
The density of a species in a cell is given by a random number at the start. During the simulation, the density of a cell is recalculated based on the local density, the average density of the neighboring cells, a fitness factor and a random factor.
The proportion of these contributions is governed by the alpha, beta, gamma, and delta parameters that must sum up to 1.

The model continues to replace the cells with the least density value and its neighbors until the specified number of iterations is complete. 

It also stores and plots the minimum density value at each iteration, which gives a view of how the lowest density in the system evolves over time.

The Gaussian function and fitness function are used to generate a fitness score, which is then used as one of the factors in the calculation of new density values.


## Bak-Sneppen Population Dynamics Conserved

Update rule is significantly more complex: The new density is calculated using the *new_density* function, and depending on this value, the densities of the neighbouring cells are also updated. This update considers a balance of population migration, ensuring that the total population (sum of all cell densities) remains conserved.


## Percolation Model








