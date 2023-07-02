# Urban-Architects
2023 Complex Systems Simulation Group 11

Isabel Klennert, Christos Perchanidis, Frenk Klein Schiphorst, Chang Lin

## Motivation:  

This project looks into the dynamics of the interactions between regional geography, population, climate, and energy.  
A CA (cellular automata) model is used to simulate this complex system, and the Bak-Sneppen model is used as an inspiration.  
The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp  


## Research Questions

How do climate change factors (such as emission increase, rising sea levels, etc.) influence the patterns and dynamics of urban expansion, and what emergent behaviours can we observe from these interactions? 

## Hypothesis

We predict that the simple Bak-Sneppen model will exhibit properties of SOC and give an overall idea of the dynamical behaviour surrounding this simple concept of evolution. 
.... (insert)

## Project Plan
0. We used Lean approach with tasks on Trello: https://trello.com/invite/b/ZNRiOvnC/ATTIae3881ca7cba479d31b5ae92f84997cd8DB6631B/urban-architects
1. Build simple Bak-Sneppen Model
    * Avalanche Time
    * Age
    * Check Power Law
3. Build Bak-Sneppen model with population dynamics
    * Avalanche Time
    * Check Power Law
4. Complicated CA Model using water map


 ## Bak-Sneppen Simple
### Overview of some results
We tried to recreate a simple Bak-Sneppen model. We see over time the average fitness is increasing as expetced. The histograms of avalanche time gives an indication of power law due to frequent smaller avalanches, and less frequent larger avalanches, however we found a negative R value when comparing to Lognormal. The model balances stability and instability. So the Small, frequent changes  represent minor evolutionary adaptations and are necessary for the system to evolve and adapt. At the same time, larger, less frequent changes represent more significant evolutionary shifts. This prevents the ecosystem from reaching a stable state where no further evolution occurs - in other words, they keep the system dynamic and evolving.

## Bak-Sneppen Population Dynamics Conserved
### Overview of some results

We extended the simple Bak-Sneppen model to include population migration. We did this by letting the system update the lowest non-zero population density at each iteration, based on the current density of that cell, the average density of its neighbors, and the fitness of the cell.
We found some clustering behavior by setting specific parameter settings for the fitness function. These parameter settings can be interpreted as the tendency for people to live together, but not in regions that are too overcrowded.
We calculated avalanche sizes and durations, and found that they follow a distribution that seemed to resemble a power-law distribution. On closer inspection, however, it turned out that it more closely resembled a log-normal distribution. This can be explained by the fact that our update functions are not linear, which causes the emergent behavior to follow slightly different patterns than the simple Bak-Sneppen model.

## Topographical (CA) Model
### Overview of some results
We managed to create a model which shows complex dynamics. We can see how overpopulation leads to the collapse in the population or extreme overcrowding. Meanwhile, increase in emissions which is linked to the rising water level and following flooding exacerbates problem further and makes emmigration more extreme. We can see in the patterns that once critical capacity is achieved rising water becomes deadly, and more extreme occurences of people moving are much more common. (Results are on the slides)

To run model based on population density go to "CA Model" folder and run:
"python __init__.py --function simple"
To run model based on fitness function and growth go to "CA Model" folder and run:
"python __init__.py --function complicated"


## Visualization:  
The following plots are generated with numpy and matplotlib:    
1. Energy, population, and cell type 2D plate plots over n time steps.    
2. Line plots of climate change, mean population, mean energy and etc.   
3. Hist plots of climate change, mean population, mean energy and etc.   

## Working with the repository

In order to locally gather the required packages, the following command can be called:
```
pip3 install -r requirements.txt
```


## Project Organization
------------


    ├── README.md                               <- 
    ├── requirements.tex                        <- Packages and versions required
    ├── Bak-Sneppen                             <- All code and results related to Bak-Sneppen
    │   ├── __init__.py    
    │   │
    │   ├── Results          
    │   │  
    │   │
    │   ├── BS_simple_avalanche.py                       <- simple 2D Bak Sneppen with avalanche calculation  
    │   ├── BakSneppen_PopulationDynamics_Conserved.py   <- Bak-sneppen with population dynamics  
    │   ├── BakSneppen_Research.ipynb                    <- 
    │   ├── BakSneppen_Simple.py                         <- Bak-Sneppen without avalanche calculations
    │   └── test_BakSneppen2D.py                         <- pytest
    │  
    │   
    │       
    ├── CA Model                                  <- 
    │   ├── __init__.py    
    │   │
    │   ├── historical_res                        <- Some statistical runs from the complicated model
    │   │   
    │   ├── results                               <- Folder with results
    │   │   
    │   ├── results_presentation                  <- Folder with results for presentation
    │   
    │   ├── PercolationModelComplicated.py        <- Model that we did not manage to run
    │   ├── PercolationModelSimple.py             <- Topological Model where people move based on Population Density
    │   ├── init_map2.py                          <- Topography Map Generator
    │   ├── run_model_complicated.py              <- Runs overcomplicated model
    │   ├── run_model_simple.py                   <- Runs Simple Cellular Automata model
    │   └── visualization                         <- File with some of our statistical results
    │      
    ├── Legacy Files                              <- (Unused files, and files used for inspiration)
    │   

--------


## References

Siqin Wang, Yan Liu, Yongjiu Feng & Zhenkun Lei (2021) To move or stay? A cellular automata model to predict urban growth in coastal regions amidst rising sea levels, International Journal of Digital Earth, 14:9, 1213-1235

Yang, Jianxin & Tang, Wenwu & Gong, Jian & Shi, Rui & Zheng, Minrui & Dai, Yunzhe. (2023). Simulating urban expansion using cellular automata model with spatiotemporally explicit representation of urban demand. Landscape and Urban Planning. 231. 104640. 

Hainan Yang, Huizhen Su, Liangjie Yang, "Evolution of Urban Resilience from a Multiscale Perspective: Evidence from Five Provinces in Northwest China", Complexity, vol. 2023, Article ID 2352094, 23 pages, 2023. 











