# Urban-Architects
2023 Complex Systems Simulation Group 11

Isabel Klennert, Christos Perchanidis, Frenk Klein Schiphorst, Chang Lin
## Introduction:  

This project looks into the dynamics of the interactions between regional geography, population, climate, and energy.  
A CA (cellular automata) model is used to simulate this complex system, and the Bak-Sneppen model is used as an inspiration.  
The initial CA model follows the percolation model see Landis 1999, "The Fermi Paradox: An Approach Based on Percolation Theory", JBIS, 51, 163-166
http://www.geoffreylandis.com/percolation.htp  

## Project Plan
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

Insert here

## Bak-Sneppen Population Dynamics
### Overview of some results

Insert here

## Bak-Sneppen Population Dynamics Conserved
### Overview of some results

Insert here

## Percolation Model
### Overview of some results
Insert here


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


# Project Organization
------------


    ├── README.md          <- 
    ├── config.py          <- 
    ├── requirements.tex          <- 
    ├── Bak-Sneppen                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── Results          <- 
    │   │  
    │   │
    │   ├── BS_simple_avalanche.py      
    │   ├── BakSneppen_PopulationDynamics.py   
    │   ├── BakSneppen_PopulationDynamics_Conserved.py
    │   ├── BakSneppen_Research.ipynb
    │   ├── BakSneppen_Simple.py  
    │   └──test_BakSneppen2D.py
    │  
    │   
    │       
    ├── CA Model               <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── historical_res  <-          
    │   │   
    │   ├── results       <- 
    │   │   
    │   ├── PercolationModelComplicated.py        <- 
    │   ├── PercolationModelSimple.py              
    │   ├── init_map2.py
    │   ├── run_model_complicated.py    
    │   ├── run_model_simple.py  
    │   └── visualization  
    │      
    ├── Legacy Files            <- Source code for use in this project.
    │   

--------


## References

Siqin Wang, Yan Liu, Yongjiu Feng & Zhenkun Lei (2021) To move or stay? A cellular automata model to predict urban growth in coastal regions amidst rising sea levels, International Journal of Digital Earth, 14:9, 1213-1235, DOI: 10.1080/17538947.2021.1946178

Yang, Jianxin & Tang, Wenwu & Gong, Jian & Shi, Rui & Zheng, Minrui & Dai, Yunzhe. (2023). Simulating urban expansion using cellular automata model with spatiotemporally explicit representation of urban demand. Landscape and Urban Planning. 231. 104640. 10.1016/j.landurbplan.2022.104640. 

Hainan Yang, Huizhen Su, Liangjie Yang, "Evolution of Urban Resilience from a Multiscale Perspective: Evidence from Five Provinces in Northwest China", Complexity, vol. 2023, Article ID 2352094, 23 pages, 2023. https://doi.org/10.1155/2023/2352094











