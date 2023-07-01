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

We assume that the climate change will have a significant impact on urban expansion. Climate change can alter the distribution of suitable habitats, potentially making previously uninhabitable areas more desirable for urban development. The increasing frequency and severity of extreme weather events due to climate change can render urban areas more vulnerable, leading to a redirection of urban expansion away from high-risk regions. And the socioeconomic effects of climate change, such as changes in agricultural productivity and employment, can influence the economic viability of urban expansion plans and contribute to population pressures on cities. 
If our system exhibits SOC, we expect that small changes and increased pressure will lead to large changes in the environment’s state, while we can conclude that the system does not exhibit SOC if it follows more linear correlations.

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

Insert here

## Bak-Sneppen Population Dynamics
### Overview of some results

Insert here

## Bak-Sneppen Population Dynamics Conserved
### Overview of some results

Insert here

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


    ├── README.md          <- 
    ├── config.py          <- 
    ├── requirements.tex          <- 
    ├── Bak-Sneppen                <- 
    │   ├── __init__.py    <- 
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
    ├── CA Model               <- 
    │   ├── __init__.py    <- 
    │   │
    │   ├── historical_res  <-                    <- Some statistical runs from the complicated model
    │   │   
    │   ├── results       <-                      <- Folder with results
    │   │   
    │   ├── results_presentation       <-                      <- Folder with results for presentation
    │   
    │   ├── PercolationModelComplicated.py        <- Model that we did not manage to run
    │   ├── PercolationModelSimple.py             <- Topological Model where people move based on Population Density
    │   ├── init_map2.py                          <- Topography Map Generator
    │   ├── run_model_complicated.py              <- Runs overcomplicated model
    │   ├── run_model_simple.py                   <- Runs Simple Cellular Automata model
    │   └── visualization                         <- File with some of our statistical results
    │      
    ├── Legacy Files            <- (Unused files, and files used for inspiration)
    │   

--------


## References

Siqin Wang, Yan Liu, Yongjiu Feng & Zhenkun Lei (2021) To move or stay? A cellular automata model to predict urban growth in coastal regions amidst rising sea levels, International Journal of Digital Earth, 14:9, 1213-1235

Yang, Jianxin & Tang, Wenwu & Gong, Jian & Shi, Rui & Zheng, Minrui & Dai, Yunzhe. (2023). Simulating urban expansion using cellular automata model with spatiotemporally explicit representation of urban demand. Landscape and Urban Planning. 231. 104640. 

Hainan Yang, Huizhen Su, Liangjie Yang, "Evolution of Urban Resilience from a Multiscale Perspective: Evidence from Five Provinces in Northwest China", Complexity, vol. 2023, Article ID 2352094, 23 pages, 2023. 











