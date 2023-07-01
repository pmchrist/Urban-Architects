import numpy as np
import copy
from init_map2 import *
import sys


class PercolationModel2D(object):    
    '''
    Class that calculates and displays behaviour of 2D cellular automata
    '''

    # Just keeping global variables for reference:
    green_transition = 0.1         # Lower means lower emissions
    migration_threshold = 1.0       # Threshold over which people want to migrate
    growthrate = 0.35                # Multiplier for the population
    view_distance = 1               # How far people can move (Increasing it creates more beautiful plots)

    # Just describing global arrays of environment here:
    pop_dens = None     # Population density map
    type = None         # Type of cell map
    # Maps for next step
    next_pop_dens = None
    next_type = None

    # Some helper arrays for utility function
    water_score = None
    migrants = []
    dead_migrants = []


    def __init__(self, ni):
        '''
        Initializes a PercolationModel2D with a specific grid size and temperature.
        
        Parameters:
        ni (int): Size of the grid.
        '''
    

        # Set environment
        self.emissions = 0              # current emissions
        self.N = ni                     # Size of 1 side
        self.Ntot = self.N*self.N       # Overall size
        self.init_grid()                # Initialize environment grid
        # Init grids for next step
        self.next_pop_dens = copy.deepcopy(self.pop_dens)
        self.next_type = copy.deepcopy(self.type)

    def init_grid(self):
        '''
        Initializes grid, which is a set of vectors which describe environment (for now randomly)
        
        Each point has some unique parameters, on which something is determined

        '''
        #self.pop_dens = np.random.rand(self.N, self.N)         # Random Grid 
        self.pop_dens = np.zeros((self.N, self.N))              # Empty Grid
        self.type = init_water_map(self.N, 0.2, 2, 5, 1)        # Map with water
        #self.type = np.zeros((self.N, self.N))                 # Empty Map

        # loop to purge population values in water area
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i, j] != 0:
                    self.pop_dens[i, j] = 0

    # Helper Functions
    # We assume that neighborhoods stop at border (it is not rolling from another side)
    def getMooreNeighbourhood(self, i,j, extent=1):
        '''
        Returns a set of indices corresponding to the Moore Neighbourhood around a given cell.
        
        Parameters:
        i, j (int): The coordinates of the cell.
        extent (int): The extent of the neighborhood. Default is 1.

        Returns:
        list: A list of indices representing the Moore Neighbourhood around the cell.
        '''

        # Check for incorrect input
        # Make it a test later
        assert i>=0 or i<self.N, "Error: Incorrect coordinates"
        assert j>=0 or j<self.N, "Error: Incorrect coordinates" 
        
        indices = []
        
        for iadd in range(i-extent,i+extent+1):
            for jadd in range(j-extent, j+extent+1):        
                if(iadd==i and jadd == j):  # If is not the same cell
                    continue
                if (iadd < 0 or iadd>=self.N):          # If is not in bounds
                    continue                
                if (jadd < 0 or jadd>=self.N):          # If is not in bounds
                    continue
                indices.append([iadd,jadd])
        
        return indices
    
    def getVonNeumannNeighbourhood(self,i,j,extent=1):
        '''
        Returns a set of indices corresponding to the Von Neumann Neighbourhood around a given cell.
        
        Parameters:
        i, j (int): The coordinates of the cell.
        extent (int): The extent of the neighborhood. Default is 1.

        Returns:
        list: A list of indices representing the Von Neumann Neighbourhood around the cell.
        '''

        # Check for incorrect input
        # Make it a test later
        assert i>=0 or i<self.N, "Error: Incorrect coordinates"
        assert j>=0 or j<self.N, "Error: Incorrect coordinates" 
        
        indices = []
        
        for iadd in range(i-extent,i+extent+1):
            if(iadd==i): continue
            if(iadd < 0 or iadd>=self.N): continue       # If is not in bounds                
            indices.append([iadd,j])
            
        for jadd in range(j-extent,j+extent+1):
            if(jadd==j): continue
            if(jadd < 0 or jadd>=self.N): continue       # If is not in bounds                            
            indices.append([i,jadd])
            
        return indices    

    # Migration function both Climate and Overpopulation
    def migration_simple(self, i, j):
        '''
        Simulates migration of population from a given cell.

        Parameters:
        i, j (int): The coordinates of the cell.

        Returns:
        float: The size of the population that migrates from the cell.
        '''
        # Do we need to leave
        if self.pop_dens[i, j] < self.migration_threshold*1.5 and self.type[i,j] == 0:
            return 0
        if self.pop_dens[i, j] <= 0:
            return 0
        # How many
        if self.type[i,j] == 0:
            #size = self.pop_dens[i,j] - self.migration_threshold
            size = self.pop_dens[i,j] - 0.5*self.migration_threshold        # Can be changed to make clearer images
            #size = self.pop_dens[i,j]
        elif self.type[i,j] != 0:
            size = self.pop_dens[i,j]
        # Where to
        destination_candidates = self.getMooreNeighbourhood(i, j, extent=self.view_distance)
        destinations = []
        for cell in destination_candidates:
            if self.type[cell[0], cell[1]] == 0:
                destinations.append(cell)
        # They die if no way to go
        if len(destinations) == 0:
            self.dead_migrants_current += size
            self.next_pop_dens[i, j] = 0
            return 0
        else:
            self.used_cells.append([i,j])
            random.shuffle(destinations)
            size /= len(destinations)
            for k in range(len(destinations)):
                self.next_pop_dens[destinations[k][0], destinations[k][1]] += size
                self.next_pop_dens[i,j] -= size
                self.migrants_current += size
        return size
    
    # Growth over the map
    def growth_simple(self, size):
        '''
        Simulates growth of population in the grid.

        Parameters:
        size (int): The size of population to grow.
        '''
        for k in range(size):
            i = random.randint(0, self.N-1)
            j = random.randint(0, self.N-1)
            # Only spawn on land
            if (self.type[i,j] == 0):
                self.pop_dens[i,j] += self.growthrate
    
    # Updating water level
    def upd_water_level(self):
        '''
        Raises water level with probability proportional to the emissions
        '''
        self.emissions = np.mean(self.pop_dens)*self.green_transition
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i,j] == 0: continue
                else:
                    if np.random.rand() < self.emissions:
                        neighbors = self.getVonNeumannNeighbourhood(i,j, extent = 1)
                        water_expansion_cell = neighbors[np.random.randint(len(neighbors))]
                        self.next_type[water_expansion_cell[0], water_expansion_cell[1]] = self.type[i,j]   # Pick random adjacent cell
        self.type = self.next_type
    
    # Updates stats for console
    def update_stats(self):
        '''
        Updates the statistics of the grid.

        Returns:
        float: The mean population density of the grid.
        '''
        pop_dens_sum = 0
        pop_dens_amount = 0
        for i in range(self.N):
            for j in range(self.N):
                # we have to calculate stats per each cell as some of them are filled with water
                if self.type[i,j] == 0:
                    pop_dens_sum += self.pop_dens[i,j]
                    pop_dens_amount += 1
        # Check for zero division
        if pop_dens_amount == 0: pop_dens_amount = 1
        return pop_dens_sum/pop_dens_amount

    def step(self):
        '''
        Constructs the self.nextgrid matrix based on the properties of self.grid
        Applies the Percolation Model Rules:
        
        1. Cells attempt to colonise their Moore Neighbourhood with probability P
        2. Cells do not make the attempt with probability 1-P
        '''
        
        # Updating water level and emissions
        self.upd_water_level()
        # Grow from Fixed Points
        self.growth_simple(int(self.N))                                     # Random Points
        #self.pop_dens[int(self.N/2), int(self.N/2)] += self.growthrate*np.sqrt(self.N)      # One Point Growth

        self.migrants_current = 0
        self.dead_migrants_current = 0
        loop = 0
        # Steps
        #for i in range(self.N):
        #    for j in range(self.N):
        #        self.migration_simple(i, j)
        #        self.pop_dens = self.next_pop_dens
        # All together changed during the step, to show avalanches and patterns more clearly
        while True:
            self.used_cells = []
            loop += 1
            migrants_temp = 0
            for i in range(self.N):
                for j in range(self.N):
                    migrants_temp += self.migration_simple(i, j)
                    self.pop_dens = self.next_pop_dens
            if migrants_temp < 1: break
            if loop > 100:
                print("Overlooped")                
                break
        if self.migrants_current < 0.01: self.migrants_current = 0
        self.migrants.append(self.migrants_current) 
        if self.dead_migrants_current < 0.01: self.dead_migrants_current = 0
        self.dead_migrants.append(self.dead_migrants_current)
        pop_dens_mean = self.update_stats()

        print()
        print("Population: ", pop_dens_mean)
        print("Emissions: ", self.emissions)
        if len(self.migrants) > 0:
            print("Displaced: ", self.migrants[-1])
        if len(self.dead_migrants) > 0:
            print("Diseased: ", self.dead_migrants[-1])
        print("\n")