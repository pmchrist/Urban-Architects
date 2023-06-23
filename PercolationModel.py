import numpy as np
import copy
from init_map2 import *


class PercolationModel2D(object):    
    '''
    Object that calculates and displays behaviour of 2D cellular automata
    '''

    # Just keeping global variables for reference:
    migration_threshold = 0.2
    burnrate = 0.1
    growthrate = 0.1
    emissions = 0.0     # Current emissions, not additive now
    emmigration_size = 0.1
    energy_replenish_chance = 0.2
    energy_replenish_size = 0.1
    energy_barrier = 0.8

    # Just describing global arrays of environment here:
    pop_dens = None     # Population density map
    energy = None       # Amount of energy map
    type = None         # Type of cell map
    fitness = None      # Fitness map
    # Maps for next step
    next_pop_dens = None
    next_energy = None

    # Some helper arrays for utility function
    water_score = None
    average_water_score = None


    # Function used to create water availability for fitness function
    def upd_available_water_map(self):
        self.water_score = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                # Getting neighbours
                neighbors = self.getMooreNeighbourhood(i,j, extent=2)
                w_counter = 0
                for neighbor in neighbors:
                    # Counting available water
                    if self.type[neighbor[0], neighbor[1]] == 1:
                        w_counter += 1
                    if self.type[neighbor[0], neighbor[1]] == 2:
                        w_counter += 0.5    # Sea water counts as half
                self.water_score[i, j] = w_counter
        self.average_water_score = np.mean(self.water_score)      # Average water availability score

    def init_grid(self):
        '''
        Initializes grid, which is a set of vectors which describe environment (for now randomly)
        
        Each point has some unique parameters, on which something is determined

        '''

        self.pop_dens = np.random.rand(self.N, self.N)       # population grid [0, 1]
        self.energy = np.random.rand(self.N, self.N)    # energy grid [0, 1]
        self.type = init_water_map(self.N, 0.2 , 2, 5, 1)

        # I believe this is too much, we can model existence/non existence of water just by changing cell type to land/toxic water etc.
        #self.water = np.zeros((self.N, self.N)) # grid to save water volumn value
        # loop to purge population values in water area
        # also loop to initialize water 'volumn' values
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i, j]!=0:
                    self.pop_dens[i, j] = 0
                    #self.water[i, j] = np.random.rand()

    def __init__(self, ni, temp):
        '''
        Constructor reads:
        N = side of grid
        
        produces N x N blank grid
        '''

        # Set environment
        self.temp = temp                # temperature
        self.emissions = 0              # current emissions
        self.N = ni                     # Size of 1 side
        self.Ntot = self.N*self.N       # Overall size
        self.init_grid()                # Initialize environment grid
        self.upd_available_water_map()  # Stores water availability score for utility function
        # Init grids for next step
        self.next_pop_dens = copy.deepcopy(self.pop_dens)
        self.next_energy = copy.deepcopy(self.energy)
        self.fitness = np.zeros((self.N, self.N))





    # We assume that neighborhoods stop at border (it is not rolling from another side)
    def getMooreNeighbourhood(self, i,j, extent=1):
        '''
        Returns a set of indices corresponding to the Moore Neighbourhood
        (These are the cells immediately adjacent to (i,j), plus those diagonally adjacent)
        '''

        # Check for incorrect input
        # Make it a test later
        if (i<0 or i>=self.N or j<0 or j>self.N):
            return ValueError
        
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
        Returns a set of indices corresponding to the Von Neumann Neighbourhood
        (These are the cells immediately adjacent to (i,j), but not diagonally adjacent)
        '''

        # Check for incorrect input
        # Make it a test later
        if (i<0 or i>=self.N or j<0 or j>self.N):
            return ValueError
        
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

    # Fitness
    # Function which values higher middle value
    def gaussian(self, val):
        return 1*np.exp((-(val - 0.5)**2/(0.05)))

    # Function which values higher high values
    def sigmoid(self, val):
        return 1/(1 + np.exp(-(val-0.5)/0.2))

    def update_fitness(self, i, j):
        # It is just made up for now
        # We are combining available energy/density/water availabiltiy
        fitness = 0
        if self.type[i,j] != 0: self.fitness[i,j] = 0      # people can't live on water
        else: 
            fitness += self.gaussian(self.pop_dens[i,j])
            fitness += self.sigmoid(self.energy[i,j])
            if self.water_score[i,j] > 3:
                fitness += 1
            elif self.water_score[i,j] > 1:
                fitness += 0.5
            fitness = fitness/3 # Getting Average
        self.fitness[i,j] = fitness


    # Emmigration
    def neighbour_feature(self, i, j):
        '''
        output: 1. sorted (descending) population values from the neighborhood.

        '''
        neighbors = self.getMooreNeighbourhood(i,j, extent=2)
        random.shuffle(neighbors)
        values = []
        for neighbor in neighbors:
            # Getting neighbours
            values.append(self.fitness[neighbor])
        return sorted(values, reverse=True)
    
    def emmigration(self, i, j, size = emmigration_size):
        '''
        size - is proportion of people who wants to leave
        '''
        if size > self.pop_dens[i,j]:
            self.next_pop_dens[i,j] = self.pop_dens[i,j]
            return

        destinations = self.getMooreNeighbourhood(i, j, extent=2)
        random.shuffle(destinations)
        for k in range(len(destinations)):
            if self.pop_dens[destinations[k][0], destinations[k][1]] < 1.0 - size and self.type[destinations[k][0], destinations[k][1]] == 0:
                self.next_pop_dens[destinations[k][0], destinations[k][1]] += size
                self.next_pop_dens[i,j] -= size
                return

    # Step Update
    def growth(self, i, j):
        '''
        burnrate = proportion of energy used by population
        growthrate = proportion of population growth if energy is sufficient
        '''

        # Growth
        if self.energy[i,j] > self.pop_dens[i,j]*self.burnrate:
            self.next_energy[i,j] -= self.pop_dens[i,j]*self.burnrate
            self.next_pop_dens[i,j] += self.pop_dens[i,j]*self.growthrate
            self.emissions += self.pop_dens[i,j]*self.burnrate
        # Decline
        else:
            self.next_pop_dens[i,j] -= self.pop_dens[i,j]*self.growthrate

    def upd_water_level(self):
        self.emissions = np.mean(self.pop_dens)
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i,j] == 0: continue
                else:
                    if np.random.rand() < self.emissions/10:
                        neighbors = self.getVonNeumannNeighbourhood(i,j, extent = 1)
                        random.shuffle(neighbors)
                        water_expansion_cell = neighbors[np.random.randint(len(neighbors))]
                        self.type[water_expansion_cell[0], water_expansion_cell[1]] = self.type[i,j]   # Pick random adjacent cell
                        self.emmigration(i,j,size=self.pop_dens[water_expansion_cell[0], water_expansion_cell[1]])

    def spawn_energy(self):
        for i in range(self.N):
            for j in range(self.N):
                if np.random.rand() < self.energy_replenish_chance and self.energy[i,j]<self.energy_barrier:
                    self.next_energy[i,j] += self.energy_replenish_size

    def fix_water_nomads(self):
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i, j]!=0:
                    destinations = self.getMooreNeighbourhood(i, j, extent=2)
                    random.shuffle(destinations)
                    size = self.pop_dens[i, j]
                    for k in range(len(destinations)):
                        if self.pop_dens[destinations[k][0], destinations[k][1]] < 1.0 - size and self.type[destinations[k][0], destinations[k][1]] == 0:
                            self.next_pop_dens[destinations[k][0], destinations[k][1]] += size
                            self.next_pop_dens[i,j] -= size
                            break
                    if k == len(destinations): self.pop_dens[i, j] = 0      # loop ended = nowhere to leave, they die :(

    # the part we can change later
    def step(self):
        '''
        Constructs the self.nextgrid matrix based on the properties of self.grid
        Applies the Percolation Model Rules:
        
        1. Cells attempt to colonise their Moore Neighbourhood with probability P
        2. Cells do not make the attempt with probability 1-P
        '''
        
        self.fix_water_nomads()
        self.upd_available_water_map()
        # We should vectorize this before final test on big map        
        for i in range(self.N):
            for j in range(self.N):
                # Find Utility
                self.update_fitness(i,j)
                # Migration
                if self.fitness[i,j] < self.migration_threshold:
                    self.emmigration(i,j)
                # Growth
                self.growth(i,j)
        self.upd_water_level()
        self.spawn_energy()

        print("Population: ", self.pop_dens.mean())
        print("Energy: ", self.energy.mean())
        print("Fitness: ", self.fitness.mean())
        print("\n")

        # Saving changes
        self.pop_dens = self.next_pop_dens
        self.energy = self.next_energy
