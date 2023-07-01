import numpy as np
import copy
from init_map2 import *


class PercolationModel2D(object):    
    '''
    Object that calculates and displays behaviour of 2D cellular automata
    '''

    # Just keeping global variables for reference:
    migration_threshold = 0.1       # Without simple migration all of the environment events much more clear
    growthrate = 0.1    # Energy Consumptio/Growth Value
    emissions = 0.0     # Current emissions, NOT Additive 
    emmigration_size = 0.1
    energy_replenish_chance = 0.4
    energy_replenish_size = 0.4
    energy_barrier = 0.8
    view_distance = 3    # Quarter Map

    # Just describing global arrays of environment here:
    pop_dens = None     # Population density map
    energy = None       # Amount of energy map
    type = None         # Type of cell map
    fitness = None      # Fitness map
    # Maps for next step
    next_pop_dens = None
    next_energy = None
    next_type = None

    # Some helper arrays for utility function
    water_score = None

    climate_migration_displaced = []
    climate_migration_dead = []
    simple_migration = []


    def init_grid(self):
        '''
        Initializes grid, which is a set of vectors which describe environment (for now randomly)
        
        Each point has some unique parameters, on which something is determined

        '''

        #self.pop_dens = np.random.rand(self.N, self.N)       # population grid [0, 1]
        self.pop_dens = np.zeros((self.N, self.N))       # population grid [0, 1]
        self.energy = np.random.rand(self.N, self.N)    # energy grid [0, 1]
        self.type = init_water_map(self.N, 0.2 , 2, 5, 1)
        #self.type = np.zeros((self.N, self.N))       # population grid [0, 1]
        self.fitness = np.zeros((self.N, self.N))

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
        self.next_type = copy.deepcopy(self.type)


    # Helper Functions
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

    # Function which values higher high values
    def inverse_poly(self, val):
        return 4*val-4*val**2

    # Function which values higher middle value
    def gaussian(self, val):
        return 1*np.exp((-(val - 0.5)**2/(0.05)))

    # Function which values higher high values
    def sigmoid(self, val):
        return 1/(1 + np.exp(-(val-0.5)/0.2))


    # Fitness functions
    # Returns neighbors based on their fitness
    def neighbour_feature(self, i, j, extent=5):
        '''
        output: cells sorted (descending) by fitness values.

        '''
        neighbors = self.getMooreNeighbourhood(i,j, extent)
        values = []
        for neighbor in neighbors:
            values.append(-self.fitness[neighbor[0], neighbor[1]])
        sort = np.argsort(values)
        sorted_neighbors = []
        for k in sort:
            sorted_neighbors.append(neighbors[k][:])
        return(sorted_neighbors)
    
    # Function which creates map of water availability for fitness function
    def upd_available_water_map(self):
        '''
        This is a helper function to calculate Fitness. It assigns score based on water quality and availability
        '''
        self.water_score = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                # Getting neighbours
                neighbors = self.getMooreNeighbourhood(i,j, extent=self.view_distance*2)
                w_counter = 0
                for neighbor in neighbors:
                    # Counting available water
                    if self.type[neighbor[0], neighbor[1]] == 1:
                        w_counter += 1.0
                    if self.type[neighbor[0], neighbor[1]] == 2:
                        w_counter += 0.5    # Sea water counts as half
                self.water_score[i, j] = w_counter/(self.view_distance**4)

    # Can be easily improved
    def update_fitness(self):
        # It is just made up for now
        # We are combining available energy/density/water availabiltiy
        self.fitness = np.zeros((self.N, self.N))   # Fitness map
        for i in range(self.N):
            for j in range(self.N):
                fitness = 0
                if self.type[i,j] != 0: self.fitness[i,j] = 0      # people can't live on water
                else:
                    neighbors = self.getMooreNeighbourhood(i,j, extent=self.view_distance)
                    pop_dens_temp = 0
                    energy_temp = 0
                    #water_temp = 0
                    for neighbor in neighbors:
                        pop_dens_temp += self.pop_dens[neighbor[0], neighbor[1]]
                        energy_temp += self.energy[neighbor[0], neighbor[1]]
                        #water_temp += self.water_score[neighbor[0], neighbor[1]]
                    pop_dens_temp /= len(neighbors)
                    energy_temp /= len(neighbors)
                    #water_temp /= len(neighbors)
                    fitness += self.gaussian(pop_dens_temp)
                    fitness += self.sigmoid(energy_temp)
                    #fitness += self.sigmoid(water_temp)
                    #fitness = fitness/3 # Getting Average
                    fitness = fitness/2 # Getting Average
                    self.fitness[i,j] = fitness

    # Emigration which covers land to land
    def land_migration(self):
        simple_migration = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i,j] == 0 and self.fitness[i,j] < self.migration_threshold and self.pop_dens[i,j] > self.emmigration_size:
                    destinations = self.neighbour_feature(i, j, extent=self.view_distance)
                    for k in range(len(destinations)):
                        if self.type[destinations[k][0], destinations[k][1]] == 0:
                            self.next_pop_dens[destinations[k][0], destinations[k][1]] += self.emmigration_size
                            self.next_pop_dens[i,j] -= self.emmigration_size
                            simple_migration += self.emmigration_size
                            if self.next_pop_dens[i,j] < 0:       # So that we do not have negative people
                                self.next_pop_dens[i,j] = 0
                            break
                        if k == len(destinations)-1:       # loop ended = nowhere to leave, they die :(
                            climate_migration_dead += size
                            self.next_pop_dens[i, j] = 0
        self.simple_migration.append(simple_migration)
        self.pop_dens = self.next_pop_dens

    def climate_emmigration(self):
        '''
        size - is proportion of people who wants to leave
        People have to be displaced because of water rise, If there is no spot to go, they die

        '''
        climate_migration_displaced = 0
        climate_migration_dead = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i,j] != 0 and self.pop_dens[i,j] > 0:     # Only people in the water are eligible
                    size = self.pop_dens[i,j]
                    destinations = self.neighbour_feature(i, j, extent=self.view_distance)
                    for k in range(len(destinations)):
                        # Should play with border for capacity
                        if self.type[destinations[k][0], destinations[k][1]] == 0:
                            self.next_pop_dens[destinations[k][0], destinations[k][1]] += size
                            self.next_pop_dens[i,j] -= size
                            climate_migration_displaced += size
                            if self.next_pop_dens[i,j] < 0:       # So that we do not have negative people
                                self.next_pop_dens[i,j] = 0
                            break
                        if k == len(destinations)-1:       # loop ended = nowhere to leave, they die :(
                            climate_migration_dead += size
                            self.next_pop_dens[i, j] = 0
        # Saving Changes
        self.climate_migration_displaced.append(climate_migration_displaced)
        self.climate_migration_dead.append(climate_migration_dead)
        self.pop_dens = self.next_pop_dens

    # Step Update
    def growth(self):
        '''
        Simulates population growth/decline based on the available resources

        growthrate = proportion of energy used by population
        growthrate = proportion of population growth if energy is sufficient
        '''
        emissions = 0
        for i in range(self.N):
            for j in range(self.N):
                # Growth
                if self.energy[i,j] > self.growthrate:
                    self.next_energy[i,j] -= self.growthrate
                    self.next_pop_dens[i,j] += self.growthrate
                    emissions += self.growthrate
                # Decline
                else:
                    self.next_pop_dens[i,j] /= 2
        self.emissions = emissions/self.Ntot
        self.energy = self.next_energy
        self.pop_dens = self.next_pop_dens  # Update Map

    def upd_water_level(self):
        '''
        Raises water level with probability proportional to the emissions
        '''
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i,j] == 0: continue
                else:
                    if np.random.rand() < self.emissions:
                        neighbors = self.getVonNeumannNeighbourhood(i,j, extent = 1)
                        random.shuffle(neighbors)
                        water_expansion_cell = neighbors[np.random.randint(len(neighbors))]
                        self.next_type[water_expansion_cell[0], water_expansion_cell[1]] = self.type[i,j]   # Pick random adjacent cell
        self.type = self.next_type

    def spawn_energy(self):
        '''
        Just replenishes energy on the map, the higher the consumption, lower the replenishment
        '''
        for i in range(self.N):
            for j in range(self.N):
                if np.random.rand() < self.energy_replenish_chance and self.energy[i,j]<self.energy_barrier:
                    self.next_energy[i,j] += self.energy_replenish_size * (1-self.emissions)
        self.energy = self.next_energy

    def update_stats(self):
        fitness_sum = 0
        fitness_amount = 0
        pop_dens_sum = 0
        pop_dens_amount = 0
        energy_sum = 0
        energy_amount = 0
        for i in range(self.N):
            for j in range(self.N):
                # we have to calculate stats per each cell as some of them are filled with water
                if self.type[i,j] == 0:
                    fitness_sum += self.fitness[i,j]
                    fitness_amount += 1
                    pop_dens_sum += self.pop_dens[i,j]
                    pop_dens_amount += 1
                    energy_sum += self.energy[i,j]
                    energy_amount += 1
        # Check for zero division
        return pop_dens_sum/pop_dens_amount, energy_sum/energy_amount, fitness_sum/fitness_amount


    # the part we can change later
    def step(self):
        '''
        Constructs the self.nextgrid matrix based on the properties of self.grid
        Applies the Percolation Model Rules:
        
        1. Cells attempt to colonise their Moore Neighbourhood with probability P
        2. Cells do not make the attempt with probability 1-P
        '''
        
        # Find Utility and Initial Migration
        # Emissions, Sea Rise
        self.emissions = 0
        self.growth()
        self.upd_water_level()
        self.update_fitness()

        loop = 0
        while True:
            loop += 1
            migrants_temp = 0
            self.climate_emmigration()
            self.land_migration()
            migrants_temp += self.simple_migration[-1] + self.climate_migration_displaced[-1] + self.climate_migration_dead[-1]
            if migrants_temp < 1: break
            if loop > 100:
                print("Overlooped")                
                break

        # Replenishing energy
        self.spawn_energy()
        pop_dens_mean, energy_mean, fitness_mean = self.update_stats()

        print()
        print("Population: ", pop_dens_mean)
        print("Energy: ", energy_mean)
        print("Fitness: ", fitness_mean)
        print("Emissions: ", self.emissions)
        if len(self.simple_migration) > 0:
            print("Displaced by Fitness: ", self.simple_migration[-1])
        if len(self.climate_migration_displaced) > 0:
            print("Displaced by Climate: ", self.climate_migration_displaced[-1])
        if len(self.climate_migration_dead) > 0:
            print("Killed by Climate: ", self.climate_migration_dead[-1])
        print("\n")