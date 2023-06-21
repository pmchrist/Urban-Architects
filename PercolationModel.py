import numpy as np

class PercolationModel2D(object):    
    '''
    Object that calculates and displays behaviour of 2D cellular automata
    '''
        
    def __init__(self, ni, temp):
        '''
        Constructor reads:
        N = side of grid
        
        produces N x N blank grid
        '''
        self.temp = temp                #temperature
        self.N = ni                     # Size of 1 side
        self.Ntot = self.N*self.N       # Overall size
        # Much more complex initializations for grid can be implemented later       
        # Each point has some unique parameters, on which something is determined

        # step grid
        self.pop = np.random.rand(self.N,self.N) # population grid
        self.energy = np.random.rand(self.N,self.N) # energy grid
        self.type = np.zeros((self.N, self.N)) # type - fresh water, sea water, and land
        # type 0-land where people can live on; 1-fresh water; 2-sea water
        self.water = np.zeros((self.N, self.N)) # grid to save water volumn value

        # Here is a random initialization of self.type (will be replaced by a map)
        row_sum = 2
        for i in range(row_sum):
            self.type.ravel()[i::self.type.shape[1]+row_sum] = 1
        np.random.shuffle(self.type)
        for i in range(row_sum):
            self.type.ravel()[i::self.type.shape[1]+row_sum] = 2
        np.random.shuffle(self.type)

        # loop to purge population values in water area
        # also loop to initialize water 'volumn' values
        for i in range(self.N):
            for j in range(self.N):
                if self.type[i, j]!=0:
                    self.pop[i, j] = 0
                    self.water[i, j] = np.random.rand()

        
        # Everything is implemented in this style instead of classes to increase performance

    # We assume that neighborhood is finishing at border (it is not rolling from another side)
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
    
    def neighbour_feature(self, i, j):
        '''
        output: 1. sorted (descending) population values from the neighborhood.
                2. number of water cells in the neighborhood (with penalty to sea water)

        '''
        neighbors = self.getMooreNeighbourhood(i,j, extent=1)
        values = []
        w_counter = 0
        for neighbor in neighbors:
            values.append(self.pop[neighbor])
            # penalty to sea water
            if self.type[neighbor] == 1:
                w_counter += 1
            if self.type[neighbor] == 2:
                w_counter += 0.5
        return sorted(values, reverse=True), w_counter
    
    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))

    def fitness(self, i,j):
        '''
        return fitness value according to the cell population density.
        '''
        # curve that shows stress with growing density / give penalty to crowdness
        def curve(x, pivot):
            if x > pivot:
                return 1/(x-pivot)
            else:
                return 1
            
        # Check cell type
        if self.type[i, j] != 0:
            pass
        
        # self density score
        y1 = curve(self.pop[i, j], 0.4)

        neighbors, w = self.neighbour_feature(i, j)
        # give fitness score if we have water nearby
        if w<=4:
            y2 = 0.1 * w
        else:
            y2=0.4

        # give penalty to the fitness score if we have crowded neighborhood
        # calculate average density of the neighborhood
        mean_density = np.mean(neighbors)
        y3 = curve(mean_density, 0.4)

        # return normalized y value
        # return self.sigmoid(y1 + y2 + y3)

        '''
        sigmoid deal with value in [-inf, inf], in this case y in [0, 3]
        I don't think sigmoid is a proper normalization function here
        for now, I just divide the sum value by max value
        '''
        return (y1+y2+y3)/2.4
    
    def new_pop(self, i, j):
        '''
        population grow if cell density < threshold value
        otherwise people in this cell move to a less crowded cell, or move out from the map(or we can say death rate> birth rate)
        new population = f(self.pop, fitness, etc.)
        '''
        if (self.pop[i,j] < 0.4):
            return self.pop[i,j]*1.2

        # Example 2, compare with neighborhood
        neighbor_threshold = 0.6
        cell_neighborhood = self.getMooreNeighbourhood(i, j)
        # 
        # Find sum of pop in the neighborhood
        # Better to simplify this thing into lambda or create functions for all rules
        neighbor_sum_pop = 0
        for neighbor in cell_neighborhood:
            neighbor_sum_pop += self.pop[neighbor[0], neighbor[1]]
        # Checking rule based on neighbors
        if (neighbor_sum_pop) > neighbor_threshold:
            return self.pop[i,j]*0.8 
 
    def new_energy(self, i, j):
        '''
        calculate the energy consumption(Delta energy) of the cell in the current step
        Assume delta energy is proportional to the delta CO2 emmision for now
        (people might use more clean energy with the development of the city)
        '''
        # some resources are renewable, if no energy renew term added to the value, it will drain out
        return self.energy[i, j] - self.pop[i, j] * 0.05 + self.energy[i, j] * 0.05

    def new_water(self, i, j, d_temp):
        '''
        imput: cell position (i, j); delta temperature
        calculate the delta water of the cell in the current step
        '''
        # log10(temp) in [1, 2] (assume average temp in [10, 40])
        # we should be causious about temp <10
        # delta_water = 0.1
        delta_water = 0.001*np.log10(100*d_temp+1)
        if self.type[i, j]==1: # fresh water (river dries)
            self.water[i, j] -= delta_water
        elif self.type[i, j]==2: # sea water(sea level increase)
            self.water[i, j] += delta_water
        return self.water[i, j]
    
    def new_temp(self, temp):
        # we can set rules for changing temperature here
        return temp

    # the part we can change later
    def step(self):
        '''
        Constructs the self.nextgrid matrix based on the properties of self.grid
        Applies the Percolation Model Rules:
        
        1. Cells attempt to colonise their Moore Neighbourhood with probability P
        2. Cells do not make the attempt with probability 1-P
        '''

        # Placeholder for next step's grid
        self.next_pop = np.zeros((self.N,self.N))       # Current step grid
        self.next_energy = np.zeros((self.N, self.N))
        self.next_water = np.zeros((self.N, self.N))

        # We should vectorize this before final test on big map
        for i in range(self.N):
            for j in range(self.N):

                # # Here we define our rules
                # # population updating rule
                # # Example 1, compare current value
                # if (self.pop[i,j] < 0.4):
                #     self.next_pop[i,j] = self.pop[i,j]*1.2
        
                # # Example 2, compare with neighborhood
                # neighbor_threshold = 0.6
                # cell_neighborhood = self.getMooreNeighbourhood(i, j)
                # # 
                # # Find sum of pop in the neighborhood
                # # Better to simplify this thing into lambda or create functions for all rules
                # neighbor_sum_pop = 0
                # for neighbor in cell_neighborhood:
                #     neighbor_sum_pop += self.pop[neighbor[0], neighbor[1]]
                # # Checking rule based on neighbors
                # if (neighbor_sum_pop) > neighbor_threshold:
                #     self.next_pop[i,j] = self.pop[i,j]*0.8 

                # # update temperature
                self.next_temp = self.new_temp(self.temp)
                d_temp = self.next_temp - self.temp
                
                # update cell values
                if self.type[i, j] == 0:
                    self.next_energy[i, j] = self.new_energy(i, j)
                    self.next_pop[i, j] = self.new_pop(i, j)
                else:
                    self.next_water[i, j] = self.new_water(i, j, d_temp)



        
        # Saving changes
        self.pop = self.next_pop
        self.energy = self.next_energy
        self.water = self.next_water
        self.temp = self.next_temp
      
