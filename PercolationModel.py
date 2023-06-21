import numpy as np

class PercolationModel2D(object):    
    '''
    Object that calculates and displays behaviour of 2D cellular automata
    '''
        
    def __init__(self, ni):
        '''
        Constructor reads:
        N = side of grid
        
        produces N x N blank grid
        '''
        
        self.N = ni                     # Size of 1 side
        self.Ntot = self.N*self.N       # Overall size
        # Much more complex initializations for grid can be implemented later       
        # Each point has some unique parameters, on which something is determined
        self.grid_param1 = np.random.rand(self.N,self.N)            # Current step grid

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
    
    def self_fitness(self, i,j):
        '''
        return fitness value according to the cell population density.
        '''
        x = self.grid_param1[i,j]
        if x < 0.4:
            y = 1
        else:
            y = 1/(x-0.4)
        return y
    
    def neighbour_fitness(self, i, j, matrix):
        '''
        return sorted values from the neighborhood.
        '''
        neighbors = self.getMooreNeighbourhood(i,j, extent=1)
        values = []
        for neighbor in neighbors:
            values.append(matrix[neighbor])

        return sorted(values, reverse=True)
    
    # the part we can change later
    def step(self):
        '''
        Constructs the self.nextgrid matrix based on the properties of self.grid
        Applies the Percolation Model Rules:
        
        1. Cells attempt to colonise their Moore Neighbourhood with probability P
        2. Cells do not make the attempt with probability 1-P
        '''

        # Placeholder for next step's grid
        self.next_grid_param1 = np.zeros((self.N,self.N))       # Current step grid

        # We should vectorize this before final test on big map
        for i in range(self.N):
            for j in range(self.N):

                # Here we define our rules

                # Example 1, compare current value
                if (self.grid_param1[i,j] < 0.4):
                    self.next_grid_param1[i,j] = self.grid_param1[i,j]*1.2
        
                # Example 2, compare with neighborhood
                neighbor_threshold = 0.6
                cell_neighborhood = self.getMooreNeighbourhood(i, j)
                # 
                # Find sum of param_1 in the neighborhood
                # Better to simplify this thing into lambda or create functions for all rules
                neighbor_sum_param1 = 0
                for neighbor in cell_neighborhood:
                    neighbor_sum_param1 += self.grid_param1[neighbor[0], neighbor[1]]
                # Checking rule based on neighbors
                if (neighbor_sum_param1) > neighbor_threshold:
                    self.next_grid_param1[i,j] = self.grid_param1[i,j]*0.8 
        
        # Saving changes
        self.grid_param1 = self.next_grid_param1
      
