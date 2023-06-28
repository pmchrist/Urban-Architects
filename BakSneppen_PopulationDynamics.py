from threading import local
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class BakSneppen2D(object):

    def __init__(self, size, save_folder, alpha, beta, gamma, delta, labda):

        # use a seed for repeatability
        # np.random.seed(2)

        # initialize the system with random population density values
        self.size = size
        self.system = np.random.rand(size, size)

        # set global parameters
        self.save_folder = save_folder

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.labda = labda

        # initialize lists for storing interesting parameter values
        self.min_fitness = []

    def getMooreNeighbourhood(self, i,j, extent=1):
        '''
        Returns a set of indices corresponding to the Moore Neighbourhood
        (These are the cells immediately adjacent to (i,j), plus those diagonally adjacent)
        '''

        # Check for incorrect input
        # Make it a test later
        assert i >= 0 and i <= self.size
        assert j >= 0 and j <= self.size
        
        indices = []
        
        for iadd in range(i-extent,i+extent+1):
            for jadd in range(j-extent, j+extent+1):        
                if(iadd==i and jadd == j):  # If is not the same cell
                    continue
                if (iadd < 0 or iadd>=self.size):          # If is not in bounds
                    continue                
                if (jadd < 0 or jadd>=self.size):          # If is not in bounds
                    continue
                indices.append([iadd,jadd])

        return indices

    # Function which values higher middle value
    def gaussian(self, val, mean, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((val - mean)/ std)**2)

    def average_density(self, indices):
        neighbour_populations = [self.system[cell[0], cell[1]] for cell in indices]
        return sum(neighbour_populations) / len(neighbour_populations)
    
    def fitness(self, density_ij, average_density_ij):
        return 0 * np.exp(-self.labda * (density_ij - average_density_ij) ** 2) + 1 * self.gaussian(density_ij, 0.5, 0.1)

    def new_density(self, average_density, local_density, fitness, random_factor):
        assert self.alpha + self.beta + self.gamma + self.delta == 1
        return self.alpha * average_density + self.beta * local_density + self.gamma * fitness + self.delta * random_factor
        

    def update_system(self):

        # get the indices of the lowest fitness value
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        i, j = min_indices

        neighbours = self.getMooreNeighbourhood(i, j)
        random.shuffle(neighbours)
        average_density = self.average_density(neighbours)
        local_density = self.system[i, j]
        fitness = self.fitness(local_density, average_density)

        number_of_neighbours = len(neighbours)

        random_factors = np.random.rand(number_of_neighbours + 1)

        for index in range(number_of_neighbours + 1):
            next_density = max(self.new_density(average_density, local_density, fitness, random_factors[index]), 0)
            if index == number_of_neighbours:
                self.system[i, j] = next_density
            else:
                self.system[neighbours[index][0], neighbours[index][1]] = next_density

        # give a new fitness value to the cell with the lowest value
        # self.system[i, j] = np.random.rand()

        # # also update the neighbors, where we use periodic boundary conditions
        # self.system[(i - 1) % self.size, j] =  np.random.rand()
        # self.system[(i + 1) % self.size, j] =  np.random.rand()
        # self.system[i, (j - 1) % self.size] =  np.random.rand()
        # self.system[i, (j + 1) % self.size] =  np.random.rand()
    
    def simulate(self, iterations):
        self.plot_system(0, initial=True)
        for iteration in range(iterations):
            self.update_system()
            self.store_system_properties()

            if iteration % 10 == 0:
                self.plot_system(iteration)

    def plot_system(self, iteration, initial=False):
        plt.imshow(self.system, cmap='hot', origin='lower')
        plt.colorbar(label='Population density')
        plt.clim(0, 1)
        if initial:
            plt.title("Initial configuration")
        else:
            plt.title(f'Bak-Sneppen model in 2D (iteration {iteration + 1})')
        plt.xlabel('Column')
        plt.ylabel('Row')
        if initial:
            plt.savefig(os.path.join(self.save_folder, f'iteration_{iteration}.png'))
        else:
            plt.savefig(os.path.join(self.save_folder, f'iteration_{iteration + 1}.png'))
        plt.close()

    def store_system_properties(self):
        self.min_fitness.append(np.min(self.system))


if __name__=="__main__":
    save_folder = 'BakSneppen_population_results'
    size = 100

    iterations = 1000

    model = BakSneppen2D(size, save_folder, 0, 0, 0.9, 0.1, 10)
    print(model.system)
    model.simulate(iterations)
    print(model.system)

    plt.plot(range(iterations), model.min_fitness)
    plt.show()