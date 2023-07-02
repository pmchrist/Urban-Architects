from threading import local
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class BakSneppen2D_PD(object):
    """
    A class representing a 2D Bak-Sneppen model with a defined population density,
    accounting for various parameters that contribute to the evolution.
    """

    def __init__(self, size, save_folder, alpha, beta, gamma, delta, labda):
        """
        Initializes the BakSneppen2D object with the given parameters.

        Parameters:
        size (int): The size of the 2D system.
        save_folder (str): The directory to save output images.
        alpha, beta, gamma, delta (float): Parameters contributing to the evolution of the system.
        labda (float): The mutation rate of the system.
        """

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
        """
        Returns a set of indices corresponding to the Moore Neighbourhood.

        Parameters:
        i, j (int): The coordinates of the current cell.
        extent (int): The range for the Moore Neighbourhood. Default is 1.

        Returns:
        list: A list of indices representing the Moore Neighbourhood.
        """

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
        """
        Calculates the average population density of a given set of cells.

        Parameters:
        indices (list): List of indices of cells.

        Returns:
        float: The average population density.
        """
        neighbour_populations = [self.system[cell[0], cell[1]] for cell in indices]
        return sum(neighbour_populations) / len(neighbour_populations)
    
    def fitness(self, density_ij, average_density_ij):
        """
        Calculates the fitness of a cell given its density and the average density of the system.

        Parameters:
        density_ij (float): The density of the cell.
        average_density_ij (float): The average density of the system.

        Returns:
        float: The calculated fitness.
        """
        return 0 * np.exp(-self.labda * (density_ij - average_density_ij) ** 2) + 1 * self.gaussian(density_ij, 0.5, 0.1)

    def new_density(self, average_density, local_density, fitness, random_factor):
        """
        Calculates the new density of a cell given the average density of the system,
        its current density, fitness, and a random factor.

        Parameters:
        average_density (float): The average density of the system.
        local_density (float): The current density of the cell.
        fitness (float): The fitness of the cell.
        random_factor (float): A random factor contributing to the density.

        Returns:
        float: The new density of the cell.
        """
        assert self.alpha + self.beta + self.gamma + self.delta == 1
        return self.alpha * average_density + self.beta * local_density + self.gamma * fitness + self.delta * random_factor
        

    def update_system(self):
        """
        Updates the system by recalculating the densities of the cells based on their fitness.
        """

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
        """
        Simulates the system for a given number of iterations, updating the system and storing its properties.

        Parameters:
        iterations (int): The number of iterations to run the simulation.
        """
        self.plot_system(0, initial=True)
        for iteration in range(iterations):
            self.update_system()
            self.store_system_properties()

            if iteration % 10 == 0:
                self.plot_system(iteration)

    def plot_system(self, iteration, initial=False, close=True):
        """
        Plots the system at a given iteration.

        Parameters:
        iteration (int): The iteration number.
        initial (bool): A flag indicating whether the system is at its initial state. Default is False.
        close (bool): A flag indicating whether to close the plot after saving. Default is True.
        """
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
        
        if close:
            plt.close()

    def store_system_properties(self):
        """
        Stores the system properties by appending the minimum fitness to the min_fitness list.
        """
        self.min_fitness.append(np.min(self.system))


if __name__=="__main__":
    save_folder = 'BakSneppen_population_results'
    size = 100

    iterations = 1000

    model = BakSneppen2D_PD(size, save_folder, 0, 0, 0.9, 0.1, 10)
    print(model.system)
    model.simulate(iterations)
    print(model.system)

    plt.plot(range(iterations), model.min_fitness)
    plt.show()