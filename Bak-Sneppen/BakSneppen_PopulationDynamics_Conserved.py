from threading import local
import numpy as np
import random
import matplotlib.pyplot as plt
import os

class BakSneppen2D_PDC(object):
    """
    A class representing a 2D Bak-Sneppen model with a conserved population density.
    It considers various parameters that contribute to the evolution of the system and fitness function.
    """

    def __init__(self, size, save_folder, alpha, beta, gamma, delta, labda, fitness_mean, fitness_std, gaussian_weight):
        """
        Initializes the BakSneppen2D_ConservedPopulation object with the given parameters.

        Parameters:
        size (int): The size of the 2D system.
        save_folder (str): The directory to save output images.
        alpha, beta, gamma, delta (float): Parameters contributing to the evolution of the system.
        labda (float): The mutation rate of the system.
        fitness_mean (float): The mean for the Gaussian function used in fitness calculation.
        fitness_std (float): The standard deviation for the Gaussian function used in fitness calculation.
        gaussian_weight (float): The weight for the Gaussian function in the fitness calculation.
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

        # set parameters for fitness function
        self.labda = labda
        self.fitness_mean = fitness_mean
        self.fitness_std = fitness_std
        self.gaussian_weight = gaussian_weight


        # initialize lists for storing interesting parameter values
        self.min_fitness = []
        self.system_size = []

        self.migrations = []

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
        return (1 - self.gaussian_weight) * np.exp(-self.labda * (density_ij - average_density_ij) ** 2) + self.gaussian_weight * self.gaussian(density_ij, self.fitness_mean, self.fitness_std)

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
        Considers the migrations between neighbouring cells and preserves the overall population.
        """

        # get the indices of the lowest fitness value
        min_value = np.min(self.system[self.system != 0])
        i, j = np.where(self.system == min_value)

        neighbours = self.getMooreNeighbourhood(i[0], j[0])
        # print(f"Neighbours: {neighbours}")
        average_density = self.average_density(neighbours)
        # print(f"Average density: {average_density}")
        local_density = self.system[i, j]
        # print(f"Local density: {local_density}")
        fitness = self.fitness(local_density, average_density)
        # print(f"Fitness: {fitness}")

        random_factor = np.random.rand()

        number_of_neighbours = len(neighbours)
        neighbour_values = [self.system[neighbour[0], neighbour[1]] for neighbour in neighbours]
        # print(f"Neighbour_values: {neighbour_values}")

        next_density = max(self.new_density(average_density, local_density, fitness, random_factor), 0)
        density_difference = self.system[i, j] - next_density
        # print(f"Density difference: {density_difference}")
        neighbour_change = (density_difference) / number_of_neighbours
        # print(f"Neighbour change: {neighbour_change}")

        # if density_difference > 0: people move out of the cell into neighbouring cells
        # if density_difference < 0: people from neighbouring cells move into our current cell
        if density_difference < 0:
            # we have _ cases:
                # 1. all neighbouring cells have high enough densities, in which case we'll just move the same number of people from each cell
                # 2. a few neighbouring cells have probability less than required, but the sum of the neighbours is still enough to cover the change
                # 3. the sum is not enough to cover the change, in which case we'll move all people from neighbouring states to the cell

            if all(neighbour_value >= abs(neighbour_change) for neighbour_value in neighbour_values):
                # print("Case 1: all neighbour values are high enough")
                for neighbour in neighbours:
                    self.system[neighbour[0], neighbour[1]] += neighbour_change
                
            elif not all(neighbour_value >= abs(neighbour_change) for neighbour_value in neighbour_values) and sum(neighbour_values) >= abs(density_difference):
                # print(f"Case 2: not all high enough, but the sum is: {sum(neighbour_values)}")
                neighbour_values, neighbours = zip(*sorted(zip(neighbour_values, neighbours)))
                neighbour_changes = [neighbour_change for _ in range(len(neighbours))]
                for neighbour_index in range(len(neighbours)):
                    if neighbour_values[neighbour_index] < neighbour_changes[neighbour_index]:
                        neighbour_changes[neighbour_index + 1] -= (neighbour_changes[neighbour_index] - neighbour_values[neighbour_index])
                        self.system[neighbours[neighbour_index][0], neighbours[neighbour_index][1]] = 0
                    else:
                        self.system[neighbours[neighbour_index][0], neighbours[neighbour_index][1]] -= neighbour_changes[neighbour_index]
            
            else:
                # print(f"Case 3: sum is not high enough: {sum(neighbour_values)}")
                next_density = self.system[i, j] + sum(neighbour_values)
                for neighbour in neighbours:
                    self.system[neighbour[0], neighbour[1]] = 0

        else:
            for neighbour in neighbours:
                self.system[neighbour[0], neighbour[1]] += neighbour_change
        
        # do we need to see the absolute values?
        self.migrations.append(abs(self.system[i, j] - next_density)[0])
        self.system[i, j] = next_density
    
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

            if iteration % 100 == 0:
                self.plot_system(iteration)

    def plot_system(self, iteration, initial=False, close=True):
        """
        Plots the system at a given iteration.

        Parameters:
        iteration (int): The current iteration.
        initial (bool): Whether the plot is for the initial configuration of the system. Default is False.
        close (bool): Whether to close the plot after saving. Default is True.
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
        else:
            plt.show()

    def store_system_properties(self):
        """
        Stores the current properties of the system.
        """
        self.min_fitness.append(np.min(self.system))
        self.system_size.append(np.sum(self.system))


if __name__=="__main__":
    save_folder = 'BakSneppen_population_conserved_results'
    size = 100

    iterations = 10001

    model = BakSneppen2D_PDC(size, save_folder, 0, 0, 1, 0, 10, 0.5, 0.19, 1)
    # print(model.system)
    model.simulate(iterations)
    # print(model.system)

    plt.plot(range(iterations), model.min_fitness)
    plt.show()

    plt.plot(range(iterations), model.system_size)
    plt.show()

    plt.plot(range(iterations), model.migrations)
    plt.show()

    # print(model.migrations)
    plt.hist(model.migrations, bins=40)
    plt.show()