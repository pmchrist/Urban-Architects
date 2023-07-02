import numpy as np
import random
import matplotlib.pyplot as plt
import os
from esda.moran import Moran


class BakSneppen2D(object):
    """
    BakSneppen2D class that represents a 2D Bak-Sneppen model
    """

    def __init__(self, size, save_folder):
        """
        Initializes the BakSneppen2D object with given size and save_folder.

        Parameters:
        size (int): The size of the 2D system.
        save_folder (str): The directory to save output images.
        """
        
        np.random.seed(2)
        assert isinstance(size, int), "Size should be an integer"
        # initialize the system with random population density values
        self.size = size
        self.system = np.random.rand(size, size)
        

        # set global parameters
        self.save_folder = save_folder

        # initialize lists for storing interesting parameter values
        self.min_fitness = []
        self.avg_fitness = [] 
        self.std_fitness = []
        self.least_fit_location = []


    def update_system(self):
        """
        Updates the system by identifying the cell with the lowest fitness value and replacing it with a new random value.
        Also updates the fitness of neighboring cells.
        """

        # get the indices of the lowest fitness value
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        min_value = self.system[min_indices]
        i, j = min_indices

        # give a new fitness value to the cell with the lowest value
        self.system[i, j] = np.random.rand()

        assert self.system[min_indices] != min_value, "System did not update"

        # # also update the neighbors, where we use periodic boundary conditions
        self.system[(i - 1) % self.size, j] =  np.random.rand()
        self.system[(i + 1) % self.size, j] =  np.random.rand()
        self.system[i, (j - 1) % self.size] =  np.random.rand()
        self.system[i, (j + 1) % self.size] =  np.random.rand()

    
    def simulate(self, iterations):
        """
        Simulates the system for a given number of iterations.

        Parameters:
        iterations (int): The number of iterations to run the simulation.
        """
        for iteration in range(iterations):
            self.update_system()
            self.store_system_properties()

            if iteration % 10 == 0:
                self.plot_system(iteration)

    def plot_system(self, iteration):
        """
        Plots the system at a given iteration.

        Parameters:
        iteration (int): The iteration number.
        """

        plt.imshow(self.system, cmap='hot', origin='lower')
        plt.colorbar(label='Fitness')
        plt.title(f'Bak-Sneppen model in 2D (iteration {iteration + 1})')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(os.path.join(self.save_folder, f'iteration_{iteration + 1}.png'))
        plt.close()

    def store_system_properties(self):
        """
        Stores the system properties such as min_fitness, avg_fitness, std_fitness, least_fit_location and distances.
        """
        min_fitness = np.min(self.system)
        avg_fitness = np.mean(self.system)
        std_fitness = np.std(self.system)
        
        self.min_fitness.append(min_fitness)
        self.avg_fitness.append(avg_fitness)
        self.std_fitness.append(std_fitness)

        self.least_fit_location.append(np.unravel_index(np.argmin(self.system), self.system.shape))  # Store least fit location

        assert self.min_fitness[-1] == min_fitness, "Min fitness not stored correctly"
        assert self.avg_fitness[-1] == avg_fitness, "Avg fitness not stored correctly"
        assert self.std_fitness[-1] == std_fitness, "Std fitness not stored correctly"


if __name__=="__main__":
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define your save_folder relative to the script location
    save_folder = os.path.join(script_dir, 'Results', 'BakSneppen_results')

    size = 100

    iterations = 100

    model = BakSneppen2D(size, save_folder)
    model.simulate(iterations)


    # Plot minimum fitness evolution
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)

    plt.plot(range(iterations), model.min_fitness)
    plt.xlabel('Iteration Number')  
    plt.ylabel('Minimum Fitness')  
    plt.title('Minimum Fitness Evolution in the Bak-Sneppen Model')  

    #Each time an avalanche occurs (i.e., the least fit cell is replaced), the minimum fitness increases 
    #because the least fit cell has been replaced by a new one with higher fitness.
    #However, over time, some scells fitness will decrease due to the random "mutations" you're applying, 
    #causing the minimum fitness to decrease again. This results in the oscillating pattern

    # Plot average fitness evolution
    plt.subplot(2, 1, 2)
    plt.plot(range(iterations), model.avg_fitness)
    plt.xlabel('Iteration Number')
    plt.ylabel('Average Fitness')
    plt.title('Average Fitness Evolution in the Bak-Sneppen Model')

    plt.tight_layout()
    plt.show()

    #Over time, the replacement of the least fit cell and its neighbors
    #leads to an overall increase in fitness throughout the system, so
    #the average fitness should increase gradually.
    

    # Create a histogram of the final fitness values
    # Can see a threshold at approx 0.22, distribution is skewed to higher values?
    final_fitness_values = model.system.flatten()
    plt.hist(final_fitness_values, bins=30, edgecolor='black')
    plt.xlabel('Fitness Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Final Fitness Values')
    plt.show()

    # Plot standard deviation of fitness evolution
    # If the model leads to a homogeneous distribution, the standard deviation should decrease over time.
    plt.figure(figsize=(10, 8))
    plt.plot(range(iterations), model.std_fitness)
    plt.xlabel('Iteration Number')
    plt.ylabel('Standard Deviation of Fitness')
    plt.title('Fitness Variability Evolution in the Bak-Sneppen Model')
    plt.show()
