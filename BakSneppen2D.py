import numpy as np
import random
import matplotlib.pyplot as plt
import os

class BakSneppen2D(object):

    def __init__(self, size, save_folder):

        # use a seed for repeatability
        np.random.seed(2)

        # initialize the system with random fitness values
        self.size = size
        self.system = np.random.rand(size, size)

        # set global parameters
        self.save_folder = save_folder

        # initialize lists for storing interesting parameter values
        self.min_fitness = []

    def update_system(self):

        # get the indices of the lowest fitness value
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        i, j = min_indices

        # give a new fitness value to the cell with the lowest value
        self.system[i, j] = np.random.rand()

        # also update the neighbors, where we use periodic boundary conditions
        self.system[(i - 1) % self.size, j] =  np.random.rand()
        self.system[(i + 1) % self.size, j] =  np.random.rand()
        self.system[i, (j - 1) % self.size] =  np.random.rand()
        self.system[i, (j + 1) % self.size] =  np.random.rand()
    
    def simulate(self, iterations):
        for iteration in range(iterations):
            self.update_system()
            self.store_system_properties()

            if iteration % 10 == 0:
                self.plot_system(iteration)

    def plot_system(self, iteration):
        plt.imshow(self.system, cmap='hot', origin='lower')
        plt.colorbar(label='Fitness')
        plt.title(f'Bak-Sneppen model in 2D (iteration {iteration + 1})')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.savefig(os.path.join(self.save_folder, f'iteration_{iteration + 1}.png'))
        plt.close()

    def store_system_properties(self):
        self.min_fitness.append(np.min(self.system))


if __name__=="__main__":
    save_folder = 'BakSneppen_results'
    size = 100

    iterations = 1000

    model = BakSneppen2D(size, save_folder)
    model.simulate(iterations)

    plt.plot(range(iterations), model.min_fitness)
    plt.show()