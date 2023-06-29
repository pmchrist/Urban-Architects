import numpy as np
import matplotlib.pyplot as plt
import os

class BakSneppen2D(object):
    def __init__(self, size, save_folder):
        # use a seed for repeatability
        np.random.seed(2)

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
        
        # initialize variables to track avalanches
        self.threshold_list = {'threshold': [], 'time_step': []}
        self.avalanche_time_list = {'avalanche_time': [], 'time_step': []}
        self.avalanche_timer = 1
        self.in_avalanche = False
        self.avalanche_sizes = []  
        self.avalanche_durations = []
        self.time_step = 0 


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


    
    def get_min(self):
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        min_fitness = self.system[min_indices]

        self.min_fitness.append(min_fitness)

        if not self.threshold_list['threshold']:
            self.threshold_list['threshold'].append(min_fitness)
            self.threshold_list['time_step'].append(self.time_step)
        elif min_fitness > max(self.threshold_list['threshold']):
            self.threshold_list['threshold'].append(min_fitness)
            self.threshold_list['time_step'].append(self.time_step)

    def get_avalanche_time(self):
        if not all(self.system.flat >= max(self.threshold_list['threshold'])):
            self.avalanche_timer += 1
        else:
            self.avalanche_time_list['avalanche_time'].append(self.avalanche_timer)
            self.avalanche_time_list['time_step'].append(self.time_step)
            # reset avalanche_timer
            self.avalanche_timer = 1

    def simulate(self, iterations):
        prev_min_fitness = np.min(self.system)
        self.time_step = 0  # Reset the time_step
        avalanche_size = 0

        for iteration in range(iterations):
            min_before = np.min(self.system)
            self.update_system()
            min_after = np.min(self.system)

            if min_after > prev_min_fitness:
                if not self.in_avalanche:
                    self.in_avalanche = True
                    self.avalanche_timer = 1
                    avalanche_size = 1
                else:
                    self.avalanche_timer += 1
                    avalanche_size += 1

            elif self.in_avalanche:
                self.get_avalanche_time()
                self.avalanche_durations.append(self.avalanche_timer)
                self.avalanche_sizes.append(avalanche_size)
                self.in_avalanche = False
                avalanche_size = 0 

            self.get_min() # remember to get min in each step
            self.store_system_properties()
            prev_min_fitness = min_after

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
        
        self.avg_fitness.append(np.mean(self.system))  # Compute average fitness
        self.std_fitness.append(np.std(self.system))  # Compute std dev of fitness
        self.least_fit_location.append(np.unravel_index(np.argmin(self.system), self.system.shape))  # Store least fit location

if __name__=="__main__":
    save_folder = 'BakSneppen_results'
    size = 100

    iterations = 1000

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
    # causing the minimum fitness to decrease again. This results in the oscillating pattern

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
    #plt.show()

    # Plot standard deviation of fitness evolution
    # If the model leads to a homogeneous distribution, the standard deviation should decrease over time.
    plt.figure(figsize=(10, 8))
    plt.plot(range(iterations), model.std_fitness)
    plt.xlabel('Iteration Number')
    plt.ylabel('Standard Deviation of Fitness')
    plt.title('Fitness Variability Evolution in the Bak-Sneppen Model')
    plt.show()

     # Create a histogram of the avalanche sizes
    # If exhibits self-organized criticality, should see smaller avalanches much more frequent than larger
    plt.figure()
    plt.hist(model.avalanche_sizes, bins=30, edgecolor='black')
    plt.xlabel('Avalanche Size')
    plt.ylabel('Frequency')
    plt.title('Histogram of Avalanche Sizes')
    plt.show()

    # Loglog
    sizes, counts = np.unique(model.avalanche_sizes, return_counts=True)
    plt.loglog(sizes, counts, 'o')
    plt.xlabel('Avalanche Size (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Histogram of Avalanche Sizes')
    plt.show()

    # Create a histogram of the avalanche durations
    # short-lived avalanches are more common than long-lasting ones
    plt.figure()
    plt.hist(model.avalanche_durations, bins=30, edgecolor='black')
    plt.xlabel('Avalanche Duration')
    plt.ylabel('Frequency')
    plt.title('Histogram of Avalanche Durations')
    plt.show()

    # Loglog
    durations, counts = np.unique(model.avalanche_durations, return_counts=True)
    plt.loglog(durations, counts, 'o')
    plt.xlabel('Avalanche Duration (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Histogram of Avalanche Durations')
    plt.show()

    # Create a scatter plot of avalanche size vs duration
    # Larger avalanches tend to last longer
    plt.figure()
    plt.scatter(model.avalanche_durations, model.avalanche_sizes)
    plt.xlabel('Avalanche Duration')
    plt.ylabel('Avalanche Size')
    plt.title('Avalanche Size vs Duration')
    plt.show()
    
    # expect to see a straight line on a log-log plot if correlation is perfect
    plt.loglog(model.avalanche_durations, model.avalanche_sizes, 'o')
    plt.xlabel('Avalanche Duration (log scale)')
    plt.ylabel('Avalanche Size (log scale)')
    plt.title('Avalanche Size vs Duration')
    plt.show()