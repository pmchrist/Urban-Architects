import numpy as np
import matplotlib.pyplot as plt
import os
import powerlaw

class BakSneppen2D_A(object):
    """
    A class representing a 2D Bak-Sneppen model, including avalanche calculations
    """
    def __init__(self, size, save_folder):
        """
        Initializes the BakSneppen2D object with the given size and save_folder.

        Parameters:
        size (int): The size of the 2D system.
        save_folder (str): The directory to save output images.
        """
       
        np.random.seed(2)
        assert isinstance(size, int), "Size should be an integer"
        # initialize the system with random population density values
        self.size = size
        self.system = np.random.rand(size, size)
        assert isinstance(size, int), "Size should be an integer"

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

        # Ages: number of iterations it has gone through without mutating
        self.ages = np.zeros((size, size), dtype=int)

        assert self.system.shape == (size, size), "System not initialized correctly"
        assert np.all(0 <= self.system) and np.all(self.system <= 1), "System should be initialized with random values between 0 and 1"
        assert np.all(self.ages == 0), "Ages should be initialized with zeros"


    def update_system(self):
        """
        Updates the system by identifying the cell with the lowest fitness value and its neighbors
        and replacing them with new random values. Also increments the ages of all cells.
        """
        self.ages += 1
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        i, j = min_indices
        self.ages[i, j] = 0
        self.system[i, j] = np.random.rand()

        # update neighbors without periodic boundary conditions
        neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
        for ni, nj in neighbors:
            if 0 <= ni < self.size and 0 <= nj < self.size:
                self.ages[ni, nj] = 0
                self.system[ni, nj] = np.random.rand()
        

    
    def get_min(self):
        """
        Calculates the minimum fitness in the system and updates the threshold list accordingly.
        """
        # Get the indices of the cell with the minimum fitness in the system
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        # Get the minimum fitness value
        min_fitness = self.system[min_indices]

        # Append the minimum fitness to the min_fitness list
        self.min_fitness.append(min_fitness)

        # If there are no thresholds set yet, or if the current min_fitness is
        # higher than all previous thresholds, append it to the threshold list
        if not self.threshold_list['threshold']:
            self.threshold_list['threshold'].append(min_fitness)
            self.threshold_list['time_step'].append(self.time_step)
        elif min_fitness > max(self.threshold_list['threshold']):
            self.threshold_list['threshold'].append(min_fitness)
            self.threshold_list['time_step'].append(self.time_step)


    def get_avalanche_time(self):
        """
        Calculates the duration of an avalanche, an event where the minimum fitness increases.
        """
        # If any cell in the system has a fitness less than the current threshold, increment the avalanche timer
        if not all(self.system.flat >= max(self.threshold_list['threshold'])):
            self.avalanche_timer += 1
        # If all cells in the system have a fitness greater than or equal to the current threshold,
        # then the avalanche has ended. Record the duration of the avalanche and the time_step it ended at
        else:
            self.avalanche_time_list['avalanche_time'].append(self.avalanche_timer)
            self.avalanche_time_list['time_step'].append(self.time_step)
            # reset avalanche_timer
            self.avalanche_timer = 1

    def simulate(self, iterations):
        """
        Simulates the system for a given number of iterations, updating the system, 
        storing its properties, and calculating avalanches.

        Parameters:
        iterations (int): The number of iterations to run the simulation.
        """
        # Get the initial minimum fitness of the system
        prev_min_fitness = np.min(self.system)
        # Reset the time_step and set initial avalanche size
        self.time_step = 0  
        avalanche_size = 0

        for iteration in range(iterations):
            # Record the minimum fitness before updating the system
            min_before = np.min(self.system)
            # Update the system
            self.update_system()
            # Record the minimum fitness after updating the system
            min_after = np.min(self.system)

            # If the new minimum fitness is higher than the previous one...
            if min_after > prev_min_fitness:
                # ...and we're not currently in an avalanche, start a new one
                if not self.in_avalanche:
                    self.in_avalanche = True
                    self.avalanche_timer = 1
                    avalanche_size = 5
                 # ...and we're already in an avalanche, continue it
                else:
                    self.avalanche_timer += 1
                    avalanche_size += 1

            # If the new minimum fitness is not higher and we're in an avalanche, the avalanche has ended
            elif self.in_avalanche:
                self.get_avalanche_time()
                self.avalanche_durations.append(self.avalanche_timer)
                self.avalanche_sizes.append(avalanche_size)
                self.in_avalanche = False
                avalanche_size = 0 

            # Get the minimum fitness and store the system properties at this time step
            self.get_min() 
            self.store_system_properties()
            # Set the previous minimum fitness for the next iteration
            prev_min_fitness = min_after

            if iteration % 1000 == 0:
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
        Stores the system properties such as avg_fitness, std_fitness, least_fit_location.
        """
        
        self.avg_fitness.append(np.mean(self.system))  # Compute average fitness
        self.std_fitness.append(np.std(self.system))  # Compute std dev of fitness
        self.least_fit_location.append(np.unravel_index(np.argmin(self.system), self.system.shape))  # Store least fit location


    

if __name__=="__main__":
    # Get the absolute path of the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define your save_folder relative to the script location
    save_folder = os.path.join(script_dir, 'Results', 'BakSneppen_results2')

    size = 100

    iterations = 50000

    model = BakSneppen2D_A(size, save_folder)
    model.simulate(iterations)

    # Plot minimum fitness evolution
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)

    plt.plot(range(iterations), model.min_fitness)
    plt.xlabel('Iteration Number')  
    plt.ylabel('Minimum Fitness')  
    plt.title('Minimum Fitness Evolution in the Bak-Sneppen Model')  

    #Each time the least fit cell gets replaced, the minimum fitness increases 
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

    # Ages
    # higher age therefore means that a cell has remained stable for a longer period of time
    plt.imshow(model.ages, cmap='viridis')
    plt.colorbar(label='Age')
    plt.title('Age distribution after 50000 iterations')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.show()

    # Kolmogorov-Smirnov to check power
    avalanche_sizes = model.avalanche_sizes
    avalanche_durations = model.avalanche_durations
    fit1 = powerlaw.Fit(np.array(avalanche_sizes), discrete=True, verbose=False)
    fit2 = powerlaw.Fit(np.array(avalanche_durations), discrete=True, verbose=False)
    # comparing the power-law distribution to an exponential distribution
    R1, p1 = fit1.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R2, p2 = fit2.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print("Test statistic, size : ", R1)
    print("p-value, size: ", p1)
    print("Power-law exponent, size: ", fit1.power_law.alpha)
    print("Test statistic, duration: ", R2)
    print("p-value, duration: ", p2)
    print("Power-law exponent duration: ", fit2.power_law.alpha)

    # comparing the power-law distribution to a log-normal distribution
    R3, p3 = fit1.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    R4, p5 = fit2.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    print("Test statistic, size: ", R3)
    print("p-value, size: ", p3)
    print("Power-law exponent, size: ", fit1.power_law.alpha)
    print("Test statistic, duration: ", R4)
    print("p-value, duration: ", p5)
    print("Power-law exponent, duration: ", fit2.power_law.alpha)


    


