import numpy as np
import matplotlib.pyplot as plt
import os
import powerlaw

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

        # Ages: number of iterations it has gone through without mutating
        self.ages = np.zeros((size, size), dtype=int)


    def update_system(self):
        # get the indices of the lowest fitness value
        min_indices = np.unravel_index(np.argmin(self.system), self.system.shape)
        i, j = min_indices

        # Reset the age of the least fit cell and its neighbours
        self.ages[i, j] = 0
        self.ages[(i - 1) % self.size, j] = 0
        self.ages[(i + 1) % self.size, j] = 0
        self.ages[i, (j - 1) % self.size] = 0
        self.ages[i, (j + 1) % self.size] = 0

        # give a new fitness value to the cell with the lowest value
        self.system[i, j] = np.random.rand()

        # also update the neighbors, where we use periodic boundary conditions
        self.system[(i - 1) % self.size, j] =  np.random.rand()
        self.system[(i + 1) % self.size, j] =  np.random.rand()
        self.system[i, (j - 1) % self.size] =  np.random.rand()
        self.system[i, (j + 1) % self.size] =  np.random.rand()

        # If we're in an avalanche, increment the avalanche size
        if self.in_avalanche:
            self.avalanche_size += 5  # We update 5 cells in total, the least fit and its 4 neighbors



    
    def get_min(self):
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
        # Get the initial minimum fitness of the system
        prev_min_fitness = np.min(self.system)
        # Reset the time_step and set initial avalanche size
        self.time_step = 0  
        self.avalanche_size = 0

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
                    self.avalanche_size = 5  # Start counting the avalanche size (we already updated 5 cells)
                # ...and we're already in an avalanche, continue it
                else:
                    self.avalanche_timer += 1

            # If the new minimum fitness is not higher and we're in an avalanche, check if it's finished
            elif self.in_avalanche:
                # If all cells in the system have a fitness greater than or equal to the current threshold,
                # then the avalanche has ended
                if all(self.system.flat >= prev_min_fitness):
                    self.get_avalanche_time()
                    self.avalanche_durations.append(self.avalanche_timer)
                    self.avalanche_sizes



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
    save_folder = 'BakSneppen_results2'
    size = 100

    iterations = 500

    model = BakSneppen2D(size, save_folder)
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

    # Age distribution
    # Kolmogorov-Smirnov to check power
    ages = model.ages.flatten()
    fit = powerlaw.Fit(np.array(ages), discrete=True, verbose=False)
    # comparing the power-law distribution to an exponential distribution
    R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    print("Test statistic: ", R)
    print("p-value: ", p)
    print("Power-law exponent: ", fit.power_law.alpha)

    # comparing the power-law distribution to a log-normal distribution
    R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    print("Test statistic: ", R)
    print("p-value: ", p)
    print("Power-law exponent: ", fit.power_law.alpha)


    #Plot age distribution
    plt.hist(ages, bins=30, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Age Distribution')
    plt.show()

    # Loglog
    ages, counts = np.unique(ages, return_counts=True)
    plt.loglog(ages, counts, 'o')
    plt.xlabel('Age (log scale)')
    plt.ylabel('Frequency (log scale)')
    plt.title('Age Distribution')
    plt.show()


