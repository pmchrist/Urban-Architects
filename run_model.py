from PercolationModel import PercolationModel2D
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
from config import N, temp, nsteps, nzeros
import powerlaw
import pandas as pd

# Create the percolation model, and seed four colony sites at the centre
cell= PercolationModel2D(N, temp)

# Set up interactive plotting

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)

istep = 0
while istep < nsteps:    

    ax.clear()
    print("Current Step: ", istep)
    # Draw the population automaton
    vis_param = cell.pop_dens
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')
    plt.draw()
    plt.savefig('./results/pop/step_pop'+str(istep).zfill(nzeros)+".png")      
    

    # Draw the energy automaton
    vis_param = cell.energy
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')
    plt.draw()
    plt.savefig('./results/energy/step_energy'+str(istep).zfill(nzeros)+".png")     


    # Draw the energy automaton
    vis_param = cell.type
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=2)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')
    plt.draw()
    plt.savefig('./results/water/type'+str(istep).zfill(nzeros)+".png")     

    # Making the fit for migrants
    data = cell.climate_migration_displaced
    if len(data) > 50:
        print("\nResults of Climate Migration Fit:")
        results = powerlaw.Fit(data)        
        print("alpha", results.power_law.alpha)
        print("xmin:", results.power_law.xmin)
        R, p = results.distribution_compare('power_law', 'lognormal')
        print("Powerlaw and Lognormal", R, p)

    # Making the fit for migrants
    data = cell.climate_migration_dead
    if len(data) > 50:
        print("\nResults of Climate Migration Dead Fit:")
        results = powerlaw.Fit(data)
        print("alpha", results.power_law.alpha)
        print("xmin:", results.power_law.xmin)
        R, p = results.distribution_compare('power_law', 'lognormal')
        print("Powerlaw and Lognormal", R, p)

    
    # Apply the Game of Life Rule, and update the grid
    cell.step()
    
    istep+=1

# plt.show()
print("Simulation complete!")

df = pd.DataFrame()
df['climate_migration_displaced'] = cell.climate_migration_displaced
df['climate_migration_dead'] = cell.climate_migration_dead
df['simple_migration'] = cell.simple_migration
df['pop_dens_mean'] = cell.l_pop_dens_mean
df['energy_mean'] = cell.l_energy_mean
df['fitness_mean'] = cell.l_fitness_mean
file_path = './results/result.csv'
df.to_csv(file_path)
print("Results saved!")