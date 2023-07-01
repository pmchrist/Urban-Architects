from PercolationModelSimple import PercolationModel2D
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
import numpy as np
from config import N, temp, nsteps, nzeros
import powerlaw

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
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=1.2)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/results_simple/pop/pop'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)
    
    # Draw the energy automaton
    vis_param = cell.type
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=2)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/results_simple/water/type'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)

    # Making the fit for migrants
    data = cell.migrants
    if ((istep+2) % 100) == 0:
        print("\nResults of Climate Migration Fit:")
        results = powerlaw.Fit(data)
        print("alpha", results.power_law.alpha)
        print("xmin:", results.power_law.xmin)
        R, p = results.distribution_compare('power_law', 'lognormal')
        print("Powerlaw and Lognormal", R, p)
        x = np.linspace(0, istep, len(data))
        y = sorted(data, reverse=True)
        ax.clear()
        plt.loglog(x, y, 'o', color='black')
        plt.savefig('./results/results_simple/survivor_migrants'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)

    # Making the fit for migrants
    data = cell.dead_migrants
    if ((istep+2) % 100) == 0:
        print("\nResults of Climate Diseased Fit:")
        results = powerlaw.Fit(data)
        print("alpha", results.power_law.alpha)
        print("xmin:", results.power_law.xmin)
        R, p = results.distribution_compare('power_law', 'lognormal')
        print("Powerlaw and Lognormal", R, p)
        x = np.linspace(0, istep, len(data))
        y = sorted(data, reverse=True)
        ax.clear()
        plt.loglog(x, y, 'o', color='black')
        plt.savefig('./results/results_simple/dead_migrants'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)
        

    # Apply the Game of Life Rule, and update the grid
    cell.step()
    
    istep+=1

 
# plt.ioff()

# hist = ax.pcolor(cell.pop,edgecolors='black',vmin=-1,vmax=1)

# plt.show()
print("Simulation complete!")
