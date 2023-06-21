from PercolationModel import PercolationModel2D
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
from config import N, temp, nsteps, nzeros

# Create the percolation model, and seed four colony sites at the centre
cell= PercolationModel2D(N, temp)

# Set up interactive plotting

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)

istep= 0
while istep < nsteps:    

    ax.clear()
    print("Current Step: ", istep)
    # Draw the population automaton
    vis_param = cell.pop
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=-1, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/pop/step_pop'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)
    
    # Draw the water automaton
    vis_param = cell.water
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=-1, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/water/step_water'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)

    # Draw the energy automaton
    vis_param = cell.energy
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=-1, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/energy/step_energy'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)

    
    # Apply the Game of Life Rule, and update the grid
    cell.step()
    
    istep+=1

# plt.ioff()

# hist = ax.pcolor(cell.pop,edgecolors='black',vmin=-1,vmax=1)

# plt.show()
print("Simulation complete!")
