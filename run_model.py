from PercolationModel import PercolationModel2D
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
from config import N, nsteps, nzeros

# Create the percolation model, and seed four colony sites at the centre
cell= PercolationModel2D(N)

# Set up interactive plotting

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)

istep= 0
while istep < nsteps:    

    ax.clear()
    print("Current Step: ", istep)
    # Draw the automaton
    vis_param = cell.grid_param1
    hist = ax.pcolor(vis_param, edgecolors='black', vmin=-1, vmax=1)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('./results/step_'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)
    
    # Apply the Game of Life Rule, and update the grid
    cell.step()
    
    istep+=1

plt.ioff()

hist = ax.pcolor(cell.grid_param1,edgecolors='black',vmin=-1,vmax=1)

plt.show()
