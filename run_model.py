from PercolationModel import PercolationModel2D
import PercolationModelPatterns as game
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
from config import N, nsteps, nzeros, icentre, jcentre,P

# Create the percolation model, and seed four colony sites at the centre
cell= PercolationModel2D(N)

# except 'block', there are also other format of the starting centre provided, like blinker etc..
game.add_block(cell,icentre,jcentre)
for i in range(0, 10):
    game.add_river(cell,icentre+4+2*i,jcentre+3+i)
# game.add_blinker(cell,icentre,jcentre)

# Set up interactive plotting

plt.ion()

fig1 = plt.figure()
ax = fig1.add_subplot(111)

istep= 0
while istep < nsteps:    

    ax.clear()
    print(istep)
    # Draw the automaton
    hist = ax.pcolor(cell.grid,edgecolors='black',vmin=-1,vmax=1)
    ax.text(0.9, 1.05,str(istep)+" Steps", horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=18)
    ax.text(0.1,1.05, "P="+str(P),horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=18)
    #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

    plt.draw()
    plt.savefig('.\\results\step_'+str(istep).zfill(nzeros)+".png")
    
    # Apply the Game of Life Rule, and update the grid
    cell.ApplyPercolationModelRule(P)
    cell.updateGrid()
    
    # Clear axes for next drawing
    sleep(0.01)
   
    if cell.check_complete():
        print("Run complete")
        break
    
    istep+=1

plt.ioff()

hist = ax.pcolor(cell.grid,edgecolors='black',vmin=-1,vmax=1)

plt.show()
