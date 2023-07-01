import argparse
import matplotlib.pyplot as plt
from time import sleep
from numpy import log10
import numpy as np
from config import N, nsteps, nzeros
import powerlaw


def function_one():
    print("Starting Simple Spatial Simulation!")
    # Create the percolation model, and seed four colony sites at the centre
    from PercolationModelSimple import PercolationModel2D
    cell = PercolationModel2D(N)

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


def function_two():
    print("Starting Complicated Spatial Simulation!")
    # Create the percolation model, and seed four colony sites at the centre
    from PercolationModelComplicated import PercolationModel2D
    cell= PercolationModel2D(N)

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
        hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=2.0)
        #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

        plt.draw()
        plt.savefig('./results/results_complicated/pop/pop'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)
        

        # Draw the energy automaton
        vis_param = cell.energy
        hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=1)
        #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

        plt.draw()
        plt.savefig('./results/results_complicated/energy/energy'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)


        # Draw the energy automaton
        vis_param = cell.fitness
        hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=1)
        #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

        plt.draw()
        plt.savefig('./results/results_complicated/fitness/fitness'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)


        # Draw the energy automaton
        vis_param = cell.type
        hist = ax.pcolor(vis_param, edgecolors='black', vmin=0, vmax=2)
        #hist = ax.pcolor(cell.nextgrid,edgecolors='black', cmap='binary')

        plt.draw()
        plt.savefig('./results/results_complicated/water/type'+str(istep).zfill(nzeros)+".png")      # change address i have linux sorry :)

        # Making the fit for migrants
        data = cell.simple_migration
        if ((istep+2) % 100) == 0:
            print("\nResults of Fit for Land Migrants:")
            results = powerlaw.Fit(data)
            print("alpha", results.power_law.alpha)
            print("xmin:", results.power_law.xmin)
            R, p = results.distribution_compare('power_law', 'lognormal')
            print("Powerlaw and Lognormal", R, p)
            x = np.linspace(0, istep, len(data))
            y = sorted(data, reverse=True)
            ax.clear()
            plt.loglog(x, y, 'o', color='black')
            plt.savefig('./results/results_complicated/land_migrants'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)

        # Making the fit for migrants
        data = cell.climate_migration_displaced
        if ((istep+2) % 100) == 0:
            print("\nResults of Fit for Climate Migrants:")
            results = powerlaw.Fit(data)
            print("alpha", results.power_law.alpha)
            print("xmin:", results.power_law.xmin)
            R, p = results.distribution_compare('power_law', 'lognormal')
            print("Powerlaw and Lognormal", R, p)
            x = np.linspace(0, istep, len(data))
            y = sorted(data, reverse=True)
            ax.clear()
            plt.loglog(x, y, 'o', color='black')
            plt.savefig('./results/results_complicated/climate_survivor_migrants'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)

        # Making the fit for migrants
        data = cell.climate_migration_dead
        if ((istep+2) % 100) == 0:
            print("\nResults of Fit for Climate Diseased:")
            results = powerlaw.Fit(data)
            print("alpha", results.power_law.alpha)
            print("xmin:", results.power_law.xmin)
            R, p = results.distribution_compare('power_law', 'lognormal')
            print("Powerlaw and Lognormal", R, p)
            x = np.linspace(0, istep, len(data))
            y = sorted(data, reverse=True)
            ax.clear()
            plt.loglog(x, y, 'o', color='black')
            plt.savefig('./results/results_complicated/climate_dead_migrants'+str(istep).zfill(nzeros)+".png", dpi=200)      # change address i have linux sorry :)
            

        
        # Apply the Game of Life Rule, and update the grid
        cell.step()
        
        istep+=1


def main():
    parser = argparse.ArgumentParser(description="Example program")
    parser.add_argument('--function', choices=['simple', 'complicated'], help='Specify function to execute')

    args = parser.parse_args()

    if args.function == 'simple':
        function_one()
    elif args.function == 'complicated':
        function_two()
    else:
        print("Invalid function choice!")

if __name__ == '__main__':
    main()
    print("Simulation complete!")
