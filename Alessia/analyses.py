
#                                                analyses


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


################################################################################################################
# PARAMETERS
################################################################################################################

bestFitness = []
median = []
q25 = []
q75 = []
x = []

notYetPlotted = True



################################################################################################################
# ANALYSES
################################################################################################################


def plotAverageFitness(tabFitnesses, cptSteps, nbSteps, funcObj = "maximisation", fileName = None):

    if funcObj == "maximisation":
        bestFitness.append(np.max(tabFitnesses))
    else :
        bestFitness.append(np.min(tabFitnesses))

    median.append(np.median(tabFitnesses))
    q25.append(np.quantile(tabFitnesses, 0.25))
    q75.append(np.quantile(tabFitnesses, 0.75))
    x.append(cptSteps)

 
    global notYetPlotted
    if cptSteps == nbSteps and notYetPlotted == True :
        print(cptSteps)
        plt.figure()

        plt.title("Swarm performance in foraging")
        plt.xlabel("Number of steps")
        plt.ylabel("Average reward")

        plt.plot(x, median, label = "Median")
        plt.plot(x, bestFitness, label = "Best fitness")
        plt.fill_between(x, q25, q75, alpha = 0.25, linewidth = 0, label = "Interquartile range")
        plt.legend()

        plt.show()
        
        notYetPlotted = False

        # Sauvegarde du tracé
        if fileName != None:
            plt.savefig(fileName, transparent = True)
    
#--------------------------------------------------------------------------------------------------------------
