
#                                                analyses


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import matplotlib.pyplot as plt


################################################################################################################
# PARAMETERS
################################################################################################################





################################################################################################################
# ANALYSES
################################################################################################################


def plotAverageFitness(strDetails, performances, periods, funcObj="maximisation", fileName=None):

    bestExpertFitness = []
    bestFitness = []
    median = []
    q25 = []
    q75 = []

    for tabFitnesses in performances:
        tabExpertsFitnesses = tabFitnesses[:strDetails['nbExpertsRobots']]
        tabFitnesses = tabFitnesses[strDetails['nbExpertsRobots']:]

        if funcObj == "maximisation":
            bestExpertFitness.append(np.max(tabExpertsFitnesses))
            bestFitness.append(np.max(tabFitnesses))
        else :
            bestExpertFitness.append(np.min(tabExpertsFitnesses))
            bestFitness.append(np.min(tabFitnesses))

        median.append(np.median(tabFitnesses))
        q25.append(np.quantile(tabFitnesses, 0.25))
        q75.append(np.quantile(tabFitnesses, 0.75))



    plt.figure()
    plt.suptitle("Swarm performance in foraging", fontsize=14)
    
    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Average reward")
    plt.plot(periods, bestExpertFitness, label = "Expert best fitness")
    plt.plot(periods, bestFitness, label = "Swarm best fitness")
    plt.plot(periods, median, label = "Swarm median")
    plt.fill_between(periods, q25, q75, alpha=0.25, linewidth=0, label="Swarm interquartile range")
    plt.legend()
    
    #-----------------------------------------------------------
    
    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "maxSizeDictMyBehaviors", "hit_ee learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]

    s = "EVALUATION DETAILS :\n\n"
    for key in strDetails:
        s += f"{key} : {strDetails[key]}       "
        if key in tab:
            s += "\n"
            
    xmin, _, ymin,_ = plt.axis()
    plt.text(xmin, ymin, s, fontsize=12)
    plt.axis('off')

    #-----------------------------------------------------------
    
    plt.subplot_tool()

    # Show plot on terminal
    plt.show()

    # Save plot on file
    if fileName != None:
        plt.savefig(fileName, transparent=True)
    

#--------------------------------------------------------------------------------------------------------------

def writeSizeDictMyBehaviors(strDetails, performances, maxSizeDictMyBehaviors):
    median = []
    with open(f"sizeDictMyBehaviors_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f:

        for tabFitnesses in performances:
            tabFitnesses = tabFitnesses[strDetails['nbExpertsRobots']:]

            median.append(np.median(tabFitnesses))
        
        s = ''
        for m in median:
            s += f"{m}\n"

        f.write(s)


#--------------------------------------------------------------------------------------------------------------

def plotAverageFitness_sizeDB(strDetails, file1, file2, file3, periods, fileName=None):

    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        medianFitnesses1 = [float(i) for i in tmp]

    with open(file2, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        medianFitnesses2 = [float(i) for i in tmp]

    with open(file3, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        medianFitnesses3 = [float(i) for i in tmp]


    plt.figure()
    plt.suptitle("Swarm performance in foraging in function of size behaviors", fontsize=14)

    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Average reward")

    plt.plot(periods, medianFitnesses1, label = file1)
    plt.plot(periods, medianFitnesses2, label = file2)
    plt.plot(periods, medianFitnesses3, label = file3)

    plt.legend()
    
    #-----------------------------------------------------------
    
    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "hit_ee learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]
    tab2 = ["maxSizeDictMyBehaviors"]

    s = "EVALUATION DETAILS :\n\n"
    for key in strDetails:
        if key not in tab2:
            s += f"{key} : {strDetails[key]}       "
            if key in tab:
                s += "\n"
            
    xmin, _, ymin,_ = plt.axis()
    plt.text(xmin, ymin, s, fontsize=12)
    plt.axis('off')

    #-----------------------------------------------------------
    
    plt.subplot_tool()

    # Show plot on terminal
    plt.show()

    # Save plot on file
    if fileName != None:
        plt.savefig(fileName, transparent=True)




