
#                                                analyses


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import part2_learningModes
from allParts_tools import behaviorsEuclideanDistance
from math import sqrt


################################################################################################################
# PARAMETERS
################################################################################################################





################################################################################################################
# ANALYSES
################################################################################################################


def plotAverageFitness(strDetails, bestExpertFitness, bestFitness, median, q25, q75, periods, fileName=None):

    # for tabFitnesses in performances[i]:
    #     tabExpertsFitnesses = tabFitnesses[:strDetails['nbExpertsRobots']]
    #     tabFitnesses = tabFitnesses[strDetails['nbExpertsRobots']:]

    #     # maximising fitness in foraging task
    #     tmp_bestExpertFitness.append(np.max(tabExpertsFitnesses))
    #     tmp_bestFitness.append(np.max(tabFitnesses))
    #     tmp_median.append(np.median(tabFitnesses))
    #     tmp_q25.append(np.quantile(tabFitnesses, 0.25))
    #     tmp_q75.append(np.quantile(tabFitnesses, 0.75))



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

def getAverageMultipleFiles(filesList):
    
    averageTab = []


    with open(filesList[0], 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data1 = [float(i) for i in tmp]

    with open(filesList[1], 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data2 = [float(i) for i in tmp]

    with open(filesList[2], 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data3 = [float(i) for i in tmp]

    with open(filesList[3], 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data4 = [float(i) for i in tmp]

    with open(filesList[4], 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data5 = [float(i) for i in tmp]


    a = np.array([data1, data2, data3, data4, data5])

    averageTab = np.mean(a, axis=0)

    return averageTab


#--------------------------------------------------------------------------------------------------------------


def getAllData():

    file1 = "bestExpertFitness_100a.txt"
    file2 = "bestExpertFitness_100b.txt"
    file3 = "bestExpertFitness_100c.txt"
    file4 = "bestExpertFitness_100d.txt"
    file5 = "bestExpertFitness_100e.txt"
    bestExpertFitness = getAverageMultipleFiles([file1, file2, file3, file4, file5])

    file1 = "bestFitness_100a.txt"
    file2 = "bestFitness_100b.txt"
    file3 = "bestFitness_100c.txt"
    file4 = "bestFitness_100d.txt"
    file5 = "bestFitness_100e.txt"
    bestFitness = getAverageMultipleFiles([file1, file2, file3, file4, file5])

    file1 = "median_100a.txt"
    file2 = "median_100b.txt"
    file3 = "median_100c.txt"
    file4 = "median_100d.txt"
    file5 = "median_100e.txt"
    median = getAverageMultipleFiles([file1, file2, file3, file4, file5])

    file1 = "q25_100a.txt"
    file2 = "q25_100b.txt"
    file3 = "q25_100c.txt"
    file4 = "q25_100d.txt"
    file5 = "q25_100e.txt"
    q25 = getAverageMultipleFiles([file1, file2, file3, file4, file5])

    file1 = "q75_100a.txt"
    file2 = "q75_100b.txt"
    file3 = "q75_100c.txt"
    file4 = "q75_100d.txt"
    file5 = "q75_100e.txt"
    q75 = getAverageMultipleFiles([file1, file2, file3, file4, file5])

    return bestExpertFitness, bestFitness, median, q25, q75


#--------------------------------------------------------------------------------------------------------------


def getOneData(maxSizeDictMyBehaviors):

    file1 = f"bestExpertFitness_{maxSizeDictMyBehaviors}.txt"
    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        bestExpertFitness = [float(i) for i in tmp]

    file1 = f"bestFitness_{maxSizeDictMyBehaviors}.txt"
    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        bestFitness = [float(i) for i in tmp]

    file1 = f"median_{maxSizeDictMyBehaviors}.txt"
    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        median = [float(i) for i in tmp]

    file1 = f"q25_{maxSizeDictMyBehaviors}.txt"
    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        q25 = [float(i) for i in tmp]

    file1 = f"q75_{maxSizeDictMyBehaviors}.txt"
    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        q75 = [float(i) for i in tmp]

    return bestExpertFitness, bestFitness, median, q25, q75


#--------------------------------------------------------------------------------------------------------------

def writeAllData(strDetails, performances, maxSizeDictMyBehaviors):
    
    bestExpertFitness = []
    bestFitness = []
    median = []
    q25 = []
    q75 = []

    for tabFitnesses in performances:
        tabExpertsFitnesses = tabFitnesses[:strDetails['nbExpertsRobots']]
        tabFitnesses = tabFitnesses[strDetails['nbExpertsRobots']:]

        # maximising fitness in foraging task
        bestExpertFitness.append(np.max(tabExpertsFitnesses))
        bestFitness.append(np.max(tabFitnesses))
        median.append(np.median(tabFitnesses))
        q25.append(np.quantile(tabFitnesses, 0.25))
        q75.append(np.quantile(tabFitnesses, 0.75))
    
    s1 = ''
    for x in bestExpertFitness:
        s1 += f"{x}\n"

    with open(f"bestExpertFitness_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f1:
        f1.write(s1)

    #-----------------------------------------------------------
    
    s2 = ''
    for x in bestFitness:
        s2 += f"{x}\n"

    with open(f"bestFitness_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f2:
        f2.write(s2)

    #-----------------------------------------------------------

    s3 = ''
    for x in median:
        s3 += f"{x}\n"

    with open(f"median_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f3:
        f3.write(s3)

    #-----------------------------------------------------------

    s4 = ''
    for x in q25:
        s4 += f"{x}\n"

    with open(f"q25_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f4:
        f4.write(s4)
    
    #-----------------------------------------------------------

    s5 = ''
    for x in q75:
        s5 += f"{x}\n"

    with open(f"q75_{maxSizeDictMyBehaviors}.txt", 'w', encoding='utf-8') as f5:
        f5.write(s5)






#--------------------------------------------------------------------------------------------------------------

def plotAverageFitnessFromFiles(strDetails, file1, file2, file3, periods, suptitle, ylabel, fileName=None):

    with open(file1, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data1 = [float(i) for i in tmp]

    with open(file2, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data2 = [float(i) for i in tmp]

    with open(file3, 'r', encoding='utf-8') as f:
        tmp = f.readlines()
        data3 = [float(i) for i in tmp]


    plt.figure()
    plt.suptitle(suptitle, fontsize=14)

    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel(ylabel)

    if len(data1) != periods:
        periods = [i for i in range(len(data1))]
    plt.scatter(periods, data1, label = file1)

    if len(data2) != periods:
        periods = [i for i in range(len(data2))]
    plt.scatter(periods, data2, label = file2)

    if len(data3) != periods:
        periods = [i for i in range(len(data2))]
    plt.scatter(periods, data3[:len(data2)], label = file3)

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



#--------------------------------------------------------------------------------------------------------------

def writeExpertVsNotExpertDistances(self,
                    swarmLearningMode,
                    cptStepsG,
                    k,
                    expertSensorsPath,
                    expertActions):


    bestNotExpertActions = []

    for inputLayer in expertSensorsPath:

        if swarmLearningMode == "neuralNetworkBackpropagation":
            action = self.myNetwork.predict(inputLayer)

        elif swarmLearningMode == "kNearestNeighbors":
            action = self.myKnnClassifier.predict(inputLayer, self.dictMyBehaviors, k)

        else:
            print("No valid learning mode found : use one of neuralNetworkBackpropagation / kNearestNeighbors / evolutionnaryAlgorithm")
            exit()

        bestNotExpertActions.append(action)


    actionsDistances = []
    for aExp, aNotExp in zip(expertActions, bestNotExpertActions):
        d = behaviorsEuclideanDistance(aExp, aNotExp)
        actionsDistances.append(d)
    


    s = ''
    for d in actionsDistances:
        s += f"{d}\n"

    with open(f"distances_atStepNb{cptStepsG}.txt", 'w', encoding='utf-8') as f:
        f.write(s)


#--------------------------------------------------------------------------------------------------------------

    



