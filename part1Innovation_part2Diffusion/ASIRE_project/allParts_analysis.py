
#                                                analysis


################################################################################################################
# IMPORTS
################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from allParts_tools import behaviorsEuclideanDistance





################################################################################################################
# ANALYSIS METHODS
################################################################################################################

def plot_data_performance(folder):

    parse = folder.split('/')
    parse = parse[-1].split('_')
    assert parse[1] == "performance" or parse[1] == "learnFromAll", "plot_data_performance works only for performance or learnFromAll data"
    assert folder != None, "Please enter a valid folder containing data"

    strDetails = getStrDetails(folder)
    labels = ["bestExpertFitness", "bestFitness", "median", "q25", "q75"]
    data = {}

    for fileName in labels:
        file = f"{folder}/{fileName}_{strDetails['nbSteps']}_{strDetails['maxSizeDictMyBehaviors']}_{strDetails['swarmLearningMode']}.txt"
        with open(file, 'r', encoding='utf-8') as f:
            tmp = f.readlines()
            data[fileName] = [float(i) for i in tmp]


    periods = [i for i in range(0, int(strDetails['nbSteps'])+1, int(strDetails['evaluationTime']))]

    #-----------------------------------------------------------

    plt.figure()
    plt.suptitle("Swarm performance in foraging", fontsize=14)
    
    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Average reward")
    plt.plot(periods, data["bestExpertFitness"], label = "Expert best fitness")
    plt.plot(periods, data["bestFitness"], label = "Swarm best fitness")
    plt.plot(periods, data["median"], label = "Swarm median")
    plt.fill_between(periods, data["q25"], data["q75"], alpha=0.25, linewidth=0, label="Swarm interquartile range")
    plt.legend()
    
    #-----------------------------------------------------------
    
    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "maxSizeDictMyBehaviors", "hit_ee_learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]

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
    plt.show()

    

#--------------------------------------------------------------------------------------------------------------

def plot_data_5performances(folder):

    parse = folder.split('/')
    parse = parse[-1].split('_')
    assert parse[1] == "5performances", "plot_data_5performances works only with 5performances folders"
    assert folder != None, "Please enter a valid folder containing data"

    strDetails = getStrDetails(folder)
    labels = ["bestExpertFitness", "bestFitness", "median", "q25", "q75"]
    series = ["a", "b", "c", "d", "e"]
    data = {}

    for fileName in labels:
        tab = np.array([])
        for serie in series:
            file = f"{folder}/{fileName}_{strDetails['maxSizeDictMyBehaviors']}{serie}.txt"
            with open(file, 'r', encoding='utf-8') as f:
                tmp = f.readlines()
                tab.append([float(i) for i in tmp])

        data[fileName] = np.mean(tab, axis=0)

    periods = [i for i in range(0, int(strDetails['nbSteps'])+1, int(strDetails['evaluationTime']))]


    #-----------------------------------------------------------

    plt.figure()
    plt.suptitle("Swarm performance in foraging", fontsize=14)
    
    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Average reward")
    plt.plot(periods, data["bestExpertFitness"], label = "Expert best fitness")
    plt.plot(periods, data["bestFitness"], label = "Swarm best fitness")
    plt.plot(periods, data["median"], label = "Swarm median")
    plt.fill_between(periods, data["q25"], data["q75"], alpha=0.25, linewidth=0, label="Swarm interquartile range")
    plt.legend()
    
    #-----------------------------------------------------------
    
    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "maxSizeDictMyBehaviors", "hit_ee_learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]

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
    plt.show()

    

#--------------------------------------------------------------------------------------------------------------

def plot_data_distExpVsNotExp(folder, time1, time2, time3):
    """
    Analysis to compare expert et best not experts action choices, in terms of distances

    """

    parse = folder.split('/')
    parse = parse[-1].split('_')
    assert parse[1] == "distExpVsNotExp", "plot_data_distExpVsNotExp works only for distExpVsNotExp data"
    assert folder != None, "Please enter a valid folder"

    strDetails = getStrDetails(folder)

    plt.figure()
    plt.suptitle("Euclidean distance between expert / notExpert action choices in expert path", fontsize=14)
    
    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Actions euclidean distance")


    fileName = f"distances_atStepNb{time1}_{strDetails['swarmLearningMode']}"
    file = f"{folder}/{fileName}.txt"
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    periods = [i for i in range(1, time1)]
    plt.scatter(periods, data, label = f"distances_atStepNb{time1}")


    fileName = f"distances_atStepNb{time2}_{strDetails['swarmLearningMode']}"
    file = f"{folder}/{fileName}.txt"
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    periods = [i for i in range(1, time2)]
    plt.scatter(periods, data, label = f"distances_atStepNb{time2}")


    fileName = f"distances_atStepNb{time3}_{strDetails['swarmLearningMode']}"
    file = f"{folder}/{fileName}.txt"
    with open(file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    data = data[:time2-1]
    plt.scatter(periods, data, label = f"distances_atStepNb{time3}")


    plt.legend()
    
    #-----------------------------------------------------------

    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "maxSizeDictMyBehaviors", "hit_ee_learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]

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
    plt.show()




#--------------------------------------------------------------------------------------------------------------

def plot_data_sizeDB(folder, size1, size2):

    parse = folder.split('/')
    parse = parse[-1].split('_')
    assert parse[1] == "sizeDB", "plot_data_sizeDB works only for sizeDB (medians) data"
    assert folder != None, "Please enter a valid folder"

    strDetails = getStrDetails(folder)

    periods = [i for i in range(0, int(strDetails['nbSteps'])+1, int(strDetails['evaluationTime']))]


    plt.figure()
    plt.suptitle("Swarm performance in foraging in function of size behaviors", fontsize=14)

    #-----------------------------------------------------------

    plt.subplot(211)
    plt.xlabel("Number of steps")
    plt.ylabel("Average reward")


    for size in [size1, size2]:
        fileName = f"median_{strDetails['nbSteps']}_{size}"
        file = f"{folder}/{fileName}.txt"
        with open(file, 'r', encoding='utf-8') as f:
            data = f.readlines()
            plt.plot(periods, data, label = fileName)

    plt.legend()
    
    #-----------------------------------------------------------

    plt.subplot(212)
    tab = ["nbNotExpertsRobots", "maxSizeDictMyBehaviors", "hit_ee_learningOnlyFromExperts", "swarmLearningMode", "nb_neuronsPerOutputs", "nbEpoch", "k"]

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
    plt.show() 




#--------------------------------------------------------------------------------------------------------------

def plot_data_timing(labelList, timeList):

    plt.figure()
    plt.suptitle("Execution time", fontsize=14)

    plt.xlabel("Methods")
    plt.ylabel("Time (min)")

    plt.bar(labelList, timeList)
    
    plt.show() 



#--------------------------------------------------------------------------------------------------------------

def writeStrDetails(nbRobots,
                nbExpertsRobots,
                nbNotExpertsRobots,
                nbFoodObjects,
                maxSizeDictMyBehaviors,
                transferRate,
                maturationDelay,
                learningOnlyFromExperts,
                swarmLearningMode,
                nb_hiddenLayers,
                nb_neuronsPerHidden,
                nb_neuronsPerOutputs,
                defaultBehavior,
                learningRate,
                distanceEpsilon,
                nbEpoch,
                k,
                nbSteps,
                evaluationTime,
                resetEvaluation,
                resetEvaluationTime):

    s = ''
    s += f"nbRobots {nbRobots}\n"
    s += f"nbExpertsRobots {nbExpertsRobots}\n"
    s += f"nbNotExpertsRobots {nbNotExpertsRobots}\n"

    s += f"nbFoodObjects {nbFoodObjects}\n"
    s += f"maxSizeDictMyBehaviors {maxSizeDictMyBehaviors}\n"

    s += f"hit_ee_transferRate {transferRate}\n"
    s += f"hit_ee_maturationDelay {maturationDelay}\n"
    s += f"hit_ee_learningOnlyFromExperts {learningOnlyFromExperts}\n"

    s += f"swarmLearningMode {swarmLearningMode}\n"

    if swarmLearningMode == "neuralNetworkBackpropagation":
        s += f"nb_hiddenLayers {nb_hiddenLayers}\n"
        s += f"nb_neuronsPerHidden {nb_neuronsPerHidden}\n"
        s += f"nb_neuronsPerOutputs {nb_neuronsPerOutputs}\n"

        s += f"defaultBehavior {defaultBehavior}\n"
        s += f"learningRate {learningRate}\n"
        s += f"distanceEpsilon {distanceEpsilon}\n"
        s += f"nbEpoch {nbEpoch}\n"

    if swarmLearningMode == "kNearestNeighbors":
        s += f"k {k}\n"

    s += f"nbSteps {nbSteps}\n"
    s += f"evaluationTime {evaluationTime}\n"
    s += f"resetEvaluation {resetEvaluation}\n"
    if resetEvaluation:
        s += f"resetEvaluationTime {resetEvaluationTime}\n"


    with open(f'lastAnalysis/strDetails.txt', 'w', encoding='utf-8') as f:
        f.write(s)



#---------------------------------------------------------------------------------------------------------------

def writePerformanceData(performances):

    strDetails = getStrDetails()
    labels = ["bestExpertFitness", "bestFitness", "median", "q25", "q75"]
    data = {}

    for fileName in labels:
        data[fileName] = []


    for tabFitnesses in performances:
        tabExpertsFitnesses = tabFitnesses[:int(strDetails['nbExpertsRobots'])]
        tabFitnesses = tabFitnesses[int(strDetails['nbExpertsRobots']):]

        # maximising fitness in foraging task
        data['bestExpertFitness'].append(np.max(tabExpertsFitnesses))
        data['bestFitness'].append(np.max(tabFitnesses))
        data['median'].append(np.median(tabFitnesses))
        data['q25'].append(np.quantile(tabFitnesses, 0.25))
        data['q75'].append(np.quantile(tabFitnesses, 0.75))
    

    for fileName in labels:
        s = ''
        for x in data['bestExpertFitness']:
            s += f"{x}\n"

        with open(f"lastAnalysis/{fileName}_{strDetails['nbSteps']}_{strDetails['maxSizeDictMyBehaviors']}_{strDetails['swarmLearningMode']}.txt", 'w', encoding='utf-8') as f:
            f.write(s)


#--------------------------------------------------------------------------------------------------------------

def writeExpertVsNotExpertDistances(self,
                    swarmLearningMode,
                    cptStepsG,
                    k,
                    expertSensorsPath,
                    expertActions):

    bestNotExpertActions = []

    for inputLayer in expertSensorsPath:

        if swarmLearningMode == "neuralNetworkBackpropagation" or swarmLearningMode == "innovation":
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


    with open(f"lastAnalysis/distances_atStepNb{cptStepsG}_{swarmLearningMode}.txt", 'w', encoding='utf-8') as f:
        f.write(s)


#--------------------------------------------------------------------------------------------------------------

def getStrDetails(folder=''):

    strDetails = {}

    file = "lastAnalysis/strDetails.txt"
    if folder != '':
        file = f"{folder}/strDetails.txt"

    with open(file, 'r', encoding='utf-8') as f:
        tmp = f.readlines()

    strDetails = {}
    for line in tmp:
        s = line.split()
        strDetails[s[0]] = s[1]

    return strDetails






####################################################################
# PARAMETERS - PLOT FROM FOLDER
####################################################################

folder = "allParts_results/data_6000steps/data_performance_6000_100_bp"
#plot_data_performance(folder)

#---------------------------------------------

#folder = "allParts_results/data_learnFromAll_20000_100_knn"
#plot_data_performance(folder)

#---------------------------------------------

time1 = 1000
time2 = 2500
time3 = 6000   # set this value
folder = "allParts_results/data_6000steps/data_distExpVsNotExp_6000_100_bp"
#plot_data_distExpVsNotExp(folder, time1, time2, time3)

#---------------------------------------------

size1 = 50
size2 = 100
folder = "allParts_results/data_10000steps/data_sizeDB_10000_knn"
#plot_data_sizeDB(folder, size1, size2)

#---------------------------------------------

#folder = "allParts_results/data_5performances_6000_100_knn"
#plot_data_5performances(folder)

#---------------------------------------------

labelList = ["innovation", "backpr_100", "backpr_50", "knn_100", "knn_50"]
innovation = 14       # 14 min
backpr_100 = 1481     # 24h 41min
backpr_50 = 203       # 3h 23min
knn_100 = 10          # 10 min
knn_50 = 96           # 1h 36 min
# timeList = [innovation, backpr_100, backpr_50, knn_100, knn_50]
# plot_data_timing(labelList, timeList)

#---------------------------------------------