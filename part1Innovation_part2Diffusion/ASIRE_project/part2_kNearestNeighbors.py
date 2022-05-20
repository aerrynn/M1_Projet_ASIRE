
#                                   k Nearest Neighborgs


################################################################################################################
# IMPORTS
################################################################################################################

from math import sqrt, floor
from statistics import mode
from tracemalloc import start



################################################################################################################
# PARAMETERS
################################################################################################################

debug = None
debugAccuracy = None


################################################################################################################
# K NEAREST NEIGHBORGS
################################################################################################################


class knn():

    def __init__(self, debugKNN=False, debugAcc=False, strRId=None):
        """

        """
        
        global strRobotId, debug, debugAccuracy
        strRobotId = strRId
        debug = debugKNN
        debugAccuracy = debugAcc

        self.halfSize = None


    #-------------------------------------------------------------

    def behaviorsEuclideanDistance(self, behavior1, behavior2):
        """
        Returns the euclidean distance between les elements in behavior1 (expert) and behavior2
        """
        d = 0
        for i in range(len(behavior1)):
            d += (behavior1[i] - behavior2[i])**2
        return sqrt(d)


#---------------------------------------------------------------------------------------------------------------

    def getKNearestNeighbors(self, behavior1, dataset, k):
        """
        Returns the list of nearest neighbors

        :param behavior1 : input from environment
        :param dataset : list of behaviors converted in lists from the robot's database
        :param k : number of Nearest Neighbors to consider
        """

        tabNearestNeighbors = []
        tabDist = []

        for b2 in range(len(dataset)):
            d = self.behaviorsEuclideanDistance(behavior1, dataset[b2])
            tabDist.append((b2, d))
        
        tabDist.sort(key = lambda x: x[1]) # sort in decreasing order of the second tuple's element
        for row, d in tabDist[:k]:
            tabNearestNeighbors.append(dataset[row])

        return tabNearestNeighbors


    #-------------------------------------------------------------

    def predict(self, behavior1, dictMyBehaviors, k, notAccuracyRunning=True):
        """
        Returns the prediction (action [t,r])

        :param behavior1 : input from environment
        :param dictMyBehaviors : list of behaviors from the robot's database
        :param k : number of Nearest Neighbors to consider
        """

        dataset = list(dictMyBehaviors.keys())
        outputs = []
        strOutputs = []
        neighbors = self.getKNearestNeighbors(behavior1, dataset, k) # most similar sensors
        for sensor in neighbors:
            outputs.append(dictMyBehaviors[sensor])
            strOutputs.append(str(dictMyBehaviors[sensor]))

        i = strOutputs.index(mode(strOutputs))
        mostCommonAction = outputs[i]

    
        if notAccuracyRunning and debug:
            if self.halfSize == None:
                self.halfSize = int(len(behavior1)/2)

            print(f"\n{strRobotId}\tTRAINING K Nearest Neighbors PROCESS DETAILS :          (k={k})")
            print("\n\t\tBehavior to match :")
            print(f"\t\t{behavior1[:self.halfSize]}")
            print(f"\t\t{behavior1[self.halfSize:]}")
            print(f"\n\t\tNearest Neighbors are:")
            for row in range(len(neighbors)):
                print("\t\t(" + ' '.join(str(elem) for elem in neighbors[row]) + ") :", outputs[row])
            print(f"\n\t\tPrediction found for 'behavior to match' : {mostCommonAction}")
            print(f"\n\t\tPrediction accuracy : {self.getAccuracy(dictMyBehaviors, k)}")
            print("\n---------------------------------------------------------------")

        return mostCommonAction


    #-------------------------------------------------------------

    def getAccuracy(self, dictMyBehaviors, k, nbTests=10):
        """ 
        Returns the estimation of the prediction quality.
        Default is 1/10 behaviors for test, 9/10 behaviors for train, for each 1/10 group of data
        
        :param dictMyBehaviors : list of behaviors from the robot's database
        :param k : number of Nearest Neighbors to consider
        """
        accuracyAverage = 0
        
        regionSize = floor(len(dictMyBehaviors)/nbTests)
        if regionSize < 3:
            return f"insufficient behaviors database length (len={len(dictMyBehaviors)}) to compute accuracy"

        if debugAccuracy:
            print(f"\n\n\n>> ACCURACY K Nearest Neighbors DETAILS :")

        for i in range(nbTests):
            
            startingPoint = i*regionSize
            trainRows = []
            testRows = [j for j in range(startingPoint, startingPoint+regionSize)]
            
            allSensors = list(dictMyBehaviors.keys()) # all sensors in the robot's behaviors database
            dict_train = {}
            dict_test = {}

            # Construction of dataset and labelset for train and test 
            for x in range(len(allSensors)):
                
                sensors = tuple(allSensors[x])
                if x in testRows:
                    dict_test[sensors] = dictMyBehaviors[sensors]
                else:
                    dict_train[sensors] = dictMyBehaviors[sensors]
                    trainRows.append(x)

            if debugAccuracy:
                print(f"\n>> Test n.{i+1} of {nbTests}")
                print(f">> Behaviors Database length : {len(dictMyBehaviors)}")
                print(f">> Training set length : {len(dict_train)} (rows : {trainRows})")
                print(f">> Test set length : {len(dict_test)} (rows : {testRows})")


            singleTestAccuracy = 0
            cpt = 0
            for behavior in list(dict_test.keys()):
                predictedOutput = self.predict(behavior, dict_train, k, notAccuracyRunning=False)
                expectedOutput = dict_test[behavior]
                if predictedOutput == expectedOutput:
                    singleTestAccuracy += 1

                if debugAccuracy:
                    cpt += 1
                    print(f"\n\t>> Behavior row n.{cpt} of {len(dict_test)}:")
                    print(f"\t>> {behavior[:self.halfSize]}")
                    print(f"\t>> {behavior[self.halfSize:]}")
                    print(f"\t>> predictedOutput : {predictedOutput}")
                    print(f"\t>> expectedOutput : {expectedOutput}")
                    print(f"\t>> Correct : {predictedOutput==expectedOutput}")
            
            a = (singleTestAccuracy * 100)/regionSize
            accuracyAverage += a # we keep the accuracy % value of each test

            if debugAccuracy:
                print(f"\n\t>> Prediction accuracy for test n.{i+1} : {round(a,1)} %")
                print(f"\t>> Prediction accuracy cumulation (sum) : {round(accuracyAverage,1)}")
                print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


        accuracyAverage = round(accuracyAverage/nbTests,1)

        if debugAccuracy:
            print(f"\n\t>> Final avarage accuracy : {round(accuracyAverage,1)} %")
            print("\n>> FIN K Nearest Neighbors DETAILS\n\n\n")

        return str(accuracyAverage) + " %"

    #-------------------------------------------------------------
