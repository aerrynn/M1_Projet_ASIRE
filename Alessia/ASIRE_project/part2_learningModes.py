
#                                  PART 2 : LEARNING MODES


################################################################################################################
# IMPORTS
################################################################################################################





################################################################################################################
# PARAMETERS
################################################################################################################





################################################################################################################
# LEARNING MODES
################################################################################################################


def learningMode_nNBackpropagation(self, inputLayer, cptStepsG, slidingWindowTime, nbEpoch, learningRate, debug_part2_diffusion):
    
    # 'slidingWindowTime==None' means that this robot trains the behaviors dataset each time it meets an expert
    # 'cptStepsG%slidingWindowTime==0' means that this robot trains the behaviors dataset at each 'slidingWindowTime' time
    if (slidingWindowTime == None and self.myHitee.newMessageReceived())     \
        or (slidingWindowTime != None and slidingWindowTime != 0 and cptStepsG%slidingWindowTime==0):

        dataset = list(self.dictMyBehaviors.keys())
        labels = [self.dictMyBehaviors[input] for input in dataset]
        self.myNetwork.train(dataset, labels, nbEpoch, learningRate)

        if debug_part2_diffusion :
            print(f"\n{self.str}\tNeural Network backtracking training performed at step {cptStepsG}")


    action = self.myNetwork.predict(inputLayer)

    if debug_part2_diffusion :
        print(f"\n{self.str}\tApprentice choice : t={action[0]}, r={action[1]}")
        print(f"\t\tobtained with supervised learning prediction 'forwardPropagation' method")
        print("\n\t\tbecause current sensors are :")
        print(f"\t\t{inputLayer[:self.halfSize]}")
        print(f"\t\t{inputLayer[self.halfSize:]}")
        print("\n---------------------------------------------------------------")

    return action[0], action[1]


#---------------------------------------------------------------------------------------------------------------

def learningMode_kNearestNeighbors(self, inputLayer, k, debug_part2_diffusion):
    action = self.myKnnClassifier.predict(inputLayer, self.dictMyBehaviors, k)

    if debug_part2_diffusion :
        print(f"\n{self.str} \tApprentice choice : t={action[0]}, r={action[1]}")
        print(f"\t\tobtained with k nearest neighbors prediction")
        print("\n---------------------------------------------------------------")
    
    return action[0], action[1]


#---------------------------------------------------------------------------------------------------------------