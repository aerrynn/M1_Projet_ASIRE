from pyroborobo import Pyroborobo, Controller, AgentObserver, CircleObject
import numpy as np
import random

sliding_window = 400
transferRate = 0.5  # 0.9
maturationDelay = 400
mutationRate = 0
NB_ITER = 800400 # Nombre de steps

class Agent(Controller):
    def __init__(self, worldModel):
        Controller.__init__(self, worldModel)
        self.genome = [np.random.normal(0, .2) for _ in range(326)]
        self.pickedUpBalls = 0
        self.message = []
        self.fitness = 0
        self.age = 0

    def move(self):
        # 0 left sens des aiguilles d'une montre
        
        # translation = self.get_distance_at(4)
        # rotation = (1-self.get_distance_at(4) - self.get_distance_at(2) + self.get_distance_at(6))
        translation = 1
        rotation = 0
        for i in range(16):
            rotation += self.get_distance_at(i) * np.sum(self.genome[10*i:10*(i+1)])
            translation += self.get_distance_at(i) * np.sum(self.genome[160 + 10*i:160+10*(i+1)])
        translation = max(-1,min(translation,1))
        rotation = max(-1, min(rotation, 1))
        # r = random.randrange(-1,1)
        # print(r, t)
        self.set_translation(translation)
        self.set_rotation(rotation)

    def step(self):
        self.hit(mutationRate, transferRate, maturationDelay)

    def reset(self):
        pass
        # print("reset")

    def update_fitness(self):
        return self.pickedUpBalls

    # verifier comment marchent les signaux si elle existe

    def broadcast(self, transferRate, fitness):
        for i in range(self.nb_sensors):
            rob = self.get_robot_id_at(i)
            idx = random.choice(list(range(len(self.genome)), int(
                transferRate * len(self.genome))), False)
            rob.message += [(self.genome, idx, fitness)]

    def hit(self, mutationRate, transferRate, maturationDelay):
        self.move()
        self.fitness = self.update_fitness()
        newGenome = False
        if self.age >= maturationDelay:
            self.age += 1
            self.broadcast(transferRate, self.fitness)
            # incomingPackets = self.listen()
            for each in self.message:
                if each.fitness >= self.fitness:
                    self.copy(genome_bits)
                    # self.mutateGenome(mutationRate)
                    newGenome = True
            if newGenome:
                self.fitness = 0
                self.age = 0


class Ball(CircleObject):
    '''
    RessourceNode :
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''

    def __init__(self, id_, data):
        CircleObject.__init__(self, id_)
        self.rob = Pyroborobo.get()
        self.regrow_time = -1
        self.cur_regrow = -1

    def step(self):
        pass

    def reset(self):
        pass

    def is_walked(self, rob_id):  # Check id type
        # If the agent can store the ressource, increase it
        self.rob.controllers[rob_id].pickedUpBalls += 1
        print(f"{rob_id} picked an object, he's at his {self.rob.controllers[rob_id].pickedUpBalls}")
        self.hide()
        self.unregister()

    def inspect(self, prefix=""):
        return f"Resource_Node_{self.id}\n"


def main():
    rob = Pyroborobo.create(
        "test.properties",
        controller_class=Agent,
        object_class_dict = {'ball': Ball}
    )

    rob.start()
    rob.update(NB_ITER)
    Pyroborobo.close()


if __name__ == "__main__":
    main()
