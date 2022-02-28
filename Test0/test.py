from pyroborobo import Pyroborobo, Controller, AgentObserver, CircleObject, WorldObserver
import numpy as np
import random

simulation_mode_at_startup = 0

sliding_window = 100
transferRate = 0.5  # 0.9
maturationDelay = 100
mutationRate = 0
NB_ITER = 800400  # Nombre de steps
EPSILON = 1


TIME_SUBSUMPTION = 10

LISTE_TIRAGE = list(range(326))

POSITION_A_X = 10
POSITION_A_Y = 10
POSITION_B_X = 100
POSITION_B_Y = 100

DIAG = (POSITION_A_X - POSITION_B_X)**2 + (POSITION_A_Y - POSITION_B_Y) ** 2

compt = 0


def randNotZero():
    r = random.random()*2-1
    while r   == 0:
        r = random.random()*2-1
    # print(r)
    return r

class Agent(Controller):
    def __init__(self, worldModel):
        Controller.__init__(self, worldModel)
        self.genome = [randNotZero() for _ in range(326)]
        self.pickedUpBalls = 0
        self.message = []
        self.fitness = [5000, 5000]
        self.age = 0
        self.rob = Pyroborobo.get()

        self.last_pos = (0,0)
        self.eval_time = 0

        self.subsumption = 0

        self.to_A = True
        self.A_to_B = True

    def move(self, t):
        # 0 left sens des aiguilles d'une montre

        # translation = self.get_distance_at(4)
        # rotation = (1-self.get_distance_at(4) - self.get_distance_at(2) + self.get_distance_at(6))
        s = abs(self.last_pos[0] - self.absolute_position[0]) \
        + abs(self.last_pos[1] - self.absolute_position[1])
        # if s < EPSILON :
        #     # print(f"{self.id} is stuck {s}")
        #     self.subsumption = TIME_SUBSUMPTION
        
        # if self.subsumption > 0:
        #     self.subsumption -= 1
        #     self.set_translation (-1)
        #     # self.set_translation(np.random.randint(-1,2)*0.25)
        #     self.last_pos = self.absolute_position
        #     return 
        # if(t % 10 == 0):
        #     self.last_pos = self.absolute_position
        translation = 1
        rotation = 1
        if self.A_to_B:
            for i in range(16):
                rotation += self.get_distance_at(i) * \
                    np.sum(self.genome[5*i:5*(i+1)])
                translation += self.get_distance_at(i) * \
                    np.sum(self.genome[80 + 5*i:80+5*(i+1)])
        else:
            for i in range(16):
                rotation += self.get_distance_at(i) * \
                    np.sum(self.genome[160+5*i:160+5*(i+1)])
                translation += self.get_distance_at(i) * \
                    np.sum(self.genome[240 + 5*i:240+5*(i+1)])
        translation = max(-1, min(translation, 1))
        print(rotation)
        rotation = max(-1, min(rotation, 1))
        self.set_translation(translation)
        self.set_rotation(rotation)

    def step(self):
        for i in range(self.nb_sensors):
            det = self.get_object_at(i)
            if det == -1:
                continue
            det -= 1
            if self.rob.objects[det].name == 'A':  # A
                if self.to_A:
                    self.to_A = False
                    print(f"I {self.id} got to A")
                if not self.A_to_B:
                    self.fitness[0] = self.eval_time
                    self.A_to_B = True
                    self.eval_time = 0
                    print(f"I {self.id} got to A")
            if self.rob.objects[det].name == 'B':  # B
                if not self.to_A:
                    if self.A_to_B:
                        self.fitness[1] = self.eval_time
                        self.A_to_B = False
                        self.eval_time = 0
                        print(f"I {self.id} got to B")
        if not self.to_A:
            self.eval_time += 1
        self.hit(mutationRate, transferRate, maturationDelay)

    def reset(self):
        pass
        # print("reset")

    def update_fitness(self):
        return self.fitness
    # verifier comment marchent les signaux si elle existe

    def broadcast(self, transferRate, fitness):
        # print("HERE")
        for i in range(self.nb_sensors):
            rob_id = self.get_robot_id_at(i)
            if rob_id == -1:
                continue
            taille_remplacement = int(transferRate * len(self.genome))
            # print(LISTE_TIRAGE, taille_remplacement, False)
            idx = np.random.choice(
                LISTE_TIRAGE, taille_remplacement, False)
            self.rob.controllers[rob_id].message += [
                (self.genome, idx, fitness, rob_id)]

    def hit(self, mutationRate, transferRate, maturationDelay):
        self.move(self.age)
        self.fitness = self.update_fitness()
        newGenome = False
        # print(self.age)
        if self.age >= maturationDelay:
            self.broadcast(transferRate, self.fitness)
            # incomingPackets = self.listen()
            for m in self.message:
                newGenome = self.transfer(m)
            if newGenome:
                self.fitness = [5000, 5000]
                self.age = 0
        self.age += 1

    def transfer(self, message):
        state = False
        n_G, n_idx, n_F, m_id= message
        if self.id == 25:
            old = np.array(self.genome)            
        for i in range(2):
            if n_F[i] <= self.fitness[i]:
                for each in n_idx:
                    self.genome[each] = n_G[each]
                state = True
                # print("Transmission")
        if self.id == 25:
            print(f"after meeting {m_id} : {old - np.array(self.genome)}")
        return state


class Ball(CircleObject):
    '''
    RessourceNode :
        TODO: Fill Description 
        What ?
        Why ?
        How ?
    '''

    def __init__(self, id_, data, name):
        CircleObject.__init__(self, id_)
        self.rob = Pyroborobo.get()
        self.regrow_time = 100
        self.cur_regrow = 0
        self.name = name

    def step(self):
        pass

    def reset(self):
        print("Reset")
        pass

    # def is_pushed(self, id_, speed):
    #     super().is_pushed(id_, speed)
    #     print(f"I'm kicked by {id_}")

    # def is_touched(self, id_):
    #     super().is_touched(id_)
    #     print("TOUCHED")
        # print(id_)
        # self.rob.controllers[rob_id].pickedUpBalls += 1
        # self.hide()
        # self.unregister()

    def is_walked(self, rob_id):  # Check id type
        pass

    def inspect(self, prefix=""):
        return f"I'm {self.name} \n"


class MyWorldObserver(WorldObserver):
    def __init__(self, world):
        super().__init__(world)
        self.rob = Pyroborobo.get()

    def init_pre(self):
        super().init_pre()

    def init_post(self):
        super().init_post()

        # init A
        a = Ball(0, {}, 'A')
        a = self.rob.add_object(a)
        a.unregister()
        a.set_coordinates(POSITION_A_X, POSITION_A_Y)
        retValue = a.can_register()
        a.register()
        a.show()

        b = Ball(1, {}, 'B')
        b = self.rob.add_object(b)
        b.unregister()
        b.set_coordinates(POSITION_B_X, POSITION_B_Y)
        retValue = b.can_register()
        b.register()
        b.show()
        #ball.set_coordinates(750, 799)

        # x, y = 20, 20
        # #delta_X, delta_Y = 40, 40
        # delta_X, delta_Y = 20, 40
        # for robot in self.rob.controllers:
        #     robot.set_absolute_orientation(random.randint(-180,180))
        #     robot.set_position(x, y)
        #     x = x + delta_X
        #     if x > 780:
        #         x = 20
        #         y += delta_Y
        # 2021-04-27 BP -- il faut créer les robots directements en python sinon ils doivent trouver un place au hasard avant

        print("End of worldobserver::init_post")

    def step_pre(self):
        super().step_pre()

    def step_post(self):
        super().step_post()


def printme():
    compt += 1
    print(compt)


def main():
    rob = Pyroborobo.create(
        "config/test2.properties",
        controller_class=Agent,
        world_observer_class=MyWorldObserver,
        object_class_dict={'ball': Ball}
    )

    rob.start()
    rob.update(NB_ITER)
    Pyroborobo.close()


if __name__ == "__main__":
    main()
