# Algorithme d'apprentissage pour la robotique en essaim



### Expérience :

#### Foraging_Task : 

Simple tâche de foraging, les agents sont déployés dans une arène où des nodes interactibles sont présent. La fitness des robots est évalué par le nombre d'objets ramassés.



> Variables (accessibles dans Const.py)

```python
NB_ITEMS = 100			# Nombres de nodes initialisés dans l'arène
REGROWTH_TIME = 10		# Temps avant réapparition d'un node
CHANGE_POSITION = True	# Booléen, si True : Les nodes changent d'emplacement lors de leur réapparition

NB_ITER = 100000		# Durée d'évaluation
```



### Modifieurs:

#### Présence d'agents experts

Pour faciliter l'apprentissage, on peut placer des agents dans l'arène avec un comportement statique - considéré comme bon - pour la tâche à évaluer. Dans cette situation, les apprenant peuvent simplement apprendre des expert, ou également diffuser les informations apprises.



> Variables (accessibles dans Const.py)

```python
# NeuralLearner
PROPAGATION = False
# MemoryAgents
LEARNT_BEHAVIOUR_PROPAGATION = False

NB_LEARNER = 24 	# Nombre d'apprenant parmi les N robots
# Par défaut, N = 30
```

Implémentation par défaut :

```python3
def expertPolicy(self, observation: np.ndarray) -> tuple:
        '''
        @Overwrite
        '''

        # Check each frontal sensor for food :
        food_spots = []
        food_spotted = False
        for sensor_id in range(0, 5):
            if observation[(sensor_id*2)+1] == 1:
                food_spots.append(observation[sensor_id*2])
                food_spotted = True
            else:
                food_spots.append(2)
        if food_spotted:
            # Go toward the closest food source
            direction = np.argmin(food_spots)
            return c.EXPERT_SPEED, (direction - 2) * 0.5
        # Check each frontal sensor for obstacles
        if (observation[(2*2)]) == 1:  # There's nothing in front
            # if there's something on the left
            if (observation[(1*2)]) < 1:
                return c.EXPERT_SPEED, 0.5
            # if there's something on the right
            if (observation[(3*2)]) < 1:
                return c.EXPERT_SPEED, -0.5
            return c.EXPERT_SPEED, 0
        # If there's something in front
        if (observation[(0*2)]) < 1:  # and on the left
            return c.EXPERT_SPEED, 1  # Turn straight right
        return c.EXPERT_SPEED, -1  # else : turn straight left
```



#### Classe des agents :

##### MemoryAgent:

​	Cette classe génère des agents qui apprennent de manière directe, sans bruit, les mouvements des experts en fonction de leurs observation. On considère ainsi que les apprenants ont une vision parfaite de l'environnement des robots experts. Pour faciliter le tri des données, on discrétise les inputs.

​	Pour une situation donnée, l'agent apprenant va évaluer ses alentours (fonction sense), discrétiser ces valeurs puis vérifier si un expert lui a déjà transmit de telles observation, et dans ce cas, agira comme l'expert l'avait fait. Dans le cas d'une absence d'information, il adoptera un comportement de base (par défaut, la propagation de son réseau de neurones).

```python3
class MemoryAgent(Agent):
    def __init__(self, wm) -> None:
        '''
        Const
        '''
        super().__init__(wm)
        self.memory = {}
        self.total_data_size = 0                        # Tracker for debugging purpose
```



Evaluations :

