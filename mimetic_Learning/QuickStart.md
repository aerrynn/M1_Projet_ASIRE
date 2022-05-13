Comment recréer les expériences citées :

Installer [Roborobo](https://github.com/nekonaute/roborobo4) et lancer une instance d'Anaconda.

Se déplacer dans le dossier mimetic_Learning :

Choisir les paramètres dans le fichier Cons.py :

```python
########################################################################################
#                                    Global                                            #
########################################################################################

VERBOSE = False                                                                 # Bool
DATA_SAVE = True                                                                # Bool
SAVE_FILE = 'NewData/Disc'
OVERWRITE_FILE = False
########################################################################################
#                                ExtendedAgents                                        #
########################################################################################
# Options pour l'apprentissage par réseau de neurones
NB_HIDDENS = 10
EVALUATION_TIME = 600				# Taille de la sliding window
MEMORY_RANGE = 20
LEARNING_STEPS = 30
LEARNING_RATE = 0.8
MUTATION_RATE = 0.
########################################################################################
#                                 Foraging Task                                        #
########################################################################################

LEARNING_ALGORITHM = "adhoc"        # "adhoc" or "neural"

NB_ITEMS = 100						# Nombre d'objets instanciés
REGROWTH_TIME = 10
CHANGE_POSITION = True				# Si les objets réapparaissent à d'autres endroits
NB_ITER = 200000					# Durée de l'expérimentation
########################################################################################
#                                 NeuralLearner                                        #
########################################################################################

PROPAGATION = False					# Apprentissage apprenant - apprenant
DECAY_FUNCTION = constantLearningRate	# Choix de la fonction de réduction de 
DECAY_RATIO = 0.01						# taux d'apprentissage
########################################################################################
#                                 MemoryAgents                                         #
########################################################################################

EXPERT_SPEED = 1
LEARNING_GAP = 60										
MEMORY_SIZE = 100					# Nombre de messages stockés simultanément
NB_LEARNER = 90						# Nombre de robots apprenants
DISCRETISE_RATIO = 2                                   	# if -1 : closest learnt
LEARNT_BEHAVIOUR_PROPAGATION = False	# Apprentissage apprenant - apprenant
########################################################################################

```

Exécuter le runner:

```bash
python Foraging_Task
```

