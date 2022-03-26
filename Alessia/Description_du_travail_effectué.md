## **Description du travail effectué : liste des fichiers en ordre chronologique**     
> NB. Tous les fichiers cités dans ce document sont testés, ils compilent et s'éxecutent sans erreurs.  

&ensp;

- ***prova1.py, prova2.py, prova3.py*** : prise en main de roborobo, définition des premiers essaims et des premiers objets.  

- ***trial1+hit_ee_v1.py*** : prèmiere expèrience de [FORAGING TASK], prèmieres approches.
    - contexte : introduction de 2 agents experts et 28 agents non experts, 60 Food_Object(CircleObject) qui disparaissent lorsqu'ils sont touchés par un robot.
    - expertBehavior : exploration de l'éspace avec un mécanisme de subsomption avoider qui évite toute entité aperçue sur les 3 senseurs frontaux. On attribue à chaque expert un vecteur de valeurs nulles à transmettre aux robots non experts via hit_ee.
    - swarmBehavior : réseau de neurones (combinaison linéaire simple 8 senseurs de base @ poids) avec des valeurs de base pour t et r, de manière à se rapporcher du comportement de l'expert lorsque la partie neuronale tend à zéro.
    - fitness : nombre de Food_Object collectés (tableau de nbRobots cases).  

- ***trial1.py*** : équivalent à *trial1+hit_ee_v1.py*, constitue une prémière structuration du code, notamment on sépare hit_ee de la classe Controller.  

- ***hit_ee.py*** : algorithmes hit_ee, pour l'instant il n'y a que la version simple de l'article "HIT-EE : a Novel Embodied Evolutionary Algorithm for Low Cost Swarm Robotics".  

- ***extendedSensors.py*** : méthodes permettant de déduire des nouveaux senseurs élaborés obtenus à partir des 8 senseurs de base.  

- ***trial2.py*** : [FORAGING TASK] similaire à *trial1.py*. Noveautés introduites :
    - utilisation de 24 senseurs pour le réseau de neurones :
            - *distance_to_robot* pour chaque capteur de base ;
            - *distance_to_object* pour chaque capteur de base ;
            - *distance_to_wall* pour chaque capteur de base.
    - mise en place de nouveaux comportements de subsomption pour les experts, tels que *avoidRobotWalls_getObjects*
    - methode *compute_neuralNetwork* pour déduire t, r à partir d'un genome et d'un ensemble de senseurs de n'importe quelle taille.

- ***robotsBehaviors.py*** : récolte de comportements définis, principalement pour les robots experts.  



