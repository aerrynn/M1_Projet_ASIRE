### Manuel utilisateur
## Méthode par transmission de genome et méthode par transmission de comportements  

&nbsp;

- Se positionner sur le répertoire *part1Innovation_part2Diffusion\ASIRE_project*.
        
- Se positionner dans le fichier :
    - *part1\_main\_innovation.py* pour exécuter l'apprentissage par genome;
    - *part2\_main\_diffusion.py* pour exécuter l'apprentissage par comportements.  
        
- Dans le fichier *main* choisi, en début de page, fixer les paramètres à utiliser :
    - **configuration file parameters**. Fixer le nombre de individus experts *nbExpertsRobots*, le nombre d'individus focaux *nbNotExpertsRobots* et le nombre d'objets *nbFoodObjects pour l'expérience. Remarque : il ne faut pas modifier le fichier de configuration qui se trouve dans le répertoire *config*.
    - ***learning mode configuration*** (*part2*). Choisir la mèthode d'apprentissage pour l'expérience courante : commenter la ligne de la mèthode à ne pas utiliser.
    - ***parameters***. Fixer le nombre total de steps *nbSteps* de l'expérience et la taille de la base de données des individus *maxSizeDictMyBehaviors*. Affecter *True* à la variable *learningOnlyFromExperts* pour limiter la possibilité de diffuser  ses propres caractéristiques (genome ou traces comportamentales) aux seuls individus experts; affecter *False* pour permettre à tous les individu de diffuser.
    - ***debug and plot parameters***. Il est possible de visualiser les détails d'execution des algorithmes en affectant la valeur \textit{True} aux différentes variables *debug_*.
        
    Remarque : les fichiers *part1\_bestParameters.txt* et *part2\_bestParameters.txt* conservent une copie des meilleurs paramètres testés lors des expériences.
        
- Exécuter le fichier *main* choisi, *part1_main_innovation.py* ou *part2_main_diffusion.py*.
