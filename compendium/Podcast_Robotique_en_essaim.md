## **Podcast « Robotique en essaim : tous pour un ! »** (franceculture.fr)
[https://t.co/g8HxXwbHUf](https://t.co/g8HxXwbHUf)

La robotique en essaim est une forme de robotique collective qui désigne l’ensemble de petit robots en miniature ayant des capacités de calcul individuelles faibles qui travaillent de façon coordonnée pour accomplir des taches complexes. Propriétés :

- *Prise de décision distribuée, comportements simples et locaux.* Chaque robot peut interagir de manière indépendante avec l’environnement et les robots proches, mais il ne peut pas avoir une connaissance globale de l’essaim. L’interaction entre individus est locale et n’a pas besoin de coordination globale centralisée.

- *Approche bio-inspiré et apprentissage social.* En imitant les interactions entre insectes sociaux ou entre collectives d’animaux il est possible de faire de l’apprentissage social, où les individus apprennent les uns des autres.

- *Robustesse du système.* Les robots sont faciles à construire (pas chers), ce qui constitue un avantage économique car si un robot se casse ou est défaillant, ses actions n’influent pas beaucoup sur l’action du groupe. La robotique en essaim garantit la réalisation des objectifs du groupe au-delà de l’éventuelle perte d’une partie des individus. Le faible cout des robots constitue un avantage

- *Homogénéité ou hétérogénéité de la programmation.* Souvent on considère que dans la robotique en essaim tous les robots sont interchangeables, possédant le même programme (homogénéité), et que leur taches se différencient uniquement en fonction des stimulus reçus dans l’environnement. Mais il est aussi possible de définir des taches spécialisées pour des sous-groupes d’individus (hétérogénéité) et ainsi répartir les rôles dans l’essaim : deux individus appartenant à deux rôles différents auront des comportements différents même en face d’une stimulus similaire. Le modèle hétérogène est moins étudiée.

- *Phénomène d’auto-organisation.* L’interaction microscopique à l’échelle des individus comporte une synergie qui a pour conséquence une auto-organisation du groupe. Si on a un grand nombre de robots on espère d’observer macroscopiquement l’émergence de comportements collectifs.

&ensp;

*Implémentation de l’essaim.* On utilise une programmation déclarative du genre « si je vois cette situation autour de moi, alors j’exécute telle action », appliquée sur des modèles tels que des automates ou des réseaux de neurones artificiels. Approches possibles :

- observer le comportement des fourmis et essayer de l’imiter, pour voir si l’auto-organisation converge en comportements collectifs

- utiliser l’apprentissage ou l’optimisation pour atteindre des objectifs de groupe fixés, en suite itérer plusieurs fois le procédé pour déterminer quel programme mettre dans chaque robot (ex : algorithmes génétiques).

> NB. Pour arriver à bien isoler un comportement individuel il faudrait une simulation d’ampleur non réalisable (l’espace de recherche est très grand), parcourant les millions d’années d’histoire évolutionniste qui ont amené aux comportements animaux que l’on observe aujourd’hui.

&ensp;

*Applications de la robotique en essaim*

- étude de groupes d’animaux sociaux en imitant leur comportement individuel, pour en déduire les comportements collectifs

- mis en place de systèmes hybrides animal-robot (= robot accolé à l’animal), capables de capturer les mouvements du groupe

- mélange de robots et d’animaux pour observer la communication entre les deux et, vu qu’on peut avoir le contrôle sur les robots, observer l’influence des comportements imposés sur les choix du groupe

&thinsp; 

*Exemples pratiques*

- Dans l’agriculture, des drones ou des robots à roues enlèvent les mauvaises herbes, éliminent les insectes, fertiliser...

- À Venise, 130 robots (de surface ou de profondeur) surveillent la lagune pour mesurer l’impact du tourisme sur l’écosystème

- étude de l’influence des vents solaires sur le champs magnétiques terrestres

- La Nasa utilise des nano-robots pour l’exploration de Mars

&ensp;

*Projets futurs*

- Les expériences de robotique en essaim doivent être répétées de nombreuses fois, car il faut calibrer le comportement du robot au sein de l’essaim vivant pour qu’il devienne acceptable par la population. Cela entraine un emploi de temps considérable pour la récolte des données et pour leur analyse. Il faut améliorer les algorithmes d’analyse des données pour qu’ils puissent produire un modèle le plus vite possible. Le modèle pourra en suite être programmé dans un robot spécifique, de façon à obtenir la duplication robotique d’un poisson vivant.

- S’intéresser à l’imitation du monde végétal, qui est très efficace pour la collecte et l’utilisation de ressources. En effet une plante peut être vue comme un essaim, où chaque component gère indépendamment les fonctions utiles au système globale.

- Utilisation des nano-robots pour le soin de l’être humaine (thromboses, tumeurs, ...).
