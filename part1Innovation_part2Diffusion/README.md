Méthode par transmission de genome et méthode par transmission de comportements
    
    \begin{itemize}
        \item Se positionner sur le répertoire \textit{part1Innovation\_part2Diffusion\\ASIRE\_project}
        
        \item Se positionner dans le fichier :
        
        \begin{itemize}
        \item \textit{part1\_main\_innovation.py} pour exécuter l'apprentissage par genome;
        \item \textit{part2\_main\_diffusion.py} pour exécuter l'apprentissage par comportements.
        \end{itemize}
        
        \item Dans le fichier main choisi, en début de page, fixer les paramètres à utiliser :
        
        \begin{itemize}
        \item section \textbf{\textit{configuration file parameters}}. Fixer le nombre de individus experts \textit{nbExpertsRobots}, le nombre d'individus focaux \textit{nbNotExpertsRobots} et le nombre d'objets \textit{nbFoodObjects} pour l'expérience. Remarque : il ne faut pas modifier le fichier de configuration qui se trouve dans le répertoie \textit{config}.
        \item section \textbf{\textit{learning mode configuration}}. Choisir la mèthode d'apprentissage pour l'expérience courante : commenter la ligne de la mèthode à ne pas utiliser.
        \item section \textbf{\textit{parameters}}. Fixer le nombre total de steps \textit{nbSteps} de l'expérience et la taille de la base de données des individus \textit{maxSizeDictMyBehaviors}. Affecter \textit{True} à la variable \textit{learningOnlyFromExperts} pour limiter la possibilité de diffuser ses traces comportamentales aux seuls individus experts; affecter \textit{False} pour permettre à tous les individu de diffuser ses propres traces comportamentales.
        \item section \textbf{\textit{debug and plot parameters}}. Il est possible de visualiser les détails d'execution des algorithmes en affectant la valeur \textit{True} aux différentes variables \textit{debug\_*}.
        \end{itemize}
        
        Remarque : les fichiers \textit{part1\_bestParameters.txt} et \textit{part2\_bestParameters.txt} conservent une copie des meilleurs paramètres testés lors des expériences.
        
        \item Exécuter le fichier main choisi, \textit{part1\_main\_innovation.py} ou \textit{part2\_main\_diffusion.py}.
    
    \end{itemize}
