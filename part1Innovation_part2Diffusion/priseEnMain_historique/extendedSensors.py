
#                                             getExtendedSensors


################################################################################################################
# IMPORTS
################################################################################################################




################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################
# getExtendedSensors retourne la distance aux robots, objets et murs pour chaque bras du robot

# arm 0 : sensor_left
# arm 1 : sensor_front_left
# arm 2 : sensor_front
# arm 3 : sensor_front_right
# arm 4 : sensor_right
# arm 5 : sensor_back_right
# arm 6 : sensor_back
# arm 7 : sensor_back_left

################################################################################################################


def get24ExtendedSensors(self):
    """ Méthode pour obtenir 24 nouveaux senseurs deduits à partir des 8 senseurs de base.
    Retourne :
        - sensors (dictionnaire) : ( distance, isRobot (bool), isObject (bool), isWall (bool), distance_to_robot, distance_to_object, distance_to_wall ) pour chaque senseur de base
        - tabExtSensors (liste) : ( distance_to_robot, distance_to_object, distance_to_wall ) pour chaque senseur de base

    Paramètres:
    :param self.nb_sensors : nombre de senseurs de base d'un robot (assert : self.nb_sensors = 8)
    :param self.get_distance_at() : valeur de la distance à un obstacle (robot ou objet ou wall)
    :param self.get_robot_id_at() : valeur de la distance à un robot
    :param self.get_object_at() : valeur de la distance à un objet
    :param self.get_wall_at() : valeur de la distance à un wall
    """
    
    assert self.nb_sensors == 8, "get24ExtendedSensors only works with 8 base sensors"

    sensors = {}
    labels = ["sensor_left", "sensor_front_left", "sensor_front", "sensor_front_right", "sensor_right", "sensor_back_right", "sensor_back", "sensor_back_left"]

    #sensors = distance, distance to robot, distance to object, distance to wall for each arm
    for i in range(self.nb_sensors):
        sensors[labels[i]] =                            \
            {"distance": self.get_distance_at(i),       \
            "isRobot": self.get_robot_id_at(i) != -1,   \
            "isObject": self.get_object_at(i) != -1,    \
            "isWall": self.get_wall_at(i) == 1} 


    for key in sensors:
        sensors[key]["distance_to_robot"] = 1.0
        sensors[key]["distance_to_object"] = 1.0
        sensors[key]["distance_to_wall"] = 1.0
        if sensors[key]["isRobot"] == True:
            sensors[key]["distance_to_robot"] = sensors[key]["distance"]
        if sensors[key]["isObject"] == True:
            sensors[key]["distance_to_object"] = sensors[key]["distance"]
        if sensors[key]["isWall"] == True:
            sensors[key]["distance_to_wall"] = sensors[key]["distance"]


    return sensors


#--------------------------------------------------------------------------------------------------------------

def extractExtSensors_float(sensors):
    """Méthode pour extraire les 24 extended sensors sous forme de tableau, de manière à les utiliser comme input dans le perceptron.
    Retourne un tabExtSensors (liste) : ( distance_to_robot, distance_to_object, distance_to_wall ) pour chaque senseur de base
    
    :param sensors : dictionnaire representant les 24 extended sensors
    """

    tabExtSensors = []
    for key in sensors:
        tabExtSensors.append(sensors[key]["distance_to_robot"])
        tabExtSensors.append(sensors[key]["distance_to_object"])
        tabExtSensors.append(sensors[key]["distance_to_wall"])

    return tabExtSensors


#--------------------------------------------------------------------------------------------------------------

def extractExtSensors_bool(sensors):
    """Méthode pour extraire les 24 extended sensors sous forme de tableau, de manière à les utiliser comme input dans le perceptron.
    Retourne un tabExtSensors (liste) : ( distance, isRobot (bool), isObject(bool), isWall(bool) ) pour chaque senseur de base
    
    :param sensors : dictionnaire representant les 24 extended sensors
    """

    tabExtSensors = []
    for key in sensors:
        tabExtSensors.append(sensors[key]["isRobot"])
        tabExtSensors.append(sensors[key]["isObject"])
        tabExtSensors.append(sensors[key]["isWall"])

    return tabExtSensors