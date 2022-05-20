
#                                             getExtendedSensors


################################################################################################################
# IMPORTS
################################################################################################################




################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################
# getExtendedSensors retours the distance at robots, objects and walls murs for each robot's arm

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
    """ 
    Methode to get 24 new sensors from the 8 default sensors provided in roborobo.
    Returns :
        - sensors (dict) : ( distance, isRobot (bool), isObject (bool), isWall (bool), distance_to_robot, distance_to_object, distance_to_wall ) for each default sensor
        - tabExtSensors (liste) : ( distance_to_robot, distance_to_object, distance_to_wall ) for each default sensor

    :param self.nb_sensors : number of default sensors (assert : self.nb_sensors = 8)
    :param self.get_distance_at() : distance value at any obstacle (robot or object or wall)
    :param self.get_robot_id_at() : distance value at a robot
    :param self.get_object_at() : distance value at an object
    :param self.get_wall_at() : distance value at a wall
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
    """
    Method to convert 24 extended sensors to list, useful for perceptron input.
    
    Returns a tabExtSensors (liste) : ( distance_to_robot, distance_to_object, distance_to_wall ) for each default sensor
    
    :param sensors : dictionnary containing 24 extended sensors
    """

    tabExtSensors = []
    for key in sensors:
        tabExtSensors.append(sensors[key]["distance_to_robot"])
        tabExtSensors.append(sensors[key]["distance_to_object"])
        tabExtSensors.append(sensors[key]["distance_to_wall"])

    return tabExtSensors


#--------------------------------------------------------------------------------------------------------------

def extractExtSensors_bool(sensors):
    """
    Method to convert 24 extended sensors to list of bools, useful for perceptron input.
    
    Returns a tabExtSensors (liste) : ( distance, isRobot (bool), isObject(bool), isWall(bool) ) pour chaque senseur de base

    :param sensors : dictionnary containing 24 extended sensors
    """

    tabExtSensors = []
    for key in sensors:
        tabExtSensors.append(sensors[key]["isRobot"])
        tabExtSensors.append(sensors[key]["isObject"])
        tabExtSensors.append(sensors[key]["isWall"])

    return tabExtSensors