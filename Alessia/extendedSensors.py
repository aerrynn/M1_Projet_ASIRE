
#                                                getExtendedSensors


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
    
    assert self.nb_sensors == 8, "getExtendedSensors only works with 8 sensors"

    sensors = {}
    labels = ["sensor_left", "sensor_front_left", "sensor_front", "sensor_front_right", "sensor_right", "sensor_back_right", "sensor_back", "sensor_back_left"]

    #sensors = distance, distance to robot, distance to object, distance to wall for each arm
    for i in range(self.nb_sensors):
        sensors[labels[i]] =                            \
            {"distance": self.get_distance_at(i),       \
            "isRobot": self.get_robot_id_at(i) != -1,   \
            "isObject": self.get_object_at(i) != -1,    \
            "isWall": self.get_wall_at(i)}

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

    return sensors, 24


