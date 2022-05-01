
#                                                robotsBehaviors


################################################################################################################
# IMPORTS
################################################################################################################




################################################################################################################
# PARAMETERS
################################################################################################################




################################################################################################################
# BRAITENBERG BEHAVIORS
################################################################################################################


def braitenberg_avoiderRobotsWalls(sensors):
    t = sensors["sensor_front"]["distance_to_wall"] + sensors["sensor_front"]["distance_to_robot"]
    r = ( 1 - sensors["sensor_front"]["distance_to_robot"] - sensors["sensor_front_left"]["distance_to_robot"] + sensors["sensor_front_right"]["distance_to_robot"] ) + ( sensors["sensor_front"]["distance_to_wall"] - 1 - sensors["sensor_front_left"]["distance_to_wall"] + sensors["sensor_front_right"]["distance_to_wall"] )

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------
    
def braitenberg_loveWalls(sensors):
    t = 1
    r = sensors["sensor_front_left"]["distance_to_wall"] - sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------

def braitenberg_hateWalls(sensors):
    t = sensors["sensor_front"]["distance_to_wall"]
    r = 1 - sensors["sensor_front"]["distance_to_wall"] - sensors["sensor_front_left"]["distance_to_wall"] + sensors["sensor_front_right"]["distance_to_wall"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------

def braitenberg_loveBot(sensors):
    t = 1
    r = sensors["sensor_front_left"]["distance_to_robot"] - sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)
    
#--------------------------------------------------------------------------------------------------------------  
    
def braitenberg_hateBot(sensors):
    t = sensors["sensor_front"]["distance_to_robot"]
    r = 1 - sensors["sensor_front"]["distance_to_robot"] - sensors["sensor_front_left"]["distance_to_robot"] + sensors["sensor_front_right"]["distance_to_robot"]

    return testControllersValidRange(t, r)



################################################################################################################
# SUBSOMPTION BEHAVIORS
################################################################################################################

def avoid_all(sensors): # used in roborobo examples, exploration behavior
    t = 1
    r = 0

    if sensors["sensor_front_left"]["distance_to_robot"] < 1            \
        or sensors["sensor_front_left"]["distance_to_object"] < 1       \
        or sensors["sensor_front_left"]["distance_to_wall"] < 1         \
        or sensors["sensor_front"]["distance_to_robot"] < 1             \
        or sensors["sensor_front"]["distance_to_object"] < 1            \
        or sensors["sensor_front"]["distance_to_wall"] < 1:
        r = 0.5
    elif sensors["sensor_front_right"]["distance_to_robot"] < 1         \
        or sensors["sensor_front_right"]["distance_to_object"] < 1      \
        or sensors["sensor_front_right"]["distance_to_wall"] < 1:
        r = -0.5

    return testControllersValidRange(t, r)


#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects(sensors):
    t = 1
    r = 0

    if sensors["sensor_left"]["distance_to_object"] < 1                \
        or sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or sensors["sensor_front"]["distance_to_object"] < 1 :
        r = -0.5
    elif sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or sensors["sensor_right"]["distance_to_object"] < 1 :
        r = 0.5
    else:
        return avoid_all(sensors)

    return testControllersValidRange(t, r)

#--------------------------------------------------------------------------------------------------------------  

def avoidRobotsWalls_getObjects_strongVersion(sensors):
    t = 0.5
    r = 0

    if sensors["sensor_left"]["distance_to_object"] < 1                \
        or sensors["sensor_front_left"]["distance_to_object"] < 1      \
        or sensors["sensor_front"]["distance_to_object"] < 1 :
        t = -1
        r = -1
    elif sensors["sensor_front_right"]["distance_to_object"] < 1       \
        or sensors["sensor_right"]["distance_to_object"] < 1 :
        t = -1
        r = 1
    else:
        return avoid_all(sensors)

    return testControllersValidRange(t, r)




################################################################################################################
# CHECK CONTROLLER VALID RANGE METHODS
################################################################################################################


def testControllersValidRange(t, r):
    # Limits the values of transition and rotation from -1 to +1
    t = max(-1, min(t, 1))
    r = max(-1, min(r, 1))
    return t, r

