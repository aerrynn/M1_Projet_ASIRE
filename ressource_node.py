from pyroborobo import Pyroborobo
from pyroborobo import CircleObject, Pyroborobo


class ResourceNode(CircleObject):
    def __init__(self, id_, data):
        CircleObject.__init__(self, id_)

        self.regrow_time = 100 			# After depleting, the node takes regrow_time steps to regrow
        self.cur_regrow = 0 			# While regrow > 0, the node is depleted, and not interactible
        self.max_ressource_level = 10   # Max amount of ressources
        self.ressource_level = 10		# Current amount of ressources
        # Boolean (faster to check than cur_regrow > 0)
        self.regrowing = False
        self.rob = Pyroborobo.get() 	# Get pyroborobo  ## Why ?

    def reset(self):
        self.show()
        self.register()
        self.regrowing = False
        self.cur_regrow = 0

    def step(self):
        if self.regrowing:
            self.cur_regrow -= 1		# Reduce the left regrowth time
            if self.cur_regrow <= 0: 	# The node has finished regrowing
                self.show()				# Displaying the node
                self.register()
                # Once regrown, set the node to max ressource
                self.ressource_level = self.max_ressource_level
                self.regrowing = False  # Regrowth ended

    def is_walked(self, rob_id):
        # If the agent can store the ressource, increase it 
        if rob_id.max_ressource_level - rob_id.current_ressource > 0:
            rob_id.current_ressource += 1
            self.ressource_level -= 1
            if self.ressource_level <= 0:  # Start the regrowth cycle
                self.regrowing = True
                self.cur_regrow = self.regrow_time
                self.hide()
                self.unregister()

    def inspect(self, prefix=""):
        return f"Ressource_Node_{self.id}\n \
        Size={self.ressource_level}"
