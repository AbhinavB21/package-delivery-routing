import uuid

class Agent:
    def __init__(self, id, pos, env):
        self.id = id
        self.pos = pos
        self.env = env
        self.package_dropped = False # added a checker for package dropping

    # movement functionality (also very simple)
    def move_up(self):
         self.pos = (self.pos[0], self.pos[1] + 1)
    def move_down(self):
        self.pos = (self.pos[0], self.pos[1] - 1)
    def move_right(self):
        self.pos = (self.pos[0] + 1, self.pos[1])
    def move_left(self):
        self.pos = (self.pos[0] - 1, self.pos[1])

    # dropping package (very simple)
    def drop_package(self):
        if not self.package_dropped: 
            self.package_dropped = True
            print("Package droppped at position {self.pos} in environment.")
        else:
            print("Package already dropped.")
            
