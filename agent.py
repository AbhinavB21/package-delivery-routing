import uuid

class Agent:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos

    def get_id(self):
        return self.id
    
    def get_pos(self):
        return self.pos