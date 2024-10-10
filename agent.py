import uuid

class Agent:
    def __init__(self, id, start_pos, env):
        self.id = uuid.uuid4()
        self.start_pos = start_pos
        self.env = env