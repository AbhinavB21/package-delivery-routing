class Package:
    def __init__(self, id, agent_id):
        self.id = id
        self.agent_id = agent_id

    def get_id(self):
        return self.id
    
    def get_agent_id(self):
        return self.agent_id