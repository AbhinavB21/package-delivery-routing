import numpy as np
from agent import Agent
from package import Package
import random

class Environment:
    def __init__(self, num_agents, grid_size, num_obstacles):
        self.num_agents = num_agents
        self.grid_size = grid_size # n * n
        self.num_packages = num_agents
        self.num_obstacles = num_obstacles
        self.start_pos = None # TODO


        self.agents = self.add_agents()
        self.packages = self.assign_package()

    # Creates list of agents with a random starting position. TODO create random starting pos
    def add_agents(self):
        agents = []
        for i in self.num_agents:
            agent = Agent(self.start_pos)
            agents.append(agent)
        return agents
        
    def create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)

        for i in range(self.num_packages):
            grid[x][y] = "P"
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
        
        return grid
    
    def assign_package(self):
        packages = []
        count = 0
        agent_list = self.agents
        for i in range(self.num_packages):
            random_agent = random.choice(agent_list)
            # Assign random agent_id to the package
            package = Package(count, random_agent.id)
            agent_list.remove(random_agent)
        return packages
