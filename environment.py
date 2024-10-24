import numpy as np
from agent import Agent
from package import Package
import random

class Environment:
    def __init__(self, num_agents, grid_size, num_obstacles):
        self.num_agents = num_agents
        # n * n grid
        self.grid_size = grid_size  
        # amount of agents
        self.num_packages = num_agents
        # amount of obstacles
        self.num_obstacles = num_obstacles
        # store agent starting positions
        self.start_positions = []  
        # store obstacles
        self.obstacles = [] 

        self.grid = self.create_grid()  
        self.agents = self.add_agents()
        self.packages = self.assign_package()

    # creates the grid and adds obstacles
    def create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        
        # place obstacles randomly
        for _ in range(self.num_obstacles):
            x, y = self.random_empty_position(grid)
            # setting mark on obstacle
            grid[x][y] = "O"  
            # keep track of obstacle positions
            self.obstacles.append((x, y)) 

        # return the initialized grid
        return grid
    
    # creates list of agents with a random starting position.
    def add_agents(self):
        agents = []
        for i in range(self.num_agents):
            start_pos = self.random_empty_position()
            self.start_positions.append(start_pos)
            agent = Agent(id=i, pos=start_pos)  
            agents.append(agent)
        return agents
    
    # helper function to find an empty spot on the grid
    def random_empty_position(self, grid=None):
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if grid is None or grid[x][y] == 0:
                return (x, y)
    
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
