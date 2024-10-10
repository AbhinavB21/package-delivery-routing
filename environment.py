import numpy as np

class Environment:
    def __init__(self, num_agents, grid_size, num_packages, num_obstacles):
        self.grid_size = grid_size
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles

        self.agents = []

    
    def add_agents(self, agent):
        self.agents.append(agent)
        
    def create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)

        for i in range(self.num_packages):
            grid[x][y] = "P"
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
        
        return grid