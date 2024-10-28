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

        # Drop location. Placeholder value for now
        self.drop_location = (0, 0)

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']
        self.rewards = {
            'UP': -1,
            'DOWN': -1,
            'LEFT': -1,
            'RIGHT': -1,
            'PICKUP': (self.grid_size * self.grid_size) * 2,
            'DROP': (self.grid_size * self.grid_size) * 5
        }
        self.package_picked = False

        self.reset()
    
    def get_state(self):
        if self.current_state['driver_position'] == self.drop_location:
            return 'dropped'
        elif self.current_state['package_picked'] == True and not self.package_picked:
            self.package_picked = True
            return 'picked'
        
        return False

    def reset(self):
        self.grid = self.create_grid()
        self.agents = self.add_agents()
        self.packages = self.assign_package()
        self.current_state = {
            'driver_position': (0, 0),
            'carry_package': False 
        }
        return 0, False


    def driver_turn(self, action):
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self.move_driver(action)
        elif action == 'PICKUP':
            return self.pickup_package()
        elif action == 'DROP':
            return self.drop_package()
        else:
            return 0
    
    def step(self, action):
        action_name = self.actions[action]
        reward = self.driver_turn(action)

        done = False
        state = self.get_state()
        if state == 'dropped':
            done = True
            reward += self.rewards['DROP']
        elif state == 'picked':
            reward += self.rewards['PICKUP']
        
        return done, reward


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

# double checking commit email