import numpy as np
from agent import Agent
from package import Package
import random
from gym import spaces

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
        # TODO: package drop multiple or single?
        self.package_drop = (grid_size - 1, grid_size - 1)

        self.grid = self.create_grid()  
        self.agents = self.add_agents()
        # TODO: set the position of the package on the grid
        self.packages = self.assign_package()

        # each move consumes a fuel. if fuel reaches 0, game over
        self.fuel = grid_size * grid_size

        self.rewards = {
            'pick': grid_size * grid_size + 1,
            'movement': -1,
            'drop': grid_size * grid_size + 1,
        }

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        self.action_space = spaces.Discrete(len(self.actions))

        obs_space_dict = {
            'player_position': spaces.Tuple((spaces.Discrete(self.grid_size), spaces.Discrete(self.grid_size))),
            'fuel': spaces.Discrete(self.fuel),
            'package_positions': spaces.Dict({
                package: spaces.Discrete(self.grid_size)
                for package in self.packages
            })
        }
        self.observation_space = spaces.Dict(obs_space_dict)

        self.current_state = {}
    #     self.reset()


    def reset(self):
        """Resets the game to the initial state"""
        self.grid = self.create_grid()
        self.agents = self.add_agents()
        self.packages = self.assign_package()
        self.fuel = self.grid_size * self.grid_size
        self.current_state = {
            'player_position': np.random.choice(self.start_positions),
            'fuel': self.fuel,
            # TODO
            'package_positions': {
                package: pos for package, pos in self.packages
            }
        }
        return self.get_observation(), 0, False, {}

    # TODO
    def get_observation(self):
        guard_in_cell = None
        guard_positions = self.current_state['guard_positions']
        player_position = self.current_state['player_position']
        for guard in guard_positions:
            if guard_positions[guard] == player_position:
                guard_in_cell = guard
                break

        obs = {
            'player_position': self.current_state['player_position'],
            'player_health': self.health_state_to_int[self.current_state['player_health']],
            'guard_in_cell': guard_in_cell if guard_in_cell else None,
        }
        return obs

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

env = Environment(num_agents=1, grid_size=8, num_obstacles=2)
print(env.package_drop)