from math import trunc

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

        # placeholder values to be updated
        # TODO: update for multi-agent
        self.pickup_location = (0, 0)
        self.drop_location = (0, 0)

        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']
        self.move_penalty = -1
        self.rewards = {
            'UP': self.move_penalty,
            'DOWN': self.move_penalty,
            'LEFT': self.move_penalty,
            'RIGHT': self.move_penalty,
            'PICKUP': (self.grid_size * self.grid_size) * 2,
            'DROP': (self.grid_size * self.grid_size) * 5
        }
        self.package_picked = False

        self.total_reward = 0
        self.done = False
        self.grid = None
        self.agents = None
        self.packages = None
        self.current_state = None

        # initialize the grid, obstacles, agents, packages, current state
        self.reset()

    # returns reward and done status
    def reset(self) -> tuple[int, bool]:
        self.grid = self._create_grid()
        self.agents = self._add_agents()
        self.packages = self._assign_package()
        self.pickup_location = self._random_empty_position()
        self.drop_location = self._random_empty_position()
        self.current_state = {
            'driver_position': (0, 0),
            'package_picked': False
        }
        self.done = False
        self.total_reward = 0
        return self.total_reward, self.done

    # returns the reward for the given action and done status
    def step(self, action) -> tuple[int, bool]:
        reward = self._driver_turn(action)
        self.total_reward += reward
        return reward, self.done

    # returns reward
    def _driver_turn(self, action) -> int:
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self._move_driver(action)
        elif action == 'PICKUP':
            return self._pickup_package()
        elif action == 'DROP':
            return self._drop_package()
        else:
            return 0

    # returns reward
    def _move_driver(self, action) -> int:
        x, y = self.current_state['driver_position']

        if action == 'UP':
            new_pos = (x - 1, y)
        elif action == 'DOWN':
            new_pos = (x + 1, y)
        elif action == 'LEFT':
            new_pos = (x, y - 1)
        elif action == 'RIGHT':
            new_pos = (x, y + 1)
        else:
            return 0

        # checking for out of bounds
        if new_pos[0] < 0 or new_pos[0] >= self.grid_size or new_pos[1] < 0 or new_pos[1] >= self.grid_size:
            return 0

        # checking for obstacles
        if self.grid[new_pos[0]][new_pos[1]] == 1:
            return 0

        self.current_state['driver_position'] = new_pos
        return self.move_penalty

    # returns reward
    def _pickup_package(self) -> int:
        if self.current_state['driver_position'] == self.pickup_location:
            self.current_state['package_picked'] = True
            return self.rewards['PICKUP']
        return 0

    # returns reward
    def _drop_package(self) -> int:
        if self.current_state['driver_position'] == self.drop_location:
            self.current_state['package_picked'] = False
            self.done = True
            return self.rewards['DROP']
        return 0

    # creates the grid and adds obstacles
    def _create_grid(self) -> np.ndarray:
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        
        # place obstacles randomly
        for _ in range(self.num_obstacles):
            x, y = self._random_empty_position(grid)
            # setting mark on obstacle
            grid[x][y] = 1
            # keep track of obstacle positions
            self.obstacles.append((x, y)) 

        # return the initialized grid
        return grid
    
    # creates list of agents with a random starting position.
    def _add_agents(self) -> list[Agent]:
        agents = []
        for i in range(self.num_agents):
            start_pos = self._random_empty_position()
            self.start_positions.append(start_pos)
            agent = Agent(id=i, pos=start_pos)  
            agents.append(agent)
        return agents
    
    # helper function to find an empty spot on the grid
    def _random_empty_position(self, grid=None) -> tuple[int, int]:
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if ((grid is None or grid[x][y] == 0)
                    and (x, y) not in self.start_positions
                    and (x, y) not in self.obstacles):
                return x, y

    # assigns packages to agents
    # TODO: remove?
    def _assign_package(self) -> list[Package]:
        packages = []
        count = 0
        agent_list = self.agents
        for i in range(self.num_packages):
            random_agent = random.choice(agent_list)
            # Assign random agent_id to the package
            # TODO: should a Package know its own position?
            package = Package(count, random_agent.id)
            agent_list.remove(random_agent)
            packages.append(package)
        return packages

