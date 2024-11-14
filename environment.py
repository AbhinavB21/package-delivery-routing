import numpy as np
from agent import Agent
from package import Package
import random
import copy

class Environment:
    def __init__(self, num_agents, grid_size, num_obstacles):
        self.num_agents = num_agents
        self.grid_size = grid_size  
        self.num_packages = num_agents
        self.num_obstacles = num_obstacles
        self.start_positions = []  
        self.obstacles = []
        self.pickup_location = (0, 0)
        self.drop_location = (self.grid_size - 1, self.grid_size - 1)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']
        self.int_to_actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'PICKUP', 5: 'DROP'}
        self.move_penalty = -1
        self.rewards = {
            'UP': self.move_penalty,
            'DOWN': self.move_penalty,
            'LEFT': self.move_penalty,
            'RIGHT': self.move_penalty,
            'PICKUP': (self.grid_size * self.grid_size) * 2,
            'DROP': (self.grid_size * self.grid_size) * 5,
            'EMPTY': -10000
        }
        self.package_picked = False
        self.fuel = self.grid_size * self.grid_size * 30
        self.total_reward = 0
        self.done = False
        self.agents = []
        self.start_positions = []
        self.packages = None
        self.current_state = None
        self.reset()

    def reset(self):
        self.start_positions = []
        self.obstacles = []
        self.grid = self._create_grid()
        self.agents = self._add_agents()
        self.packages = self._assign_package()
        
        if not self.agents:  # Check if agents list is empty
            raise ValueError("No agents were created!")
        
        start_pos = self.agents[0].pos
        self.current_state = {
            'driver_position': start_pos,
            'package_picked': False
        }
        self.fuel = self.grid_size * self.grid_size * 30
        self.pickup_location = self._random_empty_position()
        while True:
            self.drop_location = self._random_empty_position()
            if self.drop_location != self.pickup_location:
                break
        self.done = False
        self.total_reward = 0
        return self.current_state, self.total_reward, self.done

    def step(self, action):
        if self.fuel != 0:
            reward = self._driver_turn(action)
        else:
            reward = self.rewards['EMPTY']
            self.done = True
        self.total_reward += reward
        return self.current_state, reward, self.done

    def _driver_turn(self, action):
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self._move_driver(action)
        elif action == 'PICKUP':
            return self._pickup_package()
        elif action == 'DROP':
            return self._drop_package()
        else:
            return 0

    def _move_driver(self, action):
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

        if new_pos[0] < 0 or new_pos[0] >= self.grid_size or new_pos[1] < 0 or new_pos[1] >= self.grid_size:
            return 0
        if self.grid[new_pos[0]][new_pos[1]] == 1:
            return 0

        self.current_state['driver_position'] = new_pos
        self.fuel -= 1
        return self.move_penalty

    def _pickup_package(self):
        if self.current_state['driver_position'] == self.pickup_location:
            self.current_state['package_picked'] = True
            return self.rewards['PICKUP']
        return 0

    def _drop_package(self):
        if self.current_state['driver_position'] == self.drop_location:
            self.current_state['package_picked'] = False
            self.done = True
            return self.rewards['DROP']
        return 0

    def _create_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=object)
        for _ in range(self.num_obstacles):
            x, y = self._random_empty_position(grid)
            grid[x][y] = 1
            self.obstacles.append((x, y)) 
        return grid
    
    def _add_agents(self):
        self.agents = []  # Clear existing agents
        for i in range(self.num_agents):
            start_pos = self._random_empty_position()
            self.start_positions.append(start_pos)
            new_agent = Agent(id=i, pos=start_pos)  
            self.agents.append(new_agent)
        return self.agents
    
    def _random_empty_position(self, grid=None):
        while True:
            x, y = np.random.randint(0, self.grid_size, size=2)
            if ((grid is None or grid[x][y] == 0)
                    and (x, y) not in self.start_positions
                    and (x, y) not in self.obstacles):
                return x, y

    def _assign_package(self):
        packages = []
        count = 0
        agent_list = self.agents.copy()  # Create a copy to avoid modifying original
        
        # Make sure we don't try to assign more packages than we have agents
        num_packages = min(self.num_packages, len(agent_list))
        
        for i in range(num_packages):
            if not agent_list:  # Check if we have any agents left
                break
            random_agent = random.choice(agent_list)
            package = Package(count, random_agent.id)
            agent_list.remove(random_agent)
            packages.append(package)
            count += 1
        
        return packages

