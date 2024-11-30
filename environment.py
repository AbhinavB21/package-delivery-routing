import gym
import numpy as np
import random

class PackageEnv(gym.Env):
    def __init__(self, num_agents):
        super(PackageEnv, self).__init__()

        self.num_agents = num_agents
        self.num_packages = num_agents
        self.grid_size = 5
        self.roads = self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.goal_room = (self.grid_size - 1, self.grid_size - 1)

        self.rewards = {
            'UP': -1,
            'DOWN': -1,
            'LEFT': -1,
            'RIGHT': -1,
            'PICKUP': 5000,
            'DROP': 10000,
            'EMPTY': -10000
        }

        # Actions
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']

        self.reset()

    def reset(self):
        available_positions = [(i, j) for i, j in self.rooms if (i, j) != self.goal_room]

        # Sample positions for agents, packages, AND obstacles
        total = self.num_agents + self.num_packages + 2  # +2 for obstacles
        all_positions = random.sample(available_positions, total)
        
        # Distribute the positions
        self.agent_positions = all_positions[:self.num_agents]
        self.package_positions = all_positions[self.num_agents:self.num_agents + self.num_packages]
        self.obstacles = all_positions[self.num_agents + self.num_packages:]  # Last 2 positions become obstacles

        self.package_picked = [False] * self.num_agents
        self.fuel_consumed = [100] * self.num_agents

        self.current_state = {
            'agent_positions': self.agent_positions,
            'package_positions': self.package_positions,
            'package_picked': self.package_picked,
            'fuel_consumed': self.fuel_consumed
        }

        return self.current_state, 0, False

    def is_terminal(self, agent_id):
        if self.current_state['agent_positions'][agent_id] == self.goal_room:
            if self.current_state['package_picked'][agent_id]:
                return 'DROP'
        if self.current_state['fuel_consumed'][agent_id] <= 0:
            return 'EMPTY'
        return False

    def move_agent(self, agent_id, action):
        current_pos = self.current_state['agent_positions'][agent_id]
        x, y = current_pos

        directions = {
            'LEFT': (x - 1, y),
            'RIGHT': (x + 1, y),
            'UP': (x, y - 1),
            'DOWN': (x, y + 1)
        }

        new_pos = directions.get(action, self.current_state['agent_positions'][agent_id])

        # Check if move is valid (within bounds AND not into obstacle)
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size and 
            new_pos not in self.obstacles):
            self.current_state['agent_positions'][agent_id] = new_pos
            self.current_state['fuel_consumed'][agent_id] -= 1
            return f"Moved to {self.current_state['agent_positions']}", self.rewards[action]
        else:
            # Stay in current position if move is invalid
            self.current_state['fuel_consumed'][agent_id] -= 1  # Still consume fuel for attempted move
            return "Cannot move: Obstacle or boundary in the way", self.rewards[action]
    
    def pickup_package(self, agent_id):
        if self.current_state['agent_positions'][agent_id] == self.current_state['package_positions'][agent_id]:
            if not self.current_state['package_picked'][agent_id]:
                self.current_state['package_picked'][agent_id] = True
                return "Picked up package", self.rewards['PICKUP']
        return "No package to pickup", 0

    def drop_package(self, agent_id):
        if self.current_state['package_picked'][agent_id]:
            if self.current_state['agent_positions'][agent_id] == self.goal_room:
                return "Dropped package", self.rewards['DROP']
        return "No package to drop", 0

    def play_turn(self, action, agent_id):
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            return self.move_agent(agent_id, action)
        elif action == 'PICKUP':
            return self.pickup_package(agent_id)
        elif action == 'DROP':
            return self.drop_package(agent_id)
        else:
            return "Invalid action", 0
    
    def step(self, action, agent_id):
        action_name = action
        result, reward = self.play_turn(action_name, agent_id)
        done = False
        terminal_state = self.is_terminal(agent_id)
        if terminal_state == "DROP":
            done = True
            reward += self.rewards['DROP']
            result += f" You've dropped the package! {self.rewards['DROP']} points!"
        elif terminal_state == "EMPTY":
            done = True
            reward += self.rewards['EMPTY']
            result += f" You've run out of fuel! {self.rewards['EMPTY']} points!"

        info = {'result': result, 'action': action_name}
        return self.current_state, reward, done, info
    
    def render(self, mode='human'):
        """Renders the current state"""
        print(f"Current state: {self.current_state}")

    def close(self):
        """Performs cleanup"""
        pass
