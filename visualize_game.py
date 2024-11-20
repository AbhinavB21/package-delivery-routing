import pygame
import numpy as np
from environment import PackageEnv
import pickle
import time
from q_learning import hash_state

# size of each cell in pixels
CELL_SIZE = 50 

# Updated Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)  # For packages
GREEN = (0, 0, 255)  # For agents
RED = (255, 0, 0)  # For obstacles 

# initialize pygame
pygame.init()

# draw the grid
def draw_grid(screen, state, env):
    screen.fill(WHITE)
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            # Draw goal room
            if (x, y) == env.goal_room:
                pygame.draw.rect(screen, (255, 215, 0), rect)  # Gold color for goal
            
            # Draw agents (plural)
            for i, agent_pos in enumerate(state['agent_positions']):
                if (x, y) == agent_pos:
                    color = GREEN if state['package_picked'][i] else (0, 255, 0)
                    pygame.draw.rect(screen, color, rect)
            
            # Draw packages (plural)
            for package_pos in state['package_positions']:
                if (x, y) == package_pos:
                    pygame.draw.rect(screen, BLUE, rect)

# main loop
def run_visualization(env, q_table):
    running = True
    clock = pygame.time.Clock()
    screen_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Multi-Agent Package Delivery Simulation")

    state = env.reset()
    done = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:
            # Get actions for all agents
            actions = []
            for agent_id in range(env.num_agents):
                state_hash = hash_state(state, agent_id)
                if state_hash in q_table:
                    agent_action = np.argmax(q_table[state_hash])
                    actions.append(agent_action)
                else:
                    actions.append(0)  # Default action if state not in q_table
            
            # Take the combined action
            state, reward, done = env.step(actions)

        draw_grid(screen, state, env)
        pygame.display.flip()
        time.sleep(0.5)  # Add delay to make visualization easier to follow
        
        if done:
            time.sleep(2)  # Pause at the end
            running = False

    pygame.quit()

# run viz with env from environment.py
if __name__ == "__main__":
    # Load Q-table
    with open('q_table.pickle', 'rb') as f:
        q_table = pickle.load(f)
    
    # Create environment with same parameters used during training
    env = PackageEnv(num_agents=3, num_packages=3)
    
    # Run visualization
    run_visualization(env, q_table)