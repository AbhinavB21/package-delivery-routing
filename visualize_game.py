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

# Add these new colors
AGENT_COLORS = [
    (0, 255, 0),     # Green
    (0, 200, 100),   # Teal
    (100, 255, 100)  # Light green
]
AGENT_WITH_PACKAGE_COLORS = [
    (0, 150, 0),     # Dark green
    (0, 100, 50),    # Dark teal
    (50, 150, 50)    # Dark light green
]

# initialize pygame
pygame.init()

# draw the grid
def draw_grid(screen, state, env):
    screen.fill(WHITE)
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            
            # Draw obstacles
            if (x, y) in env.obstacles:
                pygame.draw.rect(screen, RED, rect)  # Fill obstacle cells with red
            
            # Draw goal room with transparency
            if (x, y) == env.goal_room:
                s = pygame.Surface((CELL_SIZE, CELL_SIZE))
                s.set_alpha(128)
                s.fill((255, 215, 0))
                screen.blit(s, (x * CELL_SIZE, y * CELL_SIZE))
            
            # Draw packages with circles
            for package_pos in state['package_positions']:
                if (x, y) == package_pos:
                    center = (x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2)
                    pygame.draw.circle(screen, BLUE, center, CELL_SIZE//3)
            
            # Draw agents with different colors and shapes
            for i, agent_pos in enumerate(state['agent_positions']):
                if (x, y) == agent_pos:
                    agent_rect = pygame.Rect(
                        x * CELL_SIZE + 5, 
                        y * CELL_SIZE + 5, 
                        CELL_SIZE - 10, 
                        CELL_SIZE - 10
                    )
                    color = AGENT_WITH_PACKAGE_COLORS[i] if state['package_picked'][i] else AGENT_COLORS[i]
                    pygame.draw.rect(screen, color, agent_rect)
                    # Add agent number
                    font = pygame.font.Font(None, 36)
                    text = font.render(str(i+1), True, BLACK)
                    text_rect = text.get_rect(center=(x * CELL_SIZE + CELL_SIZE//2, y * CELL_SIZE + CELL_SIZE//2))
                    screen.blit(text, text_rect)

# main loop
def run_visualization(env, q_table):
    running = True
    clock = pygame.time.Clock()
    screen_size = env.grid_size * CELL_SIZE
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Multi-Agent Package Delivery Simulation")

    state, _, _ = env.reset()
    done = False
    
    FPS = 2
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:
            actions = []
            for agent_id in range(env.num_agents):
                state_hash = hash_state(state, agent_id)
                
                if agent_id in q_table and state_hash in q_table[agent_id]:
                    q_values = q_table[agent_id][state_hash]
                    agent_action = np.argmax(q_values)
                    actions.append(agent_action)
                else:
                    actions.append(0)
            
            # Take actions for each agent
            for agent_id, action in enumerate(actions):
                next_state, reward, done, info = env.step(action, agent_id)
                if done:
                    break
            
            state = next_state

        draw_grid(screen, state, env)
        pygame.display.flip()
        clock.tick(FPS)

# run viz with env from environment.py
if __name__ == "__main__":
    # Load Q-table
    with open('q_table.pickle', 'rb') as f:
        q_table = pickle.load(f)
    
    # Create environment with same parameters used during training
    env = PackageEnv(num_agents=3, num_packages=3)
    
    # Run visualization
    run_visualization(env, q_table)