import pygame
import numpy as np
from environment import Environment

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
def draw_grid(grid, grid_size):
    screen.fill(WHITE)
    for x in range(grid_size):
        for y in range(grid_size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if grid[x][y] == "O":  # Obstacle
                pygame.draw.rect(screen, RED, rect)
            elif grid[x][y] == "P":  # Package
                pygame.draw.rect(screen, BLUE, rect)
            elif isinstance(grid[x][y], str) and grid[x][y].startswith("A"):  # Agent
                pygame.draw.rect(screen, GREEN, rect)

def update_environment(env):
    # Reset the grid and place agents, packages, and obstacles
    grid = np.zeros((env.grid_size, env.grid_size), dtype=object)
    
    # place obstacles, packages, and agents on the grid
    for obstacle in env.obstacles:
        grid[obstacle[0]][obstacle[1]] = "O"
    
    for package in env.packages:
        grid[package.position[0]][package.position[1]] = "P"

    for agent in env.agents:
        grid[agent.pos[0]][agent.pos[1]] = f"A{agent.id}" 
    
    return grid

# main loop
def run_visualization(env):
    running = True
    clock = pygame.time.Clock()
    screen_size = env.grid_size * CELL_SIZE
    global screen
    screen = pygame.display.set_mode((screen_size, screen_size))
    pygame.display.set_caption("Package Delivery Simulation")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # update the environment (for now static, but can be dynamic with agent moves)
        grid = update_environment(env)
        # draw the updated grid
        draw_grid(grid, env.grid_size)
    
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# run viz with env from environment.py
if __name__ == "__main__":
    # adjust num_agents, grid_size, num_obstacles here
    game = Environment(num_agents=2, grid_size=5, num_obstacles=2)
    run_visualization(game)