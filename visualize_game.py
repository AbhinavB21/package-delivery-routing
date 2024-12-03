import pygame
import sys
import numpy as np
import pickle
import random
import time

# used some of the code from old PAs to help with the visualization
# used genai and documentation for debugging and readability
# visualization is based of .pickle (or .pkl) file that contains the Q-tables
# so don't run expecting it will show you irl simulation of q-table learning and shared q-table
# instead it will show you the visualization of the game with the shared q-table already computed
# (due to computation time and complexity of the game but works with increased grids and agents)

# Set random seed for consistency
np.random.seed(42)
random.seed(42)

# Initialize Pygame
pygame.init()

# Get screen dimensions - clutch genai here for readability for all team members
infoObject = pygame.display.Info()
SCREEN_WIDTH = infoObject.current_w
SCREEN_HEIGHT = infoObject.current_h
GRID_SIZE = 5  
TEXT_MARGIN = SCREEN_WIDTH // 4  
GRID_MARGIN = 50  
available_width = SCREEN_WIDTH - TEXT_MARGIN - 2 * GRID_MARGIN
available_height = SCREEN_HEIGHT - 2 * GRID_MARGIN
CELL_SIZE = min(available_width // GRID_SIZE, available_height // GRID_SIZE)
GRID_WIDTH = GRID_SIZE * CELL_SIZE
GRID_HEIGHT = GRID_SIZE * CELL_SIZE

# Colors on grid
WHITE = (255, 255, 255)
GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
# Pickup location
GREEN = (0, 255, 0)    
# Drop-off location while delivering
RED = (255, 0, 0)      
# Obstacles
BROWN = (139, 69, 19)  
# Packages
PURPLE = (128, 0, 128) 
AGENT_COLORS = [(0, 0, 255), (255, 165, 0), (0, 255, 0)]  

# Create full-screen window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Multi-Agent Package Delivery with Obstacles")

NUM_AGENTS = 3
NUM_OBSTACLES = 2 

class PackageEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.grid_size = GRID_SIZE
        self.obstacles = []
        self.rooms = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        self.num_packages = num_agents
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        # Rewards to match Q-learning code
        self.rewards = {
            'UP': -1,
            'DOWN': -1,
            'LEFT': -1,
            'RIGHT': -1,
            'PICK': 5000,
            'DROP': 10000,
            'EMPTY': -10000
        }

    def reset(self):
        # Initialize agent positions
        self.agent_positions = [random.choice(self.rooms) for _ in range(self.num_agents)]
        # Initialize package positions
        self.package_positions = [random.choice(self.rooms) for _ in range(self.num_packages)]
        # Initialize state
        self.current_state = {
            'agent_positions': self.agent_positions.copy(),
            'package_picked': [False] * self.num_agents
        }
        return self.current_state, 0, False

class UpdatedPackageEnv(PackageEnv):
    def __init__(self, num_agents, num_obstacles=2):
        super().__init__(num_agents)
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP']
        self.num_obstacles = num_obstacles

    def reset(self):
        """Reset the environment and define available positions"""
        available_positions = [pos for pos in self.rooms]
        total_positions_needed = self.num_agents + self.num_packages + self.num_obstacles

        # Check if we have enough positions
        if len(available_positions) < total_positions_needed + self.num_packages:
            raise ValueError("Not enough available positions to assign unique positions.")

        # Sample positions for agents, packages, and obstacles
        all_positions = random.sample(available_positions, total_positions_needed)
        self.agent_positions = all_positions[:self.num_agents]
        self.package_positions = all_positions[self.num_agents:self.num_agents + self.num_packages]
        self.obstacles = all_positions[self.num_agents + self.num_packages:]

        # Collect all occupied positions
        occupied_positions = set(self.package_positions + self.agent_positions + self.obstacles)

        # Filter available positions for drop-off locations
        available_positions = [pos for pos in self.rooms if pos not in occupied_positions]

        # Check if we have enough positions for drop-off locations
        if len(available_positions) < self.num_packages:
            raise ValueError("Not enough available positions to assign unique drop-off locations.")

        # Assign drop-off positions
        self.drop_off_positions = random.sample(available_positions, self.num_packages)

        # Initialize package and agent states
        self.package_delivered = [False] * self.num_packages
        self.package_picked = [False] * self.num_packages
        self.current_state = {
            'agent_positions': self.agent_positions.copy(),
            'package_picked': [False] * self.num_agents
        }

        return self.current_state, 0, False

    def step(self, action, agent_id):
        """Performs the given action for the agent and updates the environment"""
        reward = 0
        done = False
        info = {}

        # Get current position
        x, y = self.agent_positions[agent_id]

        # Move agent if action is movement
        if action == 'UP':
            new_x, new_y = x - 1, y
            reward += self.rewards['UP']
        elif action == 'DOWN':
            new_x, new_y = x + 1, y
            reward += self.rewards['DOWN']
        elif action == 'LEFT':
            new_x, new_y = x, y - 1
            reward += self.rewards['LEFT']
        elif action == 'RIGHT':
            new_x, new_y = x, y + 1
            reward += self.rewards['RIGHT']
        elif action == 'PICK':
            # Try to pick up a package
            msg, r = self.pick_package(agent_id)
            reward += r
            info['message'] = msg
            # Stalling
            new_x, new_y = x, y 
        elif action == 'DROP':
            # Try to drop the package
            msg, r = self.drop_package(agent_id)
            reward += r
            info['message'] = msg
            # Stalling
            new_x, new_y = x, y  
        else:
            # Invalid action, no movement
            new_x, new_y = x, y  

        # Check if new position is valid and update position
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            if (0 <= new_x < self.grid_size) and (0 <= new_y < self.grid_size):
                if (new_x, new_y) not in self.obstacles:
                    self.agent_positions[agent_id] = (new_x, new_y)
                else:
                    info['message'] = "Hit an obstacle!"
            else:
                pass

        # Update current state
        self.current_state['agent_positions'][agent_id] = self.agent_positions[agent_id]

        return self.current_state, reward, done, info

    def pick_package(self, agent_id):
        """Agent picks up a package if at the correct location and not already carrying one"""
        if self.current_state['package_picked'][agent_id]:
            return "Already carrying a package", 0
        agent_position = self.current_state['agent_positions'][agent_id]
        for idx, package_position in enumerate(self.package_positions):
            if agent_position == package_position and not self.package_picked[idx]:
                self.package_picked[idx] = True
                self.current_state['package_picked'][agent_id] = True
                return "Picked up package", self.rewards['PICK']
        return "No package to pick up", 0

    def drop_package(self, agent_id):
        """Drop package only if at the correct drop-off location"""
        if self.current_state['package_picked'][agent_id]:
            agent_position = self.current_state['agent_positions'][agent_id]
            for idx, drop_position in enumerate(self.drop_off_positions):
                if agent_position == drop_position and not self.package_delivered[idx]:
                    self.package_delivered[idx] = True
                    self.current_state['package_picked'][agent_id] = False
                    return "Dropped package", self.rewards['DROP']
        return "No package to drop", 0

env = UpdatedPackageEnv(num_agents=NUM_AGENTS, num_obstacles=NUM_OBSTACLES)

# Load Q-tables
with open('q_table.pickle', 'rb') as f:
    Q_tables = pickle.load(f)

# Reset environment
current_state, total_reward, done = env.reset()

# Initialize fonts - genai came in clutch here for readability
font_size = SCREEN_HEIGHT // 40 
font = pygame.font.SysFont('Consolas', font_size)  
font_bold = pygame.font.SysFont('Consolas', font_size, bold=True)

def draw_grid():
    """Draws grid lines"""
    for x in range(GRID_MARGIN, GRID_MARGIN + GRID_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, GRID_MARGIN), (x, GRID_MARGIN + GRID_HEIGHT))
    for y in range(GRID_MARGIN, GRID_MARGIN + GRID_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (GRID_MARGIN, y), (GRID_MARGIN + GRID_WIDTH, y))

def draw_obstacles():
    """Draws obstacles"""
    for obstacle in env.obstacles:
        x, y = obstacle
        rect = pygame.Rect(GRID_MARGIN + y * CELL_SIZE, GRID_MARGIN + x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, BROWN, rect)

def draw_packages():
    """Draws packages, pickup, and drop-off locations"""
    for idx, package_position in enumerate(env.package_positions):
        drop_x, drop_y = env.drop_off_positions[idx]

        # Draw pickup location only if the package has not been picked up
        if not env.package_picked[idx]:
            x, y = package_position
            rect = pygame.Rect(
                GRID_MARGIN + y * CELL_SIZE + CELL_SIZE // 4,
                GRID_MARGIN + x * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2,
                CELL_SIZE // 2
            )
            # Package rectangle
            pygame.draw.rect(screen, PURPLE, rect)  
            pickup_rect = pygame.Rect(
                GRID_MARGIN + y * CELL_SIZE,
                GRID_MARGIN + x * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            # Green border for pickup location
            pygame.draw.rect(screen, GREEN, pickup_rect, 2)  
        else:
            # Clear the pickup spot to normal grid
            x, y = package_position
            pickup_rect = pygame.Rect(
                GRID_MARGIN + y * CELL_SIZE,
                GRID_MARGIN + x * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE
            )
            pygame.draw.rect(screen, WHITE, pickup_rect)
            pygame.draw.rect(screen, GRAY, pickup_rect, 1)

        # Draw drop-off location
        drop_rect = pygame.Rect(
            GRID_MARGIN + drop_y * CELL_SIZE,
            GRID_MARGIN + drop_x * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        if env.package_delivered[idx]:
            pygame.draw.rect(screen, WHITE, drop_rect)
            drop_mark_rect = pygame.Rect(
                GRID_MARGIN + drop_y * CELL_SIZE + CELL_SIZE // 4,
                GRID_MARGIN + drop_x * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2,
                CELL_SIZE // 2
            )
            pygame.draw.rect(screen, RED, drop_mark_rect)
        else:
            pygame.draw.rect(screen, RED, drop_rect, 2)

def draw_agents():
    """Draws agents and their carried packages"""
    for idx, agent_position in enumerate(env.agent_positions):
        x, y = agent_position
        rect = pygame.Rect(
            GRID_MARGIN + y * CELL_SIZE + CELL_SIZE // 6,
            GRID_MARGIN + x * CELL_SIZE + CELL_SIZE // 6,
            CELL_SIZE * 2 // 3,
            CELL_SIZE * 2 // 3
        )
        color = AGENT_COLORS[idx % len(AGENT_COLORS)]
        pygame.draw.ellipse(screen, color, rect)
        if env.current_state['package_picked'][idx]:
            # Draw package as part of the agent
            pygame.draw.circle(
                screen,
                PURPLE,
                (
                    GRID_MARGIN + y * CELL_SIZE + CELL_SIZE // 2,
                    GRID_MARGIN + x * CELL_SIZE + CELL_SIZE // 2
                ),
                CELL_SIZE // 6
            )

def draw_hud():
    """Draws the HUD with rewards and remaining packages"""
    pygame.draw.rect(
        screen,
        WHITE,
        (GRID_MARGIN + GRID_WIDTH, 0, SCREEN_WIDTH - GRID_WIDTH - GRID_MARGIN, SCREEN_HEIGHT)
    )

    # All text info + position for the text
    hud_x = GRID_MARGIN + GRID_WIDTH + 20
    hud_y = GRID_MARGIN
    line_height = font_size + 10  
    title_text = font_bold.render("Simulation Status", True, BLACK)
    screen.blit(title_text, (hud_x, hud_y))
    hud_y += line_height + 10 

    # Total Reward
    reward_text = font.render(f"Total Reward: {total_reward}", True, BLACK)
    screen.blit(reward_text, (hud_x, hud_y))
    hud_y += line_height

    # Remaining Packages
    remaining_packages = sum(1 for d in env.package_delivered if not d)
    remaining_text = font.render(f"Remaining Packages: {remaining_packages}", True, BLACK)
    screen.blit(remaining_text, (hud_x, hud_y))
    hud_y += line_height

    # Just margin line for readability
    pygame.draw.line(
        screen,
        GRAY,
        (hud_x, hud_y),
        (SCREEN_WIDTH - GRID_MARGIN - 20, hud_y),
        1
    )
    hud_y += 10

    # Package Status
    package_status_title = font_bold.render("Package Status:", True, BLACK)
    screen.blit(package_status_title, (hud_x, hud_y))
    hud_y += line_height

    # Display status for each package
    for idx, delivered in enumerate(env.package_delivered):
        status = "Delivered" if delivered else "In Transit"
        package_text = font.render(f"Package {idx + 1}: {status}", True, BLACK)
        screen.blit(package_text, (hud_x + 20, hud_y))
        hud_y += line_height

    # Just another margin line for readability
    hud_y += 10
    pygame.draw.line(
        screen,
        GRAY,
        (hud_x, hud_y),
        (SCREEN_WIDTH - GRID_MARGIN - 20, hud_y),
        1
    )
    hud_y += 10

    # Agent Status
    agent_status_title = font_bold.render("Agent Status:", True, BLACK)
    screen.blit(agent_status_title, (hud_x, hud_y))
    hud_y += line_height

    # Display status for each agent
    for idx in range(NUM_AGENTS):
        carrying = "Yes" if env.current_state['package_picked'][idx] else "No"
        agent_text = font.render(f"Agent {idx + 1} Carrying Package: {carrying}", True, BLACK)
        screen.blit(agent_text, (hud_x + 20, hud_y))
        hud_y += line_height

def get_action_from_Q(agent_id, observation):
    """Gets an action from the Q-table or chooses randomly if the state is unseen"""
    state_hash = str(observation)
    if state_hash in Q_tables[agent_id]:
        action_values = Q_tables[agent_id][state_hash]
        return np.argmax(action_values)
    return np.random.choice(len(env.actions))

# Main loop
running = True
clock = pygame.time.Clock()
total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                pygame.quit()
                sys.exit()

    # Agent actions
    actions = {}
    for agent_id in range(NUM_AGENTS):
        observation = {
            'position': env.agent_positions[agent_id],
            'carrying_package': env.current_state['package_picked'][agent_id]
        }
        action_idx = get_action_from_Q(agent_id, observation)
        actions[agent_id] = env.actions[action_idx]

    # Step the environment
    for agent_id, action in actions.items():
        current_state, reward, done, info = env.step(action, agent_id)
        total_reward += reward

    # Clear screen
    screen.fill(WHITE)

    # Draw environment
    draw_grid()
    draw_obstacles()
    draw_packages()
    draw_agents()
    draw_hud()

    # Update display
    pygame.display.flip()

    # After drawing, check if all packages are delivered
    if all(env.package_delivered):
        draw_hud()
        pygame.display.flip()
        print("All packages delivered. Resetting environment.")
        time.sleep(2)
        current_state, total_reward, done = env.reset()
        # Reset total reward
        total_reward = 0  

    clock.tick(20) 
