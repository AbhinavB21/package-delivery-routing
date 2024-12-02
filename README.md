## Overview
This project provides a visualization of a multi-agent reinforcement learning algorithm used to create an optimal package routing system for delivery drivers. It creates a random environment with obstacles, agents, and packages to simulate the route. Additionally, it provides graphs that visualize the average reward for the given number of agents over 50,000 episodes.


## Features
- **Multi Agent Q-Learning**
  - Creates a shared q-table that guides multiple agents on the optimal route.
  - Utilizes multi-threading to run agents concurrently for increased scalability.
  - Generates graphs to visualize the average reward of each agent.
  - Creates a pkl file, which can be passed on to the visualization.
 
- **Visualization**
   - Creates an easy to understand visualization that shows the agents in play.

 ## Document Overview
 - environment.py: backend of the model which defines the rules and heuristics of the system
 - q_learning.py: contains multi-agent q-learning algorithm
 - q_table.pickle: pkl file which is used for visualization
 - requirements.txt: project dependencies
 - visualize_game.py: visualization of the optimal package routing
 - charts: folder containing charts of 1, 2, and 3 agents' optimal reward over 50,000 episodes

## Getting Started

### Prerequisites
Ensure you have the following installed to run the front end application
- python 3.9+
- numpy <2
- pygame >= 2
- gym >= 0.21.0
- matplotlib >= 3.3.0
   

### Installation
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/package-delivery-routing.git
   ```
2. Navigate to the project directory:
   ```bash
   cd package-delivery-routing
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Q-Learning Model:
   ```bash
   python3 q_learning.py
   ```
5. Run the Visualization:
   ```bash
   python3 visualize_game.py
   ```

## Changing Number of Agents
Currently, the number of agents is set to 3. To change the number of agents, change the `num_agents` parameter in lines 8, 35, 133 to the new amount. To change the number of episodes being run, change the `num_episodes` parameter on line 132 to the preferred amount.

