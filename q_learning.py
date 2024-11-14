import numpy as np
from visualize_game import *
from environment import *
import random
from concurrent.futures import ThreadPoolExecutor

env = Environment(3, 5, 1)

def hash(obs):
    x, y = obs['driver_position']
    package_picked = int(obs['package_picked'])
    px, py = env.pickup_location
    dx, dy = env.drop_location
    

    return (x * env.grid_size * 2 * env.grid_size * env.grid_size +
            y * 2 * env.grid_size * env.grid_size +
            package_picked * env.grid_size * env.grid_size +
            px * env.grid_size + py) * env.grid_size + dx * env.grid_size + dy #Chatgpt helped me with this



def Q_learning(agent_id, Q_table, num_updates, epsilon, gamma):
    observation, reward, done = env.reset()
    prev_state = hash(observation)
    
    while not done:
            
        if prev_state not in Q_table[agent_id]:
            Q_table[agent_id][prev_state] = np.zeros(6)

        if prev_state not in num_updates[agent_id]:
            num_updates[agent_id][prev_state] = np.zeros(6)

        random_val = random.random()
        if random_val < epsilon:
            action = random.randint(0, 5)
        else:
            action = np.argmax(Q_table[agent_id][prev_state])
        
        observation, reward, done = env.step(env.int_to_actions[action])
        curr_state = hash(observation)

        if curr_state not in Q_table[agent_id]:
            Q_table[agent_id][curr_state] = np.zeros(6)
        
        eta = 1 / (1 + num_updates[agent_id][prev_state][action])

        curr_value_opt = max(Q_table[agent_id].get(curr_state, np.zeros(6)))

        shared_q_value = 0
        if len(Q_table) > 1:
            shared_values = []
            for agent in range(len(Q_table)):
                if agent != agent_id and curr_state in Q_table[agent]:
                    shared_values.append(Q_table[agent][curr_state][action])
            if shared_values:
                shared_q_value = sum(shared_values) / len(shared_values)

        Q_table[agent_id][prev_state][action] = ((1 - eta) * Q_table[agent_id][prev_state][action] +
                                              eta * (reward + gamma * (0.5 * curr_value_opt + 0.5 * shared_q_value)))
        
        num_updates[agent_id][prev_state][action] += 1

        prev_state = curr_state
    
    return Q_table[agent_id]


def q_learning_multi_agent(num_episodes, num_agents, gamma, epsilon, decay_rate):
    Q_table = []
    num_updates = []
    for i in range(num_agents):
        Q_table.append({})
        num_updates.append({})
    
    for episode in range(num_episodes):
        if episode % 100 == 0:
            print(f"Episode {episode}")
        
        with ThreadPoolExecutor(max_workers=num_agents) as executor: #Chatgpt helped me with this
            futures = [
                executor.submit(Q_learning, agent_id, Q_table, num_updates, epsilon, gamma)
                for agent_id in range(num_agents)
            ]
        
        for agent_id, future in enumerate(futures):
            Q_table[agent_id].update(future.result())
        
        epsilon = max(0.1, epsilon * decay_rate)
    
    return Q_table

q_learning_multi_agent(1000, env.num_agents, 0.9, 1, 0.999)
