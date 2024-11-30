import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from environment import PackageEnv
import pickle
import matplotlib.pyplot as plt

env = PackageEnv(num_agents=2, num_packages=2)

def hash_state(state, agent_id):
    x, y = state['agent_positions'][agent_id]
    package_picked = int(state['package_picked'][agent_id])
    px, py = state['package_positions'][agent_id]
    gx, gy = env.goal_room

    return (x * env.grid_size * 2 * env.grid_size * env.grid_size +
            y * 2 * env.grid_size * env.grid_size +
            package_picked * env.grid_size * env.grid_size +
            px * env.grid_size + py) * env.grid_size + gx * env.grid_size + gy

	
def Q_learning(agent_id, Q_table, num_updates, epsilon, gamma):
    state, _, _ = env.reset()
    prev_state = hash_state(state, agent_id)
    done = False
    total_reward = 0
    actions = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PICKUP", 5: "DROP"}

    while not done:
        if prev_state not in Q_table[agent_id]:
            Q_table[agent_id][prev_state] = np.zeros(len(env.actions))
            num_updates[agent_id][prev_state] = np.zeros(len(env.actions))

        if random.random() < epsilon:
            action = random.randrange(len(env.actions))
        else:
            action = np.argmax(Q_table[agent_id][prev_state] + 
                             np.random.random(len(env.actions)) * 1e-6)
        
        next_state, reward, done, _ = env.step(actions[action], agent_id)
        curr_state = hash_state(next_state, agent_id)
        total_reward += reward

        if curr_state not in Q_table[agent_id]:
            Q_table[agent_id][curr_state] = np.zeros(len(env.actions))
            num_updates[agent_id][curr_state] = np.zeros(len(env.actions))

        eta = 1.0 / (1.0 + num_updates[agent_id][prev_state][action])
        next_max = np.max(Q_table[agent_id][curr_state])
        
        shared_q_value = 0
        if len(Q_table) > 1:
            shared_values = []
            for agent in range(len(Q_table)):
                if agent != agent_id and curr_state in Q_table[agent]:
                    shared_values.append(Q_table[agent][curr_state][action])
            if shared_values:
                shared_q_value = sum(shared_values) / len(shared_values)

        Q_table[agent_id][prev_state][action] = ((1 - eta) * Q_table[agent_id][prev_state][action] +
                                              eta * (reward + gamma * (0.5 * next_max + 0.5 * shared_q_value)))
        
        num_updates[agent_id][prev_state][action] += 1
        prev_state = curr_state
        state = next_state
    
    return Q_table[agent_id], total_reward

def q_learning_multi_agent(num_episodes, num_agents, gamma=0.9, epsilon=1.0, decay_rate=0.999):
    Q_table = [{} for _ in range(num_agents)]
    num_updates = [{} for _ in range(num_agents)]
    
    agent_rewards = [[] for _ in range(num_agents)]
    episode_numbers = []
    
    # For calculating moving average
    window_size = 50
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"Episode {episode}")
            # Only append episode number when we're actually recording rewards
            episode_numbers.append(episode)
        
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [
                executor.submit(Q_learning, agent_id, Q_table, num_updates, epsilon, gamma)
                for agent_id in range(num_agents)
            ]
        
        # Record rewards more frequently
        for agent_id, future in enumerate(futures):
            updated_q_table, reward = future.result()
            Q_table[agent_id].update(updated_q_table)
            if episode % 10 == 0:  # More frequent reward recording
                agent_rewards[agent_id].append(reward)
        
        if episode % 10 == 0:
            avg_reward = sum(reward for _, reward in [f.result() for f in futures]) / len(futures)
            print(f"Average reward: {avg_reward}")
        
        epsilon = max(0.1, epsilon * decay_rate)
    
    plt.figure(figsize=(10, 6))
    for agent_id in range(num_agents):
        rewards = agent_rewards[agent_id]
        # Calculate moving average
        if len(rewards) > window_size:
            moving_avg = [np.mean(rewards[max(0, i-window_size):i]) for i in range(1, len(rewards)+1)]
        else:
            moving_avg = rewards
            
        # Ensure episode_numbers and moving_avg have the same length
        plt.plot(episode_numbers[:len(moving_avg)], moving_avg, label=f'Agent {agent_id + 1}')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('agent_rewards.png')
    plt.close()
    print(Q_table)
    return Q_table

if __name__ == "__main__":
    Q_table = q_learning_multi_agent(
        num_episodes=1000,
        num_agents=2,
        gamma=0.99,
        epsilon=1.0,
        decay_rate=0.9995
    )
    
    with open('q_table.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)