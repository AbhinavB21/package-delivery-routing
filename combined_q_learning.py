import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
from environment import PackageEnv
import pickle
import matplotlib.pyplot as plt

env = PackageEnv(num_agents=3)

def hash_state(state, agent_id):
    x, y = state['agent_positions'][agent_id]
    package_picked = int(state['package_picked'][agent_id])
    px, py = state['package_positions'][agent_id]
    gx, gy = env.goal_room

    return (x * env.grid_size * 2 * env.grid_size * env.grid_size +
            y * 2 * env.grid_size * env.grid_size +
            package_picked * env.grid_size * env.grid_size +
            px * env.grid_size + py) * env.grid_size + gx * env.grid_size + gy #ChatGPT helped with this formula

def calculate_eta(num_updates, state, action):
    base_learning_rate = 0.2
    return base_learning_rate / (1 + num_updates[state][action] * 0.1)

def calculate_shared_q_value(Q_table, state, action, num_agents):
    shared_q = 0
    count = 0
    for agent_id in range(num_agents):
        if state in Q_table[agent_id] and len(Q_table[agent_id][state]) > action:
            shared_q += Q_table[agent_id][state][action]
            count += 1
    return shared_q / count if count > 0 else 0 #ChatGPT helped with this

def Q_learning(agent_id, Q_table, num_updates, epsilon, gamma, learning_rate):
    q_learning_env = PackageEnv(num_agents=3)
    state, _, _ = q_learning_env.reset()
    prev_state = hash_state(state, agent_id)
    done = False
    total_reward = 0
    actions = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICKUP', 'DROP']

    while not done:
        if prev_state not in Q_table[agent_id]:
            Q_table[agent_id][prev_state] = np.ones(len(actions)) * 1000
            num_updates[agent_id][prev_state] = np.zeros(len(actions))

        if random.random() < epsilon:
            action_idx = random.randrange(len(actions))
        else:
            Q_values = Q_table[agent_id][prev_state]
            action_idx = np.argmax(Q_values)

        next_state, reward, done, _ = q_learning_env.step(actions[action_idx], agent_id)
        curr_state = hash_state(next_state, agent_id)
        
        total_reward += reward

        if curr_state not in Q_table[agent_id]:
            Q_table[agent_id][curr_state] = np.ones(6) * 1000
            num_updates[agent_id][curr_state] = np.zeros(6)

        next_best_action = np.argmax(Q_table[agent_id][curr_state])
        eta = calculate_eta(num_updates[agent_id], prev_state, action_idx)
        shared_q = calculate_shared_q_value(Q_table, prev_state, action_idx, q_learning_env.num_agents)
        alpha = 0.5 
        target = reward + (0 if done else gamma * ((1-alpha) * Q_table[agent_id][curr_state][next_best_action] + 
                                                 alpha * shared_q)) #ChatGPT helped with this line
        current_q = Q_table[agent_id][prev_state][action_idx]
        Q_table[agent_id][prev_state][action_idx] = current_q + eta * (target - current_q)
        
        num_updates[agent_id][prev_state][action_idx] += 1
        prev_state = curr_state
        state = next_state

    return Q_table[agent_id], total_reward

def plot_rewards(agent_rewards, episode_numbers, window_size=50): #ChatGPT helped with this
    plt.figure(figsize=(10, 6))
    for agent_id, rewards in enumerate(agent_rewards):
        if len(rewards) > window_size:
            moving_avg = [np.mean(rewards[max(0, i-window_size):i]) 
                        for i in range(1, len(rewards)+1)]
        else:
            moving_avg = rewards
        plt.plot(episode_numbers[:len(moving_avg)], moving_avg, label=f'Agent {agent_id + 1}')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Agent Rewards over Episodes')
    plt.legend()
    plt.grid(True)
    plt.savefig('agent_rewards.png')
    plt.close()

def q_learning_multi_agent(num_episodes, num_agents, gamma=0.99, epsilon=1.0, 
                          decay_rate=0.9995, learning_rate=0.1):
    Q_table = [{} for _ in range(num_agents)]
    num_updates = [{} for _ in range(num_agents)]
    agent_rewards = [[] for _ in range(num_agents)]
    episode_numbers = []
    
    for episode in range(num_episodes):
        if episode % 10 == 0:
            print(f"Episode {episode}")
            episode_numbers.append(episode)
            
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [
                executor.submit(Q_learning, agent_id, Q_table, num_updates, 
                              epsilon, gamma, learning_rate)
                for agent_id in range(num_agents)
            ] #ChatGPT helped with this

        for agent_id, future in enumerate(futures):
            updated_q_table, reward = future.result()
            Q_table[agent_id].update(updated_q_table)
            if episode % 10 == 0:
                agent_rewards[agent_id].append(reward)

        if episode % 10 == 0:
            avg_reward = sum(reward for _, reward in [f.result() for f in futures]) / len(futures) #ChatGPT helped with this
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.3f}")

        epsilon = max(0.25, epsilon * decay_rate)

    plot_rewards(agent_rewards, episode_numbers)
    return Q_table

if __name__ == "__main__":
    Q_table = q_learning_multi_agent(
        num_episodes=10000,
        num_agents=3,
        gamma=0.99,
        epsilon=1.0,
        decay_rate=0.9995,
        learning_rate=0.1
    )

    with open('q_table.pickle', 'wb') as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL) 