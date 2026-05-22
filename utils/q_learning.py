import numpy as np
import random

def train_smart_traffic_agent(episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """
    Simulates a Reinforcement Learning agent (Q-Learning) optimizing traffic lights
    to reduce urban noise pollution.
    
    States (4): 0=Low, 1=Moderate, 2=High, 3=Critical
    Actions (2): 0=Keep Lights (Maintain flow), 1=Switch Lights (Interrupt/Relieve flow)
    """
    # Initialize Q-Table with zeros
    q_table = np.zeros((4, 2))
    rewards_all_episodes = []

    for episode in range(episodes):
        state = random.randint(0, 3) # Start in a random noise state
        total_reward = 0
        
        # 10 steps (simulated time intervals) per episode
        for step in range(10): 
            # Exploration vs Exploitation trade-off
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, 1) # Explore
            else:
                action = np.argmax(q_table[state, :]) # Exploit

            # Environment Logic (Rewards & Transitions)
            if state == 3: # Critical Noise
                if action == 1: # Switch Lights (Good move, relieves congestion)
                    reward = 10
                    next_state = random.choice([1, 2]) # Noise drops
                else: # Keep Lights (Bad move, congestion builds up)
                    reward = -10
                    next_state = 3 
                    
            elif state == 2: # High Noise
                if action == 1:
                    reward = 5
                    next_state = random.choice([0, 1])
                else:
                    reward = -5
                    next_state = 3
                    
            elif state == 1: # Moderate Noise
                if action == 1:
                    reward = -2 # Unnecessary switch causes slight delay
                    next_state = random.choice([1, 2])
                else:
                    reward = 5
                    next_state = random.choice([0, 1])
                    
            else: # Low Noise
                if action == 1:
                    reward = -5 # Disrupting good flow causes noise spikes
                    next_state = 1
                else:
                    reward = 10 # Excellent traffic flow
                    next_state = 0

            # Update Q-Table using the Bellman Equation
            q_table[state, action] = q_table[state, action] * (1 - alpha) + \
                alpha * (reward + gamma * np.max(q_table[next_state, :]))

            total_reward += reward
            state = next_state

        rewards_all_episodes.append(total_reward)

    # Smooth the rewards array for better visualization in Plotly (batches of 50 episodes)
    batch_size = 50
    smoothed_rewards = [np.mean(rewards_all_episodes[i:i+batch_size]) 
                        for i in range(0, len(rewards_all_episodes), batch_size)]

    return {
        "q_table": q_table.tolist(),
        "smoothed_rewards": smoothed_rewards,
        "final_performance": int(np.mean(rewards_all_episodes[-100:]))
    }