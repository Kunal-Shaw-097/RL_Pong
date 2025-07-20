from game import PongEnv
from agent import DQNAgent
import matplotlib.pyplot as plt
import torch

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment
env = PongEnv(render_mode=False)

# Agent setup
agent = DQNAgent(state_dim=5, action_dim=3)
agent.policy_net.to(device)
agent.target_net.to(device)

# Training config
episodes = 1000
max_steps = 3000  # Limit per episode
target_update_freq = 20
reward_history = []

# Training loop
for ep in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < max_steps:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        agent.store((state, action, reward, next_state, float(done)))
        agent.train()

        state = next_state
        total_reward += reward
        steps += 1  # Count this step

    if ep % target_update_freq == 0:
        agent.update_target()

    reward_history.append(total_reward)
    print(f"Episode {ep + 1}/{episodes} | Total Reward: {total_reward}")

# Close environment
env.close()

# Save model to file (remains GPU-compatible)
torch.save(agent.policy_net.state_dict(), "dqn_pong.pth")

# Plot training reward history
plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Pong Training Progress")
plt.grid()
plt.show()
