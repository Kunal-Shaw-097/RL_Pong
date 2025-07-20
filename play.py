from game import PongEnv
from agent import QNetwork
import torch
import time

# Setup
env = PongEnv(render_mode=True)
state_dim = 5
action_dim = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = QNetwork(state_dim, action_dim).to(device)
model.load_state_dict(torch.load("dqn_pong.pth", map_location=device))
model.eval()

# Run one episode
state = env.reset()
done = False
total_reward = 0

while not done:
    env.render()
    time.sleep(1 / env.FPS)  # slow down to human speed (optional)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = model(state_tensor).argmax().item()

    state, reward, done = env.step(action)
    total_reward += reward

env.close()
print("Episode Reward:", total_reward)
