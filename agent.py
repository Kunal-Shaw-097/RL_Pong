import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-4, batch_size=64, memory_size=20000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.update_target()

        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 20000  # Slower decay

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store(self, transition):
        self.memory.append(transition)

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states).to(self.device),
            torch.LongTensor(actions).unsqueeze(1).to(self.device),
            torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        )

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample()

        # Double DQN update
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            target_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + (1 - dones) * self.gamma * target_q

        q_values = self.policy_net(states).gather(1, actions)
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Optional: print loss every N steps
        if self.steps_done % 500 == 0:
            print(f"Step {self.steps_done}, Loss: {loss.item():.4f}")
