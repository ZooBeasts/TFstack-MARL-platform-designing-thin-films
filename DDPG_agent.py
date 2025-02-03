import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_layers):
        super(Actor, self).__init__()

        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_layers, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=3
        )

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        batch_size, seq_len, _ = state.size()
        positions = torch.arange(0, seq_len, device=state.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.embedding(state) + self.position_embedding(positions)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, max_layers):
        super(Critic, self).__init__()

        self.embedding = nn.Linear(state_dim, hidden_dim)
        self.position_embedding = nn.Embedding(max_layers, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=3
        )

        self.fc1 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        batch_size, seq_len, _ = state.size()
        positions = torch.arange(0, seq_len, device=state.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.embedding(state) + self.position_embedding(positions)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling

        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, max_layers, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.max_layers = max_layers

        self.actor = Actor(state_dim, action_dim, hidden_dim, max_layers).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim, max_layers).to(self.device)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim, max_layers).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim, max_layers).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = deque(maxlen=1000000)
        self.batch_size = 64

        self.update_network_parameters(tau=1)

    @property
    def device(self):
        return torch.device('cpu')

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path)
        torch.save(self.critic.state_dict(), path + "_critic")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.critic.load_state_dict(torch.load(path + "_critic"))
        self.actor.eval()
        self.critic.eval()

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).to(self.device)

        # Ensure state has 3 dimensions: [batch_size, seq_len, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
        elif state.dim() == 2:
            state = state.unsqueeze(0)  # Add batch dimension

        action = self.actor(state).detach().cpu().numpy()[0]
        action += noise_scale * np.random.randn(self.action_dim)
        action = np.clip(action, -1, 1)

        action_type = int((action[0] + 1) / 2 * (self.action_dim - 1))
        layer_idx = int((action[1] + 1) / 2 * (self.max_layers - 1))  # Use max_layers to ensure valid index

        return action_type, layer_idx

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.memory[idx] for idx in indices])

        # Ensure tensors have 3 dimensions: [batch_size, seq_len, state_dim]
        states = torch.FloatTensor(states).to(self.device).unsqueeze(1) if len(states[0].shape) == 1 else torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device).unsqueeze(1) if len(next_states[0].shape) == 1 else torch.FloatTensor(next_states).to(self.device)

        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        return states, actions, rewards, next_states, dones

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_memory(batch_size)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = rewards + self.gamma * self.target_critic(next_states, next_actions) * (1 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_network_parameters()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)





















