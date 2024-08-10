
import torch
from torch import optim
from torch.nn import functional as F
import numpy as np
import torch.nn as nn




class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, lr=0.003, memory_size=1000,
                 gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = TransformerModel(state_dim, hidden_dim, nhead=8, num_encoder_layers=3,
                                       max_len=state_dim // 2, action_dim=action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = []
        self.memory_size = memory_size
        self.last_policy_loss = None
        self.last_value_loss = None
        self.mode = 'explore'  # Default to exploration mode

    @property
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()


    def switch_mode(self, mode):
        if mode not in ['explore', 'optimize']:
            raise ValueError("Mode should be 'explore' or 'optimize'.")
        self.mode = mode

    def act(self, state, current_layers, min_layers, num_materials):
        self.policy.eval()
        with torch.no_grad():
            state = self.process_state(state)
            if state.dim() != 3:
                raise ValueError(f"Expected input tensor to have 3 dimensions, but got {state.dim()} dimensions.")
            action_logits, layer_indices, parameter_values = self.policy.act(state)

            actions = []
            for i in range(action_logits.shape[0]):
                action_idx = torch.argmax(action_logits[i]).item()
                layer_index_value = layer_indices[i].item()

                if action_idx == num_materials + 2:  # 'remove' action
                    if len(current_layers) <= min_layers:
                        continue  # Skip 'remove' action if less than or equal to minimum layers

                actions.append((action_idx, layer_index_value))

            if not actions:  # Ensure there's at least one valid action
                actions.append((num_materials, 0))  # Fallback to a default action (e.g., 'done' action)

            return actions

    def process_state(self, state):
        # Ensure state has 3 dimensions and correct shape
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        elif state.dim() == 2:
            state = state.unsqueeze(0)
        return state.to(self.device)

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)  # Remove the oldest experience
        self.memory.append((state, action, reward, next_state, done))

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        rewards = rewards.squeeze()
        values = values.squeeze()
        next_values = next_values.squeeze()
        dones = dones.squeeze()

        # Ensure rewards, values, next_values, and dones are 1D tensors
        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if next_values.dim() == 0:
            next_values = next_values.unsqueeze(0)
        if dones.dim() == 0:
            dones = dones.unsqueeze(0)

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * gae * (1 - dones[step])
            advantages.insert(0, gae)
        return advantages

    def prune_memory(self):
        self.memory = [item for item in self.memory if item[2] > -10]

    def replay(self):
        if len(self.memory) == 0:
            return

        states, actions, rewards, next_states, dones = zip(*self.memory)
        self.memory = []

        states = torch.cat([self.process_state(state) for state in states]).to(self.device)
        next_states = torch.cat([self.process_state(next_state) for next_state in next_states]).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.device)

        values = self.policy.evaluate(states)[-1].detach().squeeze()
        next_values = self.policy.evaluate(next_states)[-1].detach().squeeze()

        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if next_values.dim() == 0:
            next_values = next_values.unsqueeze(0)
        if dones.dim() == 0:
            dones = dones.unsqueeze(0)

        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.tensor(advantages, dtype=torch.float).to(self.device)
        returns = advantages + values

        policy_loss = []
        value_loss = []

        for _ in range(self.K_epochs):
            values = self.policy.evaluate(states)[-1].squeeze()
            log_probs, _, _, _ = self.policy.evaluate(states)

            max_action_dim = log_probs.size(1)
            valid_actions = actions[:, 0].clamp(0, max_action_dim - 1)
            log_probs = log_probs.gather(1, valid_actions[:, None]).squeeze()
            old_log_probs = log_probs.detach()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            if values.dim() == 0:
                values = values.unsqueeze(0)
            if returns.dim() == 0:
                returns = returns.unsqueeze(0)

            critic_loss = F.mse_loss(values, returns)
            loss = 0.5 * critic_loss + actor_loss

            policy_loss.append(actor_loss.item())
            value_loss.append(critic_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.last_policy_loss = np.mean(policy_loss)
        self.last_value_loss = np.mean(value_loss)

    def update(self):
        self.prune_memory()
        self.replay()



class TransformerModel(nn.Module):
    def __init__(self, node_in_channels, hidden_channels, nhead, num_encoder_layers, max_len, action_dim):
        super(TransformerModel, self).__init__()
        self.hidden_channels = hidden_channels
        self.embedding = nn.Linear(node_in_channels, hidden_channels)
        self.position_embedding = nn.Embedding(max_len, hidden_channels)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=nhead, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.action_type = nn.Linear(hidden_channels, action_dim)
        self.layer_index = nn.Linear(hidden_channels, max_len)
        self.parameter_value = nn.Linear(hidden_channels, 1)
        self.value_decoder = nn.Linear(hidden_channels, 1)

    def forward(self, x, mask=None):
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to have 3 dimensions, but got {x.dim()} dimensions.")

        batch_size, seq_len, _ = x.size()
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} exceeds the number of position embeddings {self.position_embedding.num_embeddings}.")

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.embedding(x) + self.position_embedding(positions)
        x = self.transformer_encoder(x, mask)
        x = x.mean(dim=1)  # Global average pooling

        action = F.softmax(self.action_type(x), dim=1)
        layer_index = torch.argmax(self.layer_index(x), dim=1)
        parameter_value = torch.sigmoid(self.parameter_value(x))
        value = self.value_decoder(x)

        return action, layer_index, parameter_value, value

    def act(self, x, mask=None):
        action, layer_index, parameter_value, _ = self.forward(x, mask)
        return action, layer_index, parameter_value

    def evaluate(self, x, mask=None):
        action, layer_index, parameter_value, value = self.forward(x, mask)
        return action, layer_index, parameter_value, value

def update_ppo_based_on_ddpg(ppo_agent, ddpg_agent, shared_memory, isolated_memory, ppo_update_env, template=None,
                             threshold=0.85):
    # Retrieve designs from shared and isolated memory
    designs = shared_memory.retrieve() + isolated_memory.retrieve()
    # Filter or prioritize designs based on the threshold
    filtered_designs = [design for design in designs if design[3] >= threshold]  # design[3] is the reward
    # If no designs meet the threshold, consider using all designs
    if not filtered_designs:
        filtered_designs = designs
    # Select designs for optimization based on the presence of a template
    if template:
        # Filter designs from shared memory that have at least 90% similarity to the template
        filtered_designs = [design for design in shared_memory.retrieve() if
                            calculate_similarity(design[1], design[2], template, ppo_update_env.simulator) >= 0.9]
    else:
        # Use designs from isolated memory for further refinement
        filtered_designs = isolated_memory.retrieve()
    if not filtered_designs:
        filtered_designs = designs
    # Update PPO based on the selected designs
    for state, layers, thicknesses, reward in filtered_designs:
        ppo_update_env.layers = layers
        ppo_update_env.thicknesses = thicknesses
        ppo_update_env.current_merit = reward
        done = False
        optimization_step_count = 0
        while not done and optimization_step_count < ppo_update_env.max_layers:
            actions = ppo_agent.act(state, ppo_update_env.layers, ppo_update_env.min_layers,
                                    ppo_update_env.num_materials)
            next_state, reward, done, _ = ppo_update_env.step(actions[0])
            ppo_agent.remember(state, actions[0], reward, next_state, done)
            state = next_state
            optimization_step_count += 1
            # Debugging prints
            print(f"PPO Update from DDPG Step {optimization_step_count}")
            print(f"Reward: {reward}")

    ppo_agent.update()



def calculate_similarity(layers, thicknesses, template, simulator, alpha=0.85):
    _, T, _ = simulator.spectrum(layers, thicknesses)
    mse = np.mean((T - template) ** 2)
    cosine_similarity = np.dot(T, template) / (np.linalg.norm(T) * np.linalg.norm(template))

    # Normalize cosine similarity to range [0, 1]
    normalized_cosine_similarity = (cosine_similarity + 1) / 2

    # Combine MSE and normalized cosine similarity
    hybrid_similarity = alpha * normalized_cosine_similarity + (1 - alpha) * (1 - mse)
    return hybrid_similarity
