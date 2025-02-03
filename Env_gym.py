import gymnasium as gym
import numpy as np
from gymnasium import spaces
from utils import prepare_data_for_unified
from Rewards_config import RewardTracker, reward_shaping, ddpg_reward_fun, ppo_optimization_reward
import random


def validate_layers_thicknesses(layers, thicknesses):
    if len(layers) != len(thicknesses):
        raise ValueError("Mismatch between layers and thicknesses lengths.")



class PPOTMMEnv(gym.Env):
    def __init__(self, simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                 target_reflection=None, agent_template=None, min_actions_per_episode=100, reward_tracker=None,
                 stack_mode='periodic', desired_absorption=None, narrowbands= None,
                 upper = None, lower = None,metal_lower=None,metal_upper=None):
        super(PPOTMMEnv, self).__init__()
        self.simulator = simulator
        self.target = target
        self.available_materials = available_materials
        self.target_wavelength_ranges = target_wavelength_ranges
        self.target_reflection = None
        self.materials = [material_info['material'] for material_info in available_materials]
        self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.material_types = {material_info['material']: material_info['type'] for material_info in
                               available_materials}
        self.num_materials = len(self.materials)
        self.current_merit = 0
        self.agent_template = agent_template
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.best_design = None
        self.reward_tracker = reward_tracker or RewardTracker()
        self.stack_mode = stack_mode
        self.desired_absorption = desired_absorption
        self.min_actions_per_episode = min_actions_per_episode
        self.current_step_count = 0
        self.narrowbands = narrowbands
        self.upper = upper
        self.lower = lower
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper

        if stack_mode == 'periodic':
            self.action_space = spaces.Discrete(self.num_materials + 2)  # Add, Modify, Done
            print('Periodic Mode')
        else:
            self.action_space = spaces.Discrete(self.num_materials + 3)  # Add, Modify, Remove, Done
            print('Random Mode')

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_layers * 2,),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        action_type, layer_idx = action

        if action_type < 0 or action_type >= self.num_materials + (3 if self.stack_mode == 'random' else 2):
            raise ValueError("Invalid action: action_type out of range.")

        if len(self.layers) < self.min_layers and action_type >= self.num_materials:
            action_type = np.random.randint(0, self.num_materials)  # Force an add action

        if len(self.layers) >= self.max_layers:
            if action_type < self.num_materials:
                action_type = self.num_materials + 1  # Force a modify action

        if action_type == self.num_materials and self.current_step_count < self.min_actions_per_episode:
            if len(self.layers) >= self.max_layers:
                action_type = self.num_materials + 1  # Force a modify action if max layers reached
            else:
                action_type = np.random.randint(0, self.num_materials)  # Force a random action

        if action_type == self.num_materials:  # Done action
            print("Done action chosen. Ending episode.")
            self.current_step_count = 0
            return self._get_obs(), self.current_merit, True, {}

        elif action_type == self.num_materials + 1:  # Modify action
            if len(self.layers) == 0:
                raise ValueError("No layers to modify.")
            layers_to_modify = np.random.randint(1, 9)
            for _ in range(layers_to_modify):
                layer_to_modify = np.random.randint(len(self.layers))
                material = self.layers[layer_to_modify]
                thickness_range = self.get_thickness_range(material)
                new_thickness = np.random.randint(*thickness_range)
                self.thicknesses[layer_to_modify] = new_thickness
                print(f"Modified layer {layer_to_modify} to thickness {new_thickness}")

        elif self.stack_mode == 'random' and action_type == self.num_materials + 2:  # Remove action
            if len(self.layers) <= self.min_layers:
                print("Skipping invalid action: Cannot remove layer below minimum layers.")
                return self._get_obs(), 0, False, {}
            layer_to_remove = layer_idx if layer_idx < len(self.layers) else -1
            self.layers.pop(layer_to_remove)
            self.thicknesses.pop(layer_to_remove)
            print("Removed layer at index", layer_idx)

        else:  # Add action
            if action_type >= self.num_materials:
                raise ValueError("Invalid action: action_type out of range.")

            # Check if the last layer is Ag and add Ge instead
            if self.layers and self.layers[-1] == 'Ag':
                layer = 'Ge'
                print("Adding Ge after Ag")
            else:
                if self.stack_mode == 'periodic':
                    if len(self.layers) > 0:
                        last_layer = self.layers[-1]
                        last_material_index = self.materials.index(last_layer)
                        next_material_index = (last_material_index + 1) % self.num_materials
                        layer = self.materials[next_material_index]
                    else:
                        layer = self.materials[0]
                else:  # Random stacking
                    layer = random.choice(self.materials)

            thickness_range = self.get_thickness_range(layer)
            new_thickness = np.random.randint(*thickness_range)
            if len(self.layers) < self.max_layers:
                self.layers.append(layer)
                self.thicknesses.append(new_thickness)
                print(f"Added layer {layer} with thickness {new_thickness}")

        # Save previous state
        self.previous_layers = self.layers.copy()
        self.previous_thicknesses = self.thicknesses.copy()

        validate_layers_thicknesses(self.layers, self.thicknesses)

        _, _, A = self.simulator.spectrum(self.layers, self.thicknesses)

        reward = reward_shaping(_, _, A, self.simulator.wavelength, self.target_wavelength_ranges,
                                self.desired_absorption, self.reward_tracker, self.target_reflection, self.narrowbands)
        self.current_merit = reward

        print(f"Current layers = {self.layers}")
        print(f"Current thicknesses = {self.thicknesses}")

        self.current_step_count += 1

        return self._get_obs(), reward, False, {}

    def get_thickness_range(self, material):
        if self.material_types[material] == 'metal':
            return self.metal_lower, self.metal_upper
        elif self.material_types[material] == 'glue':
            return 4, 10
        elif self.material_types[material] == 'dielectric' or self.material_types[material] == 'oxide':
            return self.lower, self.upper
        else:
            return self.lower, self.upper

    def reset(self):
        self.layers = []
        self.thicknesses = []

        initial_material_index = 0
        for i in range(self.min_layers):
            layer = self.materials[(initial_material_index + i) % self.num_materials]
            thickness = np.random.randint(self.lower, self.upper)
            self.layers.append(layer)
            self.thicknesses.append(thickness)

        self.current_merit = 0
        self.current_step_count = 0
        return self._get_obs()

    def _get_obs(self):
        state = prepare_data_for_unified(self.layers, self.thicknesses, self.available_materials, self.max_layers, self.upper, self.lower)
        return state

    def render(self, mode='human'):
        self.simulator.spectrum(self.layers, self.thicknesses, plot=True)


class DDPGTMMEnv(gym.Env):
    def __init__(self, simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                 agent_template=None, min_actions_per_episode=100, stack_mode='periodic', desired_absorption=None,
                 narrowbands=None,upper=None,lower=None,metal_lower=None,metal_upper=None):
        super(DDPGTMMEnv, self).__init__()
        self.simulator = simulator
        self.target = target
        self.available_materials = available_materials
        self.target_wavelength_ranges = target_wavelength_ranges
        self.materials = [material_info['material'] for material_info in available_materials]
        self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.material_types = {material_info['material']: material_info['type'] for material_info in available_materials}
        self.num_materials = len(self.materials)
        self.current_merit = 0
        self.agent_template = agent_template
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.best_design = None
        self.stack_mode = stack_mode
        self.desired_absorption = desired_absorption
        self.narrowbands = narrowbands
        self.min_actions_per_episode = min_actions_per_episode
        self.current_step_count = 0
        self.upper = upper
        self.lower = lower
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper


        self.action_space = spaces.Discrete(self.num_materials + 3)
         # Add, Modify, Remove, Done actions

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_layers * 2,),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        action_type, layer_idx = action

        if action_type >= self.action_space.n:
            raise ValueError("Invalid action: action_type out of range. ddpg env")

        if len(self.layers) < self.min_layers:
            if action_type >= self.num_materials:
                action_type = np.random.randint(0, self.num_materials)

        if len(self.layers) >= self.max_layers:
            print("Max layers reached. Only 'done' or 'modify' action is allowed.")
            if action_type < self.num_materials:
                action_type = self.num_materials + 1  # Force a modify action

        if action_type == self.num_materials and self.current_step_count < self.min_actions_per_episode:
            print("Minimum actions not yet taken. Forcing a random action.")
            action_type = np.random.randint(0, self.num_materials)  # Force a random action

        if action_type == self.num_materials:
            print("Done action chosen. Ending episode.")
            self.current_step_count = 0
            return self._get_obs(), self.current_merit, True, {}

        elif action_type == self.num_materials + 1:  # Modify action
            if len(self.layers) == 0:
                raise ValueError("No layers to modify.")
            layers_to_modify = np.random.randint(1, 9)
            for _ in range(layers_to_modify):
                layer_to_modify = np.random.randint(len(self.layers))
                material = self.layers[layer_to_modify]
                thickness_range = self.get_thickness_range(material)
                new_thickness = np.random.randint(*thickness_range)
                self.thicknesses[layer_to_modify] = new_thickness
                print(f"Modified layer {layer_to_modify} to thickness {new_thickness}")

        elif self.stack_mode == 'random' and action_type == self.num_materials + 2:  # Remove action
            if len(self.layers) <= self.min_layers:
                print("Skipping invalid action: Cannot remove layer below minimum layers.")
                return self._get_obs(), 0, False, {}
            layer_to_remove = layer_idx if layer_idx < len(self.layers) else -1
            self.layers.pop(layer_to_remove)
            self.thicknesses.pop(layer_to_remove)
            print("Removed layer at index", layer_idx)

        else:  # Add action
            if self.stack_mode == 'periodic':
                if len(self.layers) > 0:
                    last_layer = self.layers[-1]
                    last_material_index = self.materials.index(last_layer)
                    next_material_index = (last_material_index + 1) % self.num_materials
                    layer = self.materials[next_material_index]
                else:
                    layer = self.materials[0]
            else:  # Random stacking
                layer = random.choice(self.materials)

            thickness_range = self.get_thickness_range(layer)
            new_thickness = np.random.randint(*thickness_range)
            if len(self.layers) < self.max_layers:
                self.layers.append(layer)
                self.thicknesses.append(new_thickness)
                print(f"Added layer {layer} with thickness {new_thickness}")

        # Save previous state
        self.previous_layers = self.layers.copy()
        self.previous_thicknesses = self.thicknesses.copy()

        validate_layers_thicknesses(self.layers, self.thicknesses)

        _, _, A = self.simulator.spectrum(self.layers, self.thicknesses)

        reward = ddpg_reward_fun(_, _, A, self.simulator.wavelength, self.target_wavelength_ranges,self.desired_absorption, self.previous_layers, self.previous_thicknesses, template=None, narrowbands=self.narrowbands)
        self.current_merit = reward

        print(f"Current layers: {self.layers}")
        print(f"Current thicknesses: {self.thicknesses}")

        self.current_step_count += 1

        return self._get_obs(), reward, False, {}

    def get_thickness_range(self, material):
        if self.material_types[material] == 'metal':
            return self.metal_lower, self.metal_upper
        elif self.material_types[material] == 'glue':
            return 4, 10
        elif self.material_types[material] == 'dielectric' or self.material_types[material] == 'oxide':
            return self.lower, self.upper
        else:
            return self.lower, self.upper

    def reset(self):
        self.layers = []
        self.thicknesses = []
        initial_material_index = 0
        for i in range(self.min_layers):
            layer = self.materials[(initial_material_index + i) % self.num_materials]
            thickness = np.random.randint(self.lower, self.upper)
            self.layers.append(layer)
            self.thicknesses.append(thickness)

        self.current_merit = 0
        self.current_step_count = 0
        return self._get_obs()

    def _get_obs(self):
        state = prepare_data_for_unified(self.layers, self.thicknesses, self.available_materials, self.max_layers, self.upper, self.lower)
        return state

    def render(self, mode='human'):
        self.simulator.spectrum(self.layers, self.thicknesses, plot=True)




class DDPGTMMEnvWithTemplate(gym.Env):
    def __init__(self, simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                 template, min_actions_per_episode=100, stack_mode='periodic',
                 desired_absorption=None,narrowbands=None, upper=None,lower=None,metal_lower=None,metal_upper=None):
        super(DDPGTMMEnvWithTemplate, self).__init__()
        self.simulator = simulator
        self.target = target
        self.available_materials = available_materials
        self.target_wavelength_ranges = target_wavelength_ranges
        self.materials = [material_info['material'] for material_info in available_materials]
        self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.material_types = {material_info['material']: material_info['type'] for material_info in available_materials}
        self.num_materials = len(self.materials)
        self.current_merit = 0
        self.agent_template = template
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.best_design = None
        self.stack_mode = stack_mode
        self.desired_absorption = desired_absorption
        self.min_actions_per_episode = min_actions_per_episode
        self.current_step_count = 0
        self.narrowbands = narrowbands
        self.lower = lower
        self.upper = upper
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper

        if template is not None and len(template) > 0:
            print("Template provided, switching to sequence mode.")
            self.action_space = spaces.Discrete(self.num_materials + 2)  # Add, Modify, Done actions
        else:
            raise ValueError("Template must be provided for DDPGTMMEnvWithTemplate.")

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_layers * 2,),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        action_type, layer_idx = action

        if action_type == self.num_materials:  # Done action
            print("Done action chosen. Ending episode.")
            self.current_step_count = 0
            return self._get_obs(), self.current_merit, True, {}

        elif action_type == self.num_materials + 1:  # Modify action
            if len(self.layers) == 0:
                raise ValueError("No layers to modify.")
            layer_to_modify = layer_idx % len(self.layers)  # Ensure layer_idx is within range
            material = self.layers[layer_to_modify]
            thickness_range = self.get_thickness_range(material)
            new_thickness = np.random.randint(*thickness_range)
            self.thicknesses[layer_to_modify] = new_thickness
            print(f"Modified layer {layer_to_modify} to thickness {new_thickness}")

        else:  # Add action
            if action_type >= self.num_materials:
                raise ValueError("Invalid action: action_type out of range. DDPG template env")
            if len(self.layers) < self.max_layers:
                if self.stack_mode == 'periodic':
                    if len(self.layers) > 0:
                        last_layer = self.layers[-1]
                        last_material_index = self.materials.index(last_layer)
                        next_material_index = (last_material_index + 1) % self.num_materials
                        layer = self.materials[next_material_index]
                    else:
                        layer = self.materials[0]
                else:  # Random stacking
                    used_materials = set(self.layers)
                    available_materials = [m for m in self.materials if m not in used_materials]
                    if not available_materials:
                        available_materials = self.materials
                    layer = random.choice(available_materials)

                thickness_range = self.get_thickness_range(layer)
                new_thickness = np.random.randint(*thickness_range)
                self.layers.append(layer)
                self.thicknesses.append(new_thickness)
                print(f"Added layer {layer} with thickness {new_thickness}")

        # Save previous state
        self.previous_layers = self.layers.copy()
        self.previous_thicknesses = self.thicknesses.copy()

        validate_layers_thicknesses(self.layers, self.thicknesses)

        _, _, A = self.simulator.spectrum(self.layers, self.thicknesses)

        reward = ddpg_reward_fun(_, _, A, self.simulator.wavelength, self.target_wavelength_ranges, self.desired_absorption,
                                 self.previous_layers, self.previous_thicknesses, template=self.agent_template, narrowbands=self.narrowbands)
        self.current_merit = reward

        print(f"Current layers: {self.layers}")
        print(f"Current thicknesses: {self.thicknesses}")

        self.current_step_count += 1

        return self._get_obs(), reward, False, {}

    def get_thickness_range(self, material):
        if self.material_types[material] == 'metal':
            return self.metal_lower, self.metal_upper
        elif self.material_types[material] == 'glue':
            return 4,10
        elif self.material_types[material] == 'dielectric' or self.material_types[material] == 'oxide':
            return self.lower, self.upper
        else:
            return self.lower, self.upper

    def reset(self):
        self.layers = []
        self.thicknesses = []

        initial_material_index = 0
        for i in range(1):  # Start with one layer for sequence mode
            layer = self.materials[initial_material_index % self.num_materials]
            thickness = np.random.randint(self.lower, self.upper)
            self.layers.append(layer)
            self.thicknesses.append(thickness)

        self.current_merit = 0
        self.current_step_count = 0
        return self._get_obs()

    def _get_obs(self):
        state = prepare_data_for_unified(self.layers, self.thicknesses, self.available_materials, self.max_layers, self.upper, self.lower)
        return state

    def render(self, mode='human'):
        self.simulator.spectrum(self.layers, self.thicknesses, plot=True)











class PPOUpdateTMMEnv(gym.Env):
    def __init__(self, simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                 agent_template=None, min_actions_per_episode=50, reward_tracker=None,
                 stack_mode='periodic',
                 desired_absorption=None,
                 narrowbands=None,
                 upper=None, lower=None,
                 metal_lower=None, metal_upper=None):
        super(PPOUpdateTMMEnv, self).__init__()
        self.simulator = simulator
        self.target = target
        self.available_materials = available_materials
        self.target_wavelength_ranges = target_wavelength_ranges
        self.materials = [material_info['material'] for material_info in available_materials]
        self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.material_types = {material_info['material']: material_info['type'] for material_info in
                               available_materials}
        self.num_materials = len(self.materials)
        self.current_merit = 0
        self.agent_template = agent_template
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.best_design = None
        self.stack_mode = stack_mode
        self.min_actions_per_episode = min_actions_per_episode
        self.current_step_count = 0
        self.desired_absorption = desired_absorption
        self.narrowbands = narrowbands
        self.upper = upper
        self.lower = lower
        self.metal_lower = metal_lower
        self.metal_upper = metal_upper


        if stack_mode == 'periodic':
            self.action_space = spaces.Discrete(self.num_materials + 2)  # Add, Modify, Done
        else:
            self.action_space = spaces.Discrete(self.num_materials + 3)  # Add, Modify, Remove, Done

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.max_layers * 2,),
            dtype=np.float32
        )

        self.reset()

    def step(self, action):
        action_type, layer_idx = action

        if action_type < 0 or action_type >= self.num_materials + (3 if self.stack_mode == 'random' else 2):
            raise ValueError("Invalid action: action_type out of range. ppo update env")

        if len(self.layers) < self.min_layers and action_type >= self.num_materials:
            action_type = np.random.randint(0, self.num_materials)  # Force an add action

        if len(self.layers) >= self.max_layers:
            if action_type < self.num_materials:
                action_type = self.num_materials + 1  # Force a modify action

        if action_type == self.num_materials and self.current_step_count < self.min_actions_per_episode:
            if len(self.layers) >= self.max_layers:
                action_type = self.num_materials + 1  # Force a modify action if max layers reached
            else:
                action_type = np.random.randint(0, self.num_materials)  # Force a random action

        if action_type == self.num_materials:  # Done action
            print("Done action chosen. Ending episode.")
            self.current_step_count = 0
            return self._get_obs(), self.current_merit, True, {}

        elif action_type == self.num_materials + 1:  # Modify action
            if len(self.layers) == 0:
                raise ValueError("No layers to modify.")
            layers_to_modify = np.random.randint(1, 9)
            for _ in range(layers_to_modify):
                layer_to_modify = np.random.randint(len(self.layers))
                material = self.layers[layer_to_modify]
                thickness_range = self.get_thickness_range(material)
                new_thickness = np.random.randint(*thickness_range)
                self.thicknesses[layer_to_modify] = new_thickness
                print(f"Modified layer {layer_to_modify} to thickness {new_thickness}")

        elif self.stack_mode == 'random' and action_type == self.num_materials + 2:  # Remove action
            if len(self.layers) <= self.min_layers:
                print("Skipping invalid action: Cannot remove layer below minimum layers.")
                return self._get_obs(), 0, False, {}
            layer_to_remove = layer_idx if layer_idx < len(self.layers) else -1
            self.layers.pop(layer_to_remove)
            self.thicknesses.pop(layer_to_remove)
            print("Removed layer at index", layer_idx)

        else:  # Add action
            if action_type >= self.num_materials:
                raise ValueError("Invalid action: action_type out of range.ppo update env")
            if self.stack_mode == 'periodic':
                if len(self.layers) > 0:
                    last_layer = self.layers[-1]
                    last_material_index = self.materials.index(last_layer)
                    next_material_index = (last_material_index + 1) % self.num_materials
                    layer = self.materials[next_material_index]
                else:
                    layer = self.materials[0]
            else:  # Random stacking
                layer = random.choice(self.materials)

            thickness_range = self.get_thickness_range(layer)
            new_thickness = np.random.randint(*thickness_range)
            if len(self.layers) < self.max_layers:
                self.layers.append(layer)
                self.thicknesses.append(new_thickness)
                print(f"Added layer {layer} with thickness {new_thickness}")

        # Save previous state
        self.previous_layers = self.layers.copy()
        self.previous_thicknesses = self.thicknesses.copy()

        validate_layers_thicknesses(self.layers, self.thicknesses)

        _, _, A = self.simulator.spectrum(self.layers, self.thicknesses)

        reward = ppo_optimization_reward(_, _, A, self.simulator.wavelength, self.target_wavelength_ranges, self.desired_absorption, narrowbands=self.narrowbands)
        self.current_merit = reward

        print(f"Current layers: {self.layers}")
        print(f"Current thicknesses: {self.thicknesses}")

        self.current_step_count += 1

        return self._get_obs(), reward, False, {}

    def get_thickness_range(self, material):
        if self.material_types[material] == 'metal':
            return self.metal_lower, self.metal_upper
        elif self.material_types[material] == 'glue':
            return 4,10
        elif self.material_types[material] == 'dielectric' or self.material_types[material] == 'oxide':
            return self.lower, self.upper
        else:
            return self.lower, self.upper

    def reset(self):
        self.layers = []
        self.thicknesses = []

        initial_material_index = 0
        for i in range(self.min_layers):
            layer = self.materials[(initial_material_index + i) % self.num_materials]
            thickness = np.random.randint(self.lower, self.upper)
            self.layers.append(layer)
            self.thicknesses.append(thickness)

        self.current_merit = 0
        self.current_step_count = 0
        return self._get_obs()

    def _get_obs(self):
        state = prepare_data_for_unified(self.layers, self.thicknesses, self.available_materials, self.max_layers,self.upper, self.lower)
        return state

    def render(self, mode='human'):
        self.simulator.spectrum(self.layers, self.thicknesses, plot=True)
