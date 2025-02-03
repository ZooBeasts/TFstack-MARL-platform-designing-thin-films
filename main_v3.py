import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from Simulator import TMM_sim
import os
from DDPG_agent import DDPGAgent
from PPO_agent import PPOAgent, update_ppo_based_on_ddpg
from Memory_env import SharedMemory, IsolatedMemory
from Rewards_config import AdaptiveExploration, ddpg_reward_fun, RewardTracker
from opt_env import ppo_optimization_phase
from utils import save_results, plot_intermediate_results, save_rewards_to_csv
from Env_gym import PPOTMMEnv, DDPGTMMEnv, PPOUpdateTMMEnv, DDPGTMMEnvWithTemplate
import pandas as pd


def train_multi_agent(ppo_agent, ddpg_agent, ddpg_env, ppo_env, ddpg_env_with_template, ppo_update_env, shared_memory,
                      isolated_memory, episodes=1000, save_interval=20, threshold=0.5,
                      output_path='results', upper=None, lower=None, ddpg_template=None,
                      ):
    rewards_per_episode = []
    ppo_rewards = []
    ddpg_rewards = []
    total_rewards = []
    exploration = AdaptiveExploration()
    reward_tracker = RewardTracker()
    # writer = SummaryWriter(log_dir=os.path.join(output_path, 'tensorboard_logs'))

    for e in range(episodes):
        exploration_rate = exploration.get_exploration_rate(e)

        # PPO Exploration Phase
        state = ppo_env.reset()
        done = False
        total_reward = 0
        ppo_episode_reward = 0
        step_count = 0

        print(f"Starting PPO Exploration Phase for Episode {e + 1}")

        while not done and step_count < 150:
            state_tensor = ppo_agent.process_state(state)
            if np.random.rand() < exploration_rate:
                actions = ppo_agent.act(state_tensor, ppo_env.layers, ppo_env.min_layers, ppo_env.num_materials)
            else:
                action_type = ppo_env.action_space.sample()
                layer_idx = np.random.randint(0, ppo_env.max_layers)
                actions = [(action_type, layer_idx)]
            next_state, reward, done, _ = ppo_env.step(actions[0])
            total_reward += reward
            ppo_episode_reward += reward
            ppo_agent.remember(state, actions[0], reward, next_state, done)
            state = next_state
            step_count += 1

            print(f"Episode {e + 1}, Step {step_count}")
            print(f"PPO explore Action: {actions[0]}")
            print(f"Reward: {reward}")

            if step_count >= 200 and done:
                break

        rewards_per_episode.append(total_reward)
        ppo_rewards.append(ppo_episode_reward)
        reward_tracker.add_reward(total_reward)
        # writer.add_scalar('Total Reward', total_reward, e)

        _, _, A = ppo_env.simulator.spectrum(ppo_env.layers, ppo_env.thicknesses)
        within_target_range = False

        if all(r == ppo_env.desired_absorption[0] for r in ppo_env.desired_absorption):
            # All desired reflections are the same
            for target_range in ppo_env.target_wavelength_ranges:
                if target_range is not None:
                    target_idx = (ppo_env.simulator.wavelength >= target_range[0]) & (
                            ppo_env.simulator.wavelength <= target_range[1])
                    if np.all(A[target_idx] <= 1.0):
                        within_target_range = True
                        break
                        # rest_T = A[~target_idx]
                        # if np.all(rest_T > 0.4) or np.min(rest_T) > 0.4:
                        #     within_target_range = True
                        #     break

        else:
            # Desired reflections vary
            for target_range, desired_absorption in zip(ppo_env.target_wavelength_ranges, ppo_env.desired_absorption):
                if target_range is not None:
                    target_idx = (ppo_env.simulator.wavelength >= target_range[0]) & (
                            ppo_env.simulator.wavelength <= target_range[1])
                    if np.all(np.abs(A[target_idx] - desired_absorption) <= 0.05):  # Allow some tolerance
                        within_target_range = True
                        break

        if within_target_range:
            save_results(ppo_env.simulator, ppo_env.layers, ppo_env.thicknesses, e + 1, step_count,
                         f'episode_{e + 1}_target_reached', output_path)
            # template also can be injected here, into the DDPG reward function
            ddpg_reward_val = ddpg_reward_fun(_, _, A, ppo_env.simulator.wavelength, ppo_env.target_wavelength_ranges,
                                              ppo_env.desired_absorption,
                                              ppo_env.previous_layers, ppo_env.previous_thicknesses, template=None,
                                              narrowbands=ppo_env.narrowbands)
            shared_memory.store((state, ppo_env.layers, ppo_env.thicknesses, ddpg_reward_val))

        # DDPG Optimization Phase
        print(f"Starting DDPG Optimization Phase for Episode {e + 1}")
        ddpg_episode_reward = 0

        if ddpg_env_with_template is not None:
            # Template-based sequence search
            for state, layers, thicknesses, reward in shared_memory.retrieve():
                ddpg_state = state
                ddpg_env_with_template.layers = layers
                ddpg_env_with_template.thicknesses = thicknesses
                ddpg_done = False
                ddpg_step_count = 0

                while not ddpg_done and ddpg_step_count < 50:
                    ddpg_action = ddpg_agent.select_action(ddpg_state)
                    ddpg_next_state, ddpg_reward, ddpg_done, _ = ddpg_env_with_template.step(ddpg_action)
                    ddpg_agent.remember(ddpg_state, ddpg_action, ddpg_reward, ddpg_next_state, ddpg_done)
                    ddpg_state = ddpg_next_state
                    ddpg_step_count += 1
                    ddpg_episode_reward += ddpg_reward

                    print(f"DDPG with Template Episode {e + 1}, Step {ddpg_step_count}")
                    print(f"DDPG with Template Action: {ddpg_action}")
                    print(f"Reward: {ddpg_reward}")

                ddpg_agent.update()

                _, _, A = ddpg_env_with_template.simulator.spectrum(ddpg_env_with_template.layers,
                                                                    ddpg_env_with_template.thicknesses)
                within_target_range = False

                if all(r == ddpg_env_with_template.desired_absorption[0] for r in
                       ddpg_env_with_template.desired_absorption):
                    for target_range in ddpg_env_with_template.target_wavelength_ranges:
                        if target_range is not None:
                            target_idx = (ddpg_env_with_template.simulator.wavelength >= target_range[0]) & (
                                    ddpg_env_with_template.simulator.wavelength <= target_range[1])
                            if np.all(A[target_idx] <=1.0):
                                within_target_range = True
                                break
                                # rest_T = A[~target_idx]
                                # if np.all(rest_T > 0.4) or np.min(rest_T) > 0.4:
                                #     within_target_range = True
                                #     break

                else:
                    for target_range, desired_absorption in zip(ddpg_env_with_template.target_wavelength_ranges,
                                                                ddpg_env_with_template.desired_absorption):
                        if target_range is not None:
                            target_idx = (ddpg_env_with_template.simulator.wavelength >= target_range[0]) & (
                                    ddpg_env_with_template.simulator.wavelength <= target_range[1])
                            if np.all(np.abs(A[target_idx] - desired_absorption) <= 0.05):  # Allow some tolerance
                                within_target_range = True
                                break

                if within_target_range:
                    save_results(ddpg_env_with_template.simulator, ddpg_env_with_template.layers,
                                 ddpg_env_with_template.thicknesses, e + 1, ddpg_step_count,
                                 f'episode_{e + 1}_ddpg_template_optimized', output_path)
                    isolated_memory.store(
                        (ddpg_state, ddpg_env_with_template.layers, ddpg_env_with_template.thicknesses, ddpg_reward))
        else:
            # Random search
            for state, layers, thicknesses, reward in shared_memory.retrieve():
                ddpg_state = state
                ddpg_env.layers = layers
                ddpg_env.thicknesses = thicknesses
                ddpg_done = False
                ddpg_step_count = 0

                while not ddpg_done and ddpg_step_count < 50:
                    ddpg_action = ddpg_agent.select_action(ddpg_state)
                    ddpg_next_state, ddpg_reward, ddpg_done, _ = ddpg_env.step(ddpg_action)
                    ddpg_agent.remember(ddpg_state, ddpg_action, ddpg_reward, ddpg_next_state, ddpg_done)
                    ddpg_state = ddpg_next_state
                    ddpg_step_count += 1
                    ddpg_episode_reward += ddpg_reward

                    print(f"DDPG Episode {e + 1}, Step {ddpg_step_count}")
                    print(f"DDPG Action: {ddpg_action}")
                    print(f"Reward: {ddpg_reward}")

                ddpg_agent.update()

                _, _, A = ddpg_env.simulator.spectrum(ddpg_env.layers, ddpg_env.thicknesses)
                within_target_range = False

                if all(r == ddpg_env.desired_absorption[0] for r in ddpg_env.desired_absorption):
                    for target_range in ddpg_env.target_wavelength_ranges:
                        if target_range is not None:
                            target_idx = (ddpg_env.simulator.wavelength >= target_range[0]) & (
                                    ddpg_env.simulator.wavelength <= target_range[1])
                            if np.all(A[target_idx] <= 1.0):
                                within_target_range = True
                                break
                                # rest_T = A[~target_idx]
                                # if np.all(rest_T > 0.4) or np.min(rest_T) > 0.4:


                else:
                    for target_range, desired_absorption in zip(ddpg_env.target_wavelength_ranges,
                                                                ddpg_env.desired_absorption):
                        if target_range is not None:
                            target_idx = (ddpg_env.simulator.wavelength >= target_range[0]) & (
                                    ddpg_env.simulator.wavelength <= target_range[1])
                            if np.all(np.abs(A[target_idx] - desired_absorption) <= 0.05):  # Allow some tolerance
                                within_target_range = True
                                break

                if within_target_range:
                    save_results(ddpg_env.simulator, ddpg_env.layers, ddpg_env.thicknesses, e + 1, ddpg_step_count,
                                 'ddpg_optimized_design', output_path)
                    isolated_memory.store((ddpg_state, ddpg_env.layers, ddpg_env.thicknesses, ddpg_reward))

        ddpg_rewards.append(ddpg_episode_reward)

        # PPO Optimization Phase
        print(f"Starting PPO Optimization Phase for Episode {e + 1}")

        # template can be injected here if needed
        ppo_optimization_phase(ppo_agent, ppo_update_env, shared_memory, isolated_memory, simulator=ppo_env.simulator,
                               target_ranges=ppo_env.target_wavelength_ranges,
                               template=ddpg_template, alpha=threshold, episode=e, output_path=output_path,
                               desired_absorption=ppo_update_env.desired_absorption,
                               narrowbands=ppo_update_env.narrowbands, upper=upper, lower=lower)

        # Update PPO based on DDPG feedback
        update_ppo_based_on_ddpg(ppo_agent, ddpg_agent, shared_memory, isolated_memory, ppo_update_env, template=None,
                                 threshold=threshold)

        total_rewards.append(total_reward + ddpg_episode_reward)

        # Save intermediate results and plot metrics
        if (e + 1) % save_interval == 0:
            save_results(ppo_env.simulator, ppo_env.layers, ppo_env.thicknesses, e + 1, step_count,
                         f"episode_{e + 1}_intermediate_results", output_path)
            plot_intermediate_results(list(range(1, e + 2)), ppo_rewards, ddpg_rewards, total_rewards, output_path)
            save_rewards_to_csv(list(range(1, e + 2)), ppo_rewards, ddpg_rewards, total_rewards, output_path)
            print(f"Intermediate results saved at episode {e + 1}")
            os.makedirs(os.path.join(output_path, 'ppo_agent_model'), exist_ok=True)
            # ppo_agent.save_model(os.path.join(output_path, 'ppo_agent_model', f'ppo_agent_{e + 1}.pth'))
            os.makedirs(os.path.join(output_path, 'ddpg_agent_model'), exist_ok=True)
            # ddpg_agent.save_model(os.path.join(output_path, 'ddpg_agent_model', f'ddpg_agent_{e + 1}.pth'))

        # Final Validation
        # validate_saved_designs(ppo_env.simulator, output_path)

    # writer.close()


if __name__ == "__main__":

    output_path = 'abs_test'
    os.makedirs(output_path, exist_ok=True)

    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed: {seed}")
    seed_df = pd.DataFrame([seed], columns=['seed'])
    seed_df.to_csv(os.path.join(output_path, 'seed.csv'))

    # saved_path = 'resim'
    # os.makedirs(saved_path, exist_ok=True)

    available_materials = [
        {"material": "TiO2", "refractive_index_file": "TiO2.csv", "type": "oxide"},
        {"material": "SiO2", "refractive_index_file": "SiO2.csv", "type": "oxide"},
        {"material": "Fe2O3", "refractive_index_file": "Fe2O3.csv", "type": "oxide"},
        {"material": "Al2O3", "refractive_index_file": "Al2O3.csv", "type": "oxide"},
        {"material": "Ge", "refractive_index_file": "Ge.csv", "type": "glue"},
        {"material": "Ag", "refractive_index_file": "Ag.csv", "type": "metal"},
        {"material": "Ti", "refractive_index_file": "Ti.csv", "type": "metal"},
        {"material": "HfO2", "refractive_index_file": "HfO2.csv", "type": "oxide"},
    ]

    # {"material": "Fe2O3", "refractive_index_file": "Fe2O3.csv", "type": "oxide"},
    # {"material": "Fe2O3", "refractive_index_file": "Fe2O3.csv", "type": "oxide"},
    # {"material": "Al2O3", "refractive_index_file": "Al2O3.csv", "type": "oxide"},
    # {"material": "HfO2", "refractive_index_file": "HfO2.csv", "type": "oxide"},
    substrate_materials = [
        {"material": "Glass", "refractive_index_file": "Glass.csv"},
        # {"material": "Sapphire", "refractive_index_file": "Sapphire.csv"}
    ]
    # wavelengths = np.linspace(500, 800, 300)
    wavelengths = np.arange(450, 1100, 2)
    substrate_material = "Glass"
    substrate_thickness = 500
    min_layers = 10
    max_layers = 46
    upper_thickness = 200
    lower_thickness = 15
    metal_lower = 15
    metal_upper = 150
    stacking_mode = ['periodic', 'random'][1]

    # narrowbands = [30, 14, 30] # narrow band, each value corresponds target wavelength range values
    narrowbands = [0]
    # For the target wavelength range (570, 600) with a narrowband of 30 nm:
    # The center wavelength of the target range is (570 + 600) / 2 = 585 nm.
    # The Gaussian profile will be centered at 585 nm with a FWHM of 30 nm.
    # This means that the Gaussian curve will drop to half of its peak value at 585 ± 15 nm (i.e., at 570 nm and 600 nm).
    desired_reflections = [0]  # define the desired reflection for the narrowband

    # Define a template if needed
    # ddpg_template = np.array(np.loadtxt('template_step5.txt'))   # Set to None if no template is needed
    # ddpg_template = None
    # temp = np.array(
    #     [0.85,0.85,0.85,0.85,0.85,0.85 ,0.85 ,0.85, 0.85, 0.85, 0.85, 0.7 ,0.7, 0.7,0.5,
    #      0.5,0.5, 0.5,0.5,0.3,0.3,0.3,0.85,0.85 ,0.85 ,0.85, 0.85 ,0.85,0.85,0.85
    #      ])
    ddpg_template = None

    simulator = TMM_sim(available_materials, substrate_materials, wavelengths, substrate_material, substrate_thickness)
    target = {'A': np.ones_like(simulator.wavelength)}
    target_wavelength_ranges = [(450,500)]

    ppo_env = PPOTMMEnv(simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                        stack_mode=stacking_mode, desired_absorption=desired_reflections, narrowbands=narrowbands,
                        upper=upper_thickness, lower=lower_thickness, metal_lower=metal_lower, metal_upper=metal_upper)
    ddpg_env = DDPGTMMEnv(simulator, target, available_materials, target_wavelength_ranges, min_layers, max_layers,
                          stack_mode=stacking_mode, desired_absorption=desired_reflections, narrowbands=narrowbands,
                          upper=upper_thickness, lower=lower_thickness, metal_lower=metal_lower,
                          metal_upper=metal_upper)

    if ddpg_template is not None and len(ddpg_template) > 0:
        ddpg_env_with_template = DDPGTMMEnvWithTemplate(simulator, target, available_materials,
                                                        target_wavelength_ranges, min_layers, max_layers, ddpg_template,
                                                        stack_mode=stacking_mode, desired_absorption=desired_reflections,
                                                        narrowbands=narrowbands, upper=upper_thickness,
                                                        lower=lower_thickness, metal_lower=metal_lower,
                                                        metal_upper=metal_upper)
    else:
        ddpg_env_with_template = None

    ppo_update_env = PPOUpdateTMMEnv(simulator, target, available_materials, target_wavelength_ranges, min_layers,
                                     max_layers, stack_mode=stacking_mode, desired_absorption=desired_reflections,
                                     narrowbands=narrowbands, upper=upper_thickness, lower=lower_thickness,
                                     metal_lower=metal_lower, metal_upper=metal_upper)

    node_in_channels = 2
    hidden_channels = 128
    action_dim = ppo_env.action_space.n

    ppo_agent = PPOAgent(state_dim=ppo_env.observation_space.shape[0], action_dim=action_dim,
                         hidden_dim=hidden_channels)
    ddpg_agent = DDPGAgent(state_dim=ddpg_env.observation_space.shape[0], action_dim=2, hidden_dim=hidden_channels,
                           max_layers=max_layers)

    if ddpg_env_with_template:
        ddpg_agent_with_template = DDPGAgent(state_dim=ddpg_env_with_template.observation_space.shape[0],
                                             action_dim=ddpg_env_with_template.action_space.n,
                                             hidden_dim=hidden_channels, max_layers=max_layers)
    else:
        ddpg_agent_with_template = None

    shared_memory = SharedMemory(max_size=100)
    isolated_memory = IsolatedMemory(max_size=100)

    train_multi_agent(ppo_agent, ddpg_agent, ppo_env, ddpg_env, ddpg_env_with_template,
                      ppo_update_env, shared_memory, isolated_memory, episodes=400000, save_interval=5, threshold=0.8,
                      output_path=output_path, upper=upper_thickness, lower=lower_thickness,
                      ddpg_template=ddpg_template)
