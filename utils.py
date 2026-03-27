
import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import csv
import matplotlib.pyplot as plt


def plot_intermediate_results(episodes, ppo_rewards, ddpg_rewards, total_rewards, save_path):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(episodes, ppo_rewards, label='PPO Reward')
    plt.xlabel('Episodes')
    plt.ylabel('PPO Reward')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(episodes, ddpg_rewards, label='DDPG Reward')
    plt.xlabel('Episodes')
    plt.ylabel('DDPG Reward')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(episodes, total_rewards, label='Total Reward')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'intermediate_plot_{episodes[-1]}.png'))
    plt.close()


def save_rewards_to_csv(episodes, ppo_rewards, ddpg_rewards, total_rewards, save_path):
    csv_file = os.path.join(save_path, 'reward_data.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Episode', 'PPO Reward', 'DDPG Reward', 'Total Reward'])
        for e, ppo_r, ddpg_r, total_r in zip(episodes, ppo_rewards, ddpg_rewards, total_rewards):
            writer.writerow([e, ppo_r, ddpg_r, total_r])


def save_results(simulator, layers, thicknesses, episode, step, file_prefix, output_directory):
    result = simulator.append_properties_to_output(layers, thicknesses)
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, f'{file_prefix}_episode_{episode}_step_{step}.json')
    with open(output_file_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)
    print(f"Saved results to {output_file_path}")

def save_ddpg_optimized_design(simulator, layers, thicknesses, episode, step, output_directory):
    result = simulator.append_properties_to_output(layers, thicknesses)
    filename = f'ddpg_optimized_episode_{episode}_step_{step}.json'
    output_file_path = os.path.join(output_directory, filename)
    with open(output_file_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)
    print(f"Saved DDPG optimized design to {output_file_path}")

def save_ppo_optimized_design(simulator, layers, thicknesses, episode, step, output_directory):
    result = simulator.append_properties_to_output(layers, thicknesses)
    filename = f'ppo_optimized_episode_{episode}_step_{step}.json'
    output_file_path = os.path.join(output_directory, filename)
    with open(output_file_path, 'w') as ppo_opt_file:
        json.dump(result, ppo_opt_file, indent=4)
    print(f"Saved PPO optimized design to {output_file_path}")

def save_ppo_optimized_bgs_design(simulator, layers, thicknesses, episode, step, output_directory):
    result = simulator.append_properties_to_output(layers, thicknesses)
    filename = f'ppo_optimized_phase_CO_episode_{episode}_step_{step}.json'
    output_file_path = os.path.join(output_directory, filename)
    with open(output_file_path, 'w') as ppo_opt_file:
        json.dump(result, ppo_opt_file, indent=4)
    print(f"Saved PPO optimized design to {output_file_path}")



# def validate_saved_designs(simulator, saved_path):
#     for file_name in os.listdir(saved_path):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(saved_path, file_name)
#             with open(file_path, 'r') as file:
#                 design = json.load(file)
#                 layers = design['layers']
#                 thicknesses = design['thicknesses']
#                 saved_transmission = design['transmission']
#
#                 # Re-simulate transmission
#                 _, T, _ = simulator.spectrum(layers, thicknesses)
#
#                 # Compare saved transmission with re-simulated transmission
#                 if np.allclose(T, saved_transmission):
#                     print(f"{file_name}: Checked")
#                 else:
#                     print(f"{file_name}: Transmission mismatch, deleting file.")
#                     os.remove(file_path)




#
#
# def prepare_data_for_unified(layers, thicknesses, available_materials, max_layers, upper, lower):
#     THICKNESS_MEAN = np.mean((lower + upper) / 2)
#     THICKNESS_STD = np.std(np.arange(lower, upper, 1))
#
#     encoder = LabelEncoder()
#     material_list = [material_info['material'] for material_info in available_materials]
#     encoder.fit(material_list)
#     material_encoded = encoder.transform(layers)
#
#     thickness_normalized = [(thickness - THICKNESS_MEAN) / THICKNESS_STD for thickness in thicknesses]
#     data = []
#
#     for material, thickness in zip(material_encoded, thickness_normalized):
#         data.extend([material, thickness])
#
#     # Ensure the data length matches max_layers * 2 (2 features per layer)
#     expected_length = max_layers * 2
#     if len(data) < expected_length:
#         padding_length = expected_length - len(data)
#         data.extend([0, -2.0] * (padding_length // 2))
#     elif len(data) > expected_length:
#         data = data[:expected_length]
#
#     return np.array(data, dtype=np.float32)


def prepare_data_for_unified(layers, thicknesses, available_materials, max_layers, upper, lower):
    encoder = LabelEncoder()
    material_list = [m['material'] for m in available_materials]
    encoder.fit(material_list)
    material_encoded = encoder.transform(layers)

    thickness_range = float(upper - lower)
    if thickness_range <= 0:
        thickness_range = 1.0  # avoid divide-by-zero

    # [lower, upper] -> [0, 1]
    thickness_normalized = [(th - lower) / thickness_range for th in thicknesses]

    data = []
    for material, thickness in zip(material_encoded, thickness_normalized):
        data.extend([material, thickness])

    expected_length = max_layers * 2
    if len(data) < expected_length:
        padding_length = expected_length - len(data)
        data.extend([0, 0.0] * (padding_length // 2))
    elif len(data) > expected_length:
        data = data[:expected_length]

    return np.array(data, dtype=np.float32)










