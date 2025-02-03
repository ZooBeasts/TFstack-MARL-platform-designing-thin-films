import numpy as np
from collections import deque


class AdaptiveExploration:
    def __init__(self, initial_epsilon=1.0, min_epsilon=0.1, decay=0.995):
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def get_exploration_rate(self, episode):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        return self.epsilon

    def update_epsilon(self, average_reward, reward_trend):
        if reward_trend > 0:
            self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
        else:
            self.epsilon = min(1.0, self.epsilon * 1.01)

class RewardTracker:
    def __init__(self):
        self.rewards = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def get_average_reward(self):
        if not self.rewards:
            return 0
        return np.mean(self.rewards)

    def get_reward_trend(self):
        if len(self.rewards) < 2:
            return 0
        return self.rewards[-1] - self.rewards[-2]


def inverted_trapezoid_profile(x, lower_start, lower_end, upper_start, upper_end, lower_value, upper_value):
    profile = np.zeros_like(x)
    lower_mask = (x >= lower_start) & (x <= lower_end)
    upper_mask = (x >= upper_start) & (x <= upper_end)
    slope_left = (upper_value - lower_value) / (lower_start - upper_start)
    slope_right = (upper_value - lower_value) / (lower_end - upper_end)

    profile[lower_mask] = lower_value
    profile[upper_mask & (x < lower_start)] = slope_left * (
                x[upper_mask & (x < lower_start)] - upper_start) + upper_value
    profile[upper_mask & (x > lower_end)] = slope_right * (x[upper_mask & (x > lower_end)] - upper_end) + upper_value

    return profile


def reward_shaping(R, T, A, wavelength, target_ranges, desired_absorption, reward_tracker, baseline_reflection=None, narrowbands=None):
    reward = 0.0
    penalty_target_range = 10  # Penalty for deviation from desired reflection within the target range
    reward_outside_target = 25  # Reward for high reflection outside the target range
    smoothness_weight = 0.1  # Weight for smoothness penalty
    baseline_weight = 0.5  # Weight for baseline comparison penalty
    transition_reward_weight =8  # Weight for transition region reward
    trapezoid_penalty_weight = 15  # Penalty weight for inverted trapezoid profile mismatch
    high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

    for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands):
        if target_range is None:
            continue

        idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
        idx_outside_target = ~idx_target

        reflection_in_target_range = A[idx_target]
        reflection_outside_target = A[idx_outside_target]

        if narrowband is not None and narrowband != 0:
            target_wavelengths = wavelength[idx_target]
            lower_bound_start = target_range[0]
            lower_bound_end = target_range[1]
            upper_bound_start = lower_bound_start - (narrowband - (lower_bound_end - lower_bound_start)) / 2
            upper_bound_end = lower_bound_end + (narrowband - (lower_bound_end - lower_bound_start)) / 2

            # Desired trapezoid profile
            trapezoid_profile = np.piecewise(
                target_wavelengths,
                [target_wavelengths < lower_bound_start,
                 (target_wavelengths >= lower_bound_start) & (target_wavelengths <= lower_bound_end),
                 target_wavelengths > lower_bound_end],
                [lambda x: np.interp(x, [upper_bound_start, lower_bound_start], [high_reflection_value, desired_reflection]),
                 desired_reflection,
                 lambda x: np.interp(x, [lower_bound_end, upper_bound_end], [desired_reflection, high_reflection_value])]
            )

            # Calculate the trapezoid matching penalty
            trapezoid_mismatch = np.abs(reflection_in_target_range - trapezoid_profile)
            trapezoid_penalty = trapezoid_penalty_weight * trapezoid_mismatch.sum()
            reward -= trapezoid_penalty
        else:
            transmittance_mismatch = np.abs(reflection_in_target_range - desired_reflection)
            reward_penalty = penalty_target_range * transmittance_mismatch.sum()
            reward -= reward_penalty

        reward_outside = reward_outside_target * np.mean(reflection_outside_target)
        reward += reward_outside

    # Calculate reward for high reflection in intermediate regions
    for i in range(len(target_ranges) - 1):
        if target_ranges[i] is not None and target_ranges[i + 1] is not None:
            intermediate_range_start = target_ranges[i][1]
            intermediate_range_end = target_ranges[i + 1][0]
            idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
            reflection_in_intermediate_range = A[idx_intermediate]
            transition_reward = transition_reward_weight * (np.mean(reflection_in_intermediate_range) - high_reflection_value)
            reward += transition_reward

    gradient_penalty = np.sum(np.abs(np.diff(A))) * smoothness_weight
    reward -= gradient_penalty

    if baseline_reflection is not None:
        baseline_mismatch = np.abs(A - baseline_reflection)
        baseline_penalty = baseline_mismatch.sum() * baseline_weight
        reward -= baseline_penalty

    average_reward = reward_tracker.get_average_reward()
    reward_trend = reward_tracker.get_reward_trend()
    if reward_trend > 0 and average_reward != 0:
        reward *= 1 + reward_trend / average_reward

    return reward





def ppo_optimization_reward(R: np.ndarray, T: np.ndarray, A: np.ndarray, wavelength: np.array, target_ranges: list, desired_absorption: list, narrowbands: list) -> float:
    reward = 0.0
    penalty_target_range = 10  # Penalty for deviation from desired reflection within the target range
    reward_outside_target = 20  # Reward for high reflection outside the target range
    smoothness_weight = 0.1  # Weight for smoothness penalty
    # asymmetry_weight = 0.5  # Weight for asymmetry penalty
    transition_reward_weight = 8  # Weight for transition region reward
    high_reflection_value = 1.0  # Desired high reflection value in intermediate regions
    trapezoid_penalty_weight = 15  # Penalty weight for trapezoid profile mismatch

    for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands):
        if target_range is None:
            continue

        # Define the target wavelength range
        idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
        idx_outside_target = ~idx_target  # Indices outside the target range

        # Extract the reflection values
        reflection_in_target_range = A[idx_target]
        reflection_outside_target = A[idx_outside_target]

        if narrowband is not None and narrowband != 0:
            lower_start, lower_end = target_range
            upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
            upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

            target_wavelengths = wavelength[idx_target]
            trapezoid_profile_values = inverted_trapezoid_profile(
                target_wavelengths,
                lower_start, lower_end,
                upper_start, upper_end,
                desired_reflection, high_reflection_value
            )

            trapezoid_mismatch = np.abs(reflection_in_target_range - trapezoid_profile_values)
            trapezoid_penalty = trapezoid_penalty_weight * trapezoid_mismatch.sum()
            reward -= trapezoid_penalty
        else:
            # Calculate the penalty for the target range
            transmittance_mismatch = np.abs(reflection_in_target_range - desired_reflection)
            reward_penalty = penalty_target_range * transmittance_mismatch.sum()
            reward -= reward_penalty

        # Calculate the reward for reflection outside the target range
        reward_outside = reward_outside_target * np.mean(reflection_outside_target)
        reward += reward_outside

    # Add smoothness penalty based on the gradient
    gradient_penalty = np.sum(np.abs(np.diff(A))) * smoothness_weight
    reward -= gradient_penalty

    # Calculate reward for high reflection in intermediate regions
    for i in range(len(target_ranges) - 1):
        if target_ranges[i] is not None and target_ranges[i + 1] is not None:
            intermediate_range_start = target_ranges[i][1]
            intermediate_range_end = target_ranges[i + 1][0]
            idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
            reflection_in_intermediate_range = A[idx_intermediate]
            transition_reward = transition_reward_weight * (np.mean(reflection_in_intermediate_range) - high_reflection_value)
            reward += transition_reward

    return reward











def ddpg_reward_fun(R, T, A, wavelength, target_ranges, desired_absorption, previous_layers, previous_thicknesses, template=None, narrowbands=None):
    trapezoid_penalty_weight = 15
    high_reflection_value = 1.0
    penalty_target_range = 7

    if template is not None:
        mse = np.mean((A - template) ** 2)
        reward = -mse
        for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands):
            if target_range is None:
                continue
            idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
            reflection_in_target_range = A[idx_target]

            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, high_reflection_value
                )

                trapezoid_mismatch = np.abs(reflection_in_target_range - trapezoid_profile_values)
                trapezoid_penalty = trapezoid_penalty_weight * trapezoid_mismatch.sum()
                reward -= trapezoid_penalty
            else:
                transmittance_mismatch = np.abs(reflection_in_target_range - desired_reflection)
                reward_penalty = penalty_target_range * transmittance_mismatch.sum()
                reward -= reward_penalty
    else:
        reward = 0.0
        penalty_target_range = 10
        reward_outside_target = 10
        smoothness_weight = 0.1
        transition_reward_weight = 5  # Weight for transition region reward
        high_reflection_value = 1.0  # Desired high reflection value in intermediate regions
        trapezoid_penalty_weight = 15  # Penalty weight for trapezoid profile mismatch

        for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands):
            if target_range is None:
                continue

            idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
            idx_outside_target = ~idx_target

            reflection_in_target_range = A[idx_target]
            reflection_outside_target = A[idx_outside_target]

            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, high_reflection_value
                )

                trapezoid_mismatch = np.abs(reflection_in_target_range - trapezoid_profile_values)
                trapezoid_penalty = trapezoid_penalty_weight * trapezoid_mismatch.sum()
                reward -= trapezoid_penalty
            else:
                transmittance_mismatch = np.abs(reflection_in_target_range - desired_reflection)
                reward_penalty = penalty_target_range * transmittance_mismatch.sum()
                reward -= reward_penalty

            reward_outside = reward_outside_target * np.mean(reflection_outside_target)
            reward += reward_outside

        # Add smoothness penalty based on the gradient
        gradient_penalty = np.sum(np.abs(np.diff(A))) * smoothness_weight
        reward -= gradient_penalty

        # Calculate reward for high reflection in intermediate regions
        for i in range(len(target_ranges) - 1):
            if target_ranges[i] is not None and target_ranges[i + 1] is not None:
                intermediate_range_start = target_ranges[i][1]
                intermediate_range_end = target_ranges[i + 1][0]
                idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
                reflection_in_intermediate_range = A[idx_intermediate]
                transition_reward = transition_reward_weight * (np.mean(reflection_in_intermediate_range) - high_reflection_value)
                reward += transition_reward

    previous_thicknesses = np.array(previous_thicknesses, dtype=float)
    layer_change_penalty = 0.1 * len(set(previous_layers) - set(previous_layers))
    thickness_change_penalty = 0.1 * np.sum(np.abs(previous_thicknesses - previous_thicknesses))

    reward -= layer_change_penalty
    reward -= thickness_change_penalty

    novelty_reward = 0.1 * np.sum(np.abs(np.diff(A)))
    reward += novelty_reward

    return reward












def bandpass_reward_shaping(R, T, A, wavelength, target_ranges, reward_tracker, baseline_reflection=None):
    reward = 0.0
    reward_target_range = 10  # Reward for high reflection within the target range
    penalty_outside_target = 7  # Penalty for high reflection outside the target range
    smoothness_weight = 0.1  # Weight for smoothness penalty
    baseline_weight = 0.5  # Weight for baseline comparison penalty

    for target_range in target_ranges:
        if target_range is None:
            continue

        # Define the target wavelength range
        idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
        idx_outside_target = ~idx_target  # Indices outside the target range

        # Extract the reflection values
        reflection_in_target_range = A[idx_target]
        reflection_outside_target = A[idx_outside_target]

        # Calculate the reward for the target range
        reward_target = reward_target_range * reflection_in_target_range.sum()
        reward += reward_target

        # Calculate the penalty for reflection outside the target range
        transmittance_mismatch = np.abs(reflection_outside_target)
        reward_penalty = penalty_outside_target * transmittance_mismatch.sum()
        reward -= reward_penalty

    # Add smoothness penalty based on the gradient
    gradient_penalty = np.sum(np.abs(np.diff(A))) * smoothness_weight
    reward -= gradient_penalty

    # Add comparison to baseline if provided
    if baseline_reflection is not None:
        baseline_mismatch = np.abs(A - baseline_reflection)
        baseline_penalty = baseline_mismatch.sum() * baseline_weight
        reward -= baseline_penalty

    # Adjust the reward based on reward trends
    average_reward = reward_tracker.get_average_reward()
    reward_trend = reward_tracker.get_reward_trend()
    if reward_trend > 0 and average_reward != 0:  # Ensure non-zero division
        reward *= 1 + reward_trend / average_reward

    return reward




def ddpg_bandpass_reward_fun(R, T, A, wavelength, target_ranges, previous_layers, previous_thicknesses, template=None):
    reward = 0.0
    reward_target_range = 10  # Reward for high reflection within the target range
    penalty_outside_target = 7  # Penalty for high reflection outside the target range
    smoothness_weight = 0.1  # Weight for smoothness penalty

    for target_range in target_ranges:
        if target_range is None:
            continue

        # Define the target wavelength range
        idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
        idx_outside_target = ~idx_target  # Indices outside the target range

        # Extract the reflection values
        reflection_in_target_range = A[idx_target]
        reflection_outside_target = A[idx_outside_target]

        # Calculate the reward for the target range
        reward_target = reward_target_range * reflection_in_target_range.sum()
        reward += reward_target

        # Calculate the penalty for reflection outside the target range
        transmittance_mismatch = np.abs(reflection_outside_target)
        reward_penalty = penalty_outside_target * transmittance_mismatch.sum()
        reward -= reward_penalty

    # Add smoothness penalty based on the gradient
    gradient_penalty = np.sum(np.abs(np.diff(A))) * smoothness_weight
    reward -= gradient_penalty

    # Encourage changes to the layers and thicknesses
    layer_change_penalty = 0.1 * len(set(previous_layers) - set(previous_layers))
    thickness_change_penalty = 0.1 * np.sum(np.abs(np.array(previous_thicknesses) - np.array(previous_thicknesses)))

    reward -= layer_change_penalty
    reward -= thickness_change_penalty

    # Introduce a novelty term to encourage exploration
    novelty_reward = 0.1 * np.sum(np.abs(np.diff(A)))
    reward += novelty_reward

    return reward


def bandpass_ppo_optimization_reward(R: np.ndarray, T: np.ndarray, A: np.ndarray, wavelength: np.array, target_ranges: list) -> float:
    reward = 0.0
    reward_target_range = 10  # Reward for high reflection within the target range
    penalty_outside_target = 10  # Penalty for high reflection outside the target range

    for target_range in target_ranges:
        if target_range is None:
            continue

        # Define the target wavelength range
        idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
        idx_outside_target = ~idx_target  # Indices outside the target range

        # Extract the reflection values
        reflection_in_target_range = A[idx_target]
        reflection_outside_target = A[idx_outside_target]

        # Calculate the reward for high reflection within the target range
        reward_target = reward_target_range * reflection_in_target_range.sum()
        reward += reward_target

        # Calculate the penalty for high reflection outside the target range
        penalty_outside = penalty_outside_target * reflection_outside_target.sum()
        reward -= penalty_outside

    return reward

