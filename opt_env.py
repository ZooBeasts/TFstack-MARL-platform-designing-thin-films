
from scipy.optimize import minimize
import numpy as np
from utils import save_ppo_optimized_design, save_ppo_optimized_bgs_design
from joblib import Parallel, delayed

from pyswarm import pso
from scipy.optimize import NonlinearConstraint,Bounds



def inverted_trapezoid_profile(x, lower_start, lower_end, upper_start, upper_end, lower_value, upper_value):
    profile = np.ones_like(x) * upper_value
    profile[(x >= lower_start) & (x <= lower_end)] = lower_value
    slope1 = (upper_value - lower_value) / (lower_start - upper_start)
    slope2 = (lower_value - upper_value) / (upper_end - lower_end)
    profile[(x >= upper_start) & (x < lower_start)] = slope1 * (x[(x >= upper_start) & (x < lower_start)] - upper_start) + upper_value
    profile[(x > lower_end) & (x <= upper_end)] = slope2 * (x[(x > lower_end) & (x <= upper_end)] - lower_end) + lower_value
    return profile


def inverted_trapezoid_profile_v2(wavelengths, lower_start, lower_end, upper_start, upper_end, lower_value, upper_value):
    profile = np.ones_like(wavelengths) * upper_value
    profile[(wavelengths >= lower_start) & (wavelengths <= lower_end)] = lower_value
    profile[(wavelengths >= upper_start) & (wavelengths < lower_start)] = np.linspace(upper_value, lower_value, np.sum((wavelengths >= upper_start) & (wavelengths < lower_start)))
    profile[(wavelengths > lower_end) & (wavelengths <= upper_end)] = np.linspace(lower_value, upper_value, np.sum((wavelengths > lower_end) & (wavelengths <= upper_end)))
    return profile


def inverted_trapezoid_profile_v3(wavelengths, lower_start, lower_end, upper_start, upper_end, target_value,
                               outside_value):
    profile = np.full_like(wavelengths, outside_value, dtype=np.float64)
    inside_range = (wavelengths >= lower_start) & (wavelengths <= lower_end)
    transition_range1 = (wavelengths >= upper_start) & (wavelengths < lower_start)
    transition_range2 = (wavelengths > lower_end) & (wavelengths <= upper_end)

    profile[inside_range] = target_value
    profile[transition_range1] = np.interp(wavelengths[transition_range1], [upper_start, lower_start],
                                           [outside_value, target_value])
    profile[transition_range2] = np.interp(wavelengths[transition_range2], [lower_end, upper_end],
                                           [target_value, outside_value])

    return profile


def refine_design_with_trust_constr(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption, narrowbands, upper, lower):
    # Set initial thicknesses
    initial_thicknesses = np.array(thicknesses)

    # Set bounds for each thickness to ensure they remain positive
    bounds = [(lower, upper) for _ in range(len(initial_thicknesses))]

    def objective_function_parallel_v3(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption,
                                    narrowbands):
        _, _, A = simulator.spectrum(layers, thicknesses.tolist())

        def calculate_penalty(target_range, desired_reflection, narrowband):
            if target_range is not None:
                idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
                target_reflection = A[idx_target]

                if narrowband is not None and narrowband != 0:
                    lower_start, lower_end = target_range
                    upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                    upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                    target_wavelengths = wavelength[idx_target]
                    trapezoid_profile_values = inverted_trapezoid_profile_v3(
                        target_wavelengths,
                        lower_start, lower_end,
                        upper_start, upper_end,
                        desired_reflection, 1.0
                    )

                    trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                    return np.mean(trapezoid_mismatch)
                else:
                    transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                    return np.mean(transmittance_mismatch)
            return 0

        penalties = (
            calculate_penalty(target_range, desired_reflection, narrowband) for
            target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
        )

        # Penalty for reflection within the target range
        target_penalty_weight = 10  # Weight for the penalty within the target range
        target_penalty = sum(
            target_penalty_weight * np.mean(A[(wavelength >= tr[0]) & (wavelength <= tr[1])]) for tr in target_ranges if
            tr is not None)

        # Reward the reflection outside the target ranges
        outside_reward_weight = 30  # Weight for the reward outside the target ranges
        idx_outside = np.ones_like(wavelength, dtype=bool)
        for tr in target_ranges:
            if tr is not None:
                idx_outside &= (wavelength < tr[0]) | (wavelength > tr[1])
        reflection_outside_range = A[idx_outside]
        outside_reward = outside_reward_weight * np.mean(reflection_outside_range)

        # Combine the penalties and rewards
        total_penalty = sum(penalties) + target_penalty - outside_reward

        return total_penalty

    # Use trust-constr method which supports bounds and constraints
    result = minimize(objective_function_parallel_v3, initial_thicknesses,
                      args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
                      method='trust-constr', bounds=bounds, options={'disp': True})

    # Ensure result thicknesses are correctly formatted
    optimized_thicknesses = result.x.tolist()
    if len(layers) != len(optimized_thicknesses):
        raise ValueError("Mismatch between optimized layers and thicknesses lengths.")

    return optimized_thicknesses



# def refine_design_with_trust_constr(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption, narrowbands, upper, lower):
#     # Set initial thicknesses
#     initial_thicknesses = np.array(thicknesses)
#
#     # Set bounds for each thickness to ensure they remain positive
#     bounds = [(lower, upper) for _ in range(len(initial_thicknesses))]
#
#     def objective_function_parallel_v3(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption,
#                                     narrowbands):
#         R, _, _ = simulator.spectrum(layers, thicknesses.tolist())
#
#         def calculate_penalty(target_range, desired_reflection, narrowband):
#             if target_range is not None:
#                 idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
#                 target_reflection = R[idx_target]
#
#                 if narrowband is not None and narrowband != 0:
#                     lower_start, lower_end = target_range
#                     upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
#                     upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2
#
#                     target_wavelengths = wavelength[idx_target]
#                     trapezoid_profile_values = inverted_trapezoid_profile_v3(
#                         target_wavelengths,
#                         lower_start, lower_end,
#                         upper_start, upper_end,
#                         desired_reflection, 1.0
#                     )
#
#                     trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
#                     return np.mean(trapezoid_mismatch)
#                 else:
#                     transmittance_mismatch = np.abs(target_reflection - desired_reflection)
#                     return np.mean(transmittance_mismatch)
#             return 0
#
#         # penalties = Parallel(n_jobs=-1)(
#         #     delayed(calculate_penalty)(target_range, desired_reflection, narrowband) for
#         #     target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
#         # )
#         penalties = (
#             (calculate_penalty)(target_range, desired_reflection, narrowband) for
#             target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
#         )
#
#         # Penalty for reflection within the target range
#         target_penalty_weight = 10  # Weight for the penalty within the target range
#         target_penalty = sum(
#             target_penalty_weight * np.mean(R[(wavelength >= tr[0]) & (wavelength <= tr[1])]) for tr in target_ranges if
#             tr is not None)
#
#         # Reward the reflection outside the target ranges
#         outside_reward_weight = 1  # Weight for the reward outside the target ranges
#         idx_outside = np.ones_like(wavelength, dtype=bool)
#         for tr in target_ranges:
#             if tr is not None:
#                 idx_outside &= (wavelength < tr[0]) | (wavelength > tr[1])
#         reflection_outside_range = R[idx_outside]
#         outside_reward = outside_reward_weight * np.mean(reflection_outside_range)
#
#         # Combine the penalties and rewards
#         total_penalty = sum(penalties) + target_penalty - outside_reward
#
#         return total_penalty
#
#     # Use trust-constr method which supports bounds and constraints
#     result = minimize(objective_function_parallel_v3, initial_thicknesses,
#                       args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
#                       method='trust-constr', bounds=bounds, options={'disp': True})
#
#     # Ensure result thicknesses are correctly formatted
#     optimized_thicknesses = result.x.tolist()
#     if len(layers) != len(optimized_thicknesses):
#         raise ValueError("Mismatch between optimized layers and thicknesses lengths.")
#
#     return optimized_thicknesses

def refine_design_with_bfgs(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption,
                            narrowbands, upper, lower):
    # Set initial thicknesses
    initial_thicknesses = np.array(thicknesses)

    # Set bounds for each thickness to ensure they remain positive
    bounds = [(lower, upper) for _ in range(len(initial_thicknesses))]

    def objective_function_parallel(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption,
                                    narrowbands):
        _, _, A = simulator.spectrum(layers, thicknesses.tolist())

        def calculate_penalty(target_range, desired_reflection, narrowband):
            if target_range is not None:
                idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
                target_reflection = A[idx_target]

                if narrowband is not None and narrowband != 0:
                    lower_start, lower_end = target_range
                    upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                    upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                    target_wavelengths = wavelength[idx_target]
                    trapezoid_profile_values = inverted_trapezoid_profile(
                        target_wavelengths,
                        lower_start, lower_end,
                        upper_start, upper_end,
                        desired_reflection, 1.0
                    )

                    trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                    return np.mean(trapezoid_mismatch)
                else:
                    transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                    return np.mean(transmittance_mismatch)
            return 0

        # penalties = Parallel(n_jobs=-1)(
        #     delayed(calculate_penalty)(target_range, desired_reflection, narrowband) for
        #     target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
        # )
        penalties = (
            calculate_penalty(target_range, desired_reflection, narrowband) for
            target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
        )

        total_penalty = sum(penalties)

        # Introduce a reward for high reflection values in intermediate regions
        transition_reward_weight = 5  # Weight for transition region reward
        high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

        for i in range(len(target_ranges) - 1):
            if target_ranges[i] is not None and target_ranges[i + 1] is not None:
                intermediate_range_start = target_ranges[i][1]
                intermediate_range_end = target_ranges[i + 1][0]
                idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
                reflection_in_intermediate_range = A[idx_intermediate]
                transition_reward = transition_reward_weight * np.mean(
                    reflection_in_intermediate_range - high_reflection_value)
                total_penalty -= transition_reward

        return total_penalty

    # Use L-BFGS-B method which supports bounds
    result = minimize(objective_function_parallel, initial_thicknesses,
                      args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
                      method='L-BFGS-B', bounds=bounds, options={'disp': True})

    # Ensure result thicknesses are correctly formatted
    optimized_thicknesses = result.x.tolist()
    if len(layers) != len(optimized_thicknesses):
        raise ValueError("Mismatch between optimized layers and thicknesses lengths.")

    return optimized_thicknesses





def objective_function_parallel_v2(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands):
    _, _, A = simulator.spectrum(layers, thicknesses.tolist())

    def calculate_penalty(target_range, desired_reflection, narrowband):
        if target_range is not None:
            idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
            target_reflection = A[idx_target]

            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile_v2(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, 1.0
                )

                trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                return np.mean(trapezoid_mismatch)
            else:
                transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                return np.mean(transmittance_mismatch)
        return 0

    penalties = Parallel(n_jobs=-1)(
        delayed(calculate_penalty)(target_range, desired_reflection, narrowband) for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
    )

    total_penalty = sum(penalties)

    # Introduce a reward for high reflection values in intermediate regions
    transition_reward_weight = 5  # Weight for transition region reward
    high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

    for i in range(len(target_ranges) - 1):
        if target_ranges[i] is not None and target_ranges[i + 1] is not None:
            intermediate_range_start = target_ranges[i][1]
            intermediate_range_end = target_ranges[i + 1][0]
            idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
            reflection_in_intermediate_range = A[idx_intermediate]
            transition_reward = transition_reward_weight * np.mean(reflection_in_intermediate_range - high_reflection_value)
            total_penalty -= transition_reward

    return total_penalty




def refine_design_with_bfgs_v2(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption, narrowbands):
    # Set initial thicknesses
    initial_thicknesses = np.array(thicknesses)

    # Set bounds for each thickness to ensure they remain positive
    bounds = [(5, 550) for _ in range(len(initial_thicknesses))]

    # Use L-BFGS-B method which supports bounds and constraints
    result = minimize(objective_function_parallel_v2, initial_thicknesses,
                      args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
                      method='L-BFGS-B', bounds=bounds, options={'disp': True})

    # Ensure result thicknesses are correctly formatted
    optimized_thicknesses = result.x.tolist()
    if len(layers) != len(optimized_thicknesses):
        raise ValueError("Mismatch between optimized layers and thicknesses lengths.")





def refine_design_with_bfgs_v3(layers, thicknesses, simulator,
                               wavelength, target_ranges, upper, lower):
    # Set initial thicknesses
    initial_thicknesses = np.array(thicknesses)

    # Set bounds for each thickness to ensure they remain positive
    bounds = [(lower, upper) for _ in range(len(initial_thicknesses))]

    def objective_function_parallel_v2(thicknesses, layers, simulator, wavelength, target_ranges):
        _, _, A = simulator.spectrum(layers, thicknesses.tolist())

        def calculate_penalty(target_range):
            if target_range is not None:
                target_idx = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
                target_reflection = A[target_idx]
                return np.mean(target_reflection)
            return 0

        # Use joblib to parallelize the penalty calculation
        total_penalty = Parallel(n_jobs=-1)(delayed(calculate_penalty)(target_range) for target_range in target_ranges)

        return sum(total_penalty)

    # Use L-BFGS-B method which supports bounds
    result = minimize(objective_function_parallel_v2, initial_thicknesses,
                      args=(layers, simulator, wavelength, target_ranges),
                      method='L-BFGS-B', bounds=bounds, options={'disp': True})

    # Ensure result thicknesses are correctly formatted
    optimized_thicknesses = result.x.tolist()
    if len(layers) != len(optimized_thicknesses):
        raise ValueError("Mismatch between optimized layers and thicknesses lengths.")

    return optimized_thicknesses


















def pso_objective_function(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands):
    _, _, A = simulator.spectrum(layers, thicknesses.tolist())

    def calculate_penalty(target_range, desired_reflection, narrowband):
        if target_range is not None:
            idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
            target_reflection = A[idx_target]

            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, 1.0
                )

                trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                return np.mean(trapezoid_mismatch)
            else:
                transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                return np.mean(transmittance_mismatch)
        return 0

    penalties = Parallel(n_jobs=-1)(
        delayed(calculate_penalty)(target_range, desired_reflection, narrowband) for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
    )

    total_penalty = sum(penalties)

    # Introduce a reward for high reflection values in intermediate regions
    transition_reward_weight = 5  # Weight for transition region reward
    high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

    for i in range(len(target_ranges) - 1):
        if target_ranges[i] is not None and target_ranges[i + 1] is not None:
            intermediate_range_start = target_ranges[i][1]
            intermediate_range_end = target_ranges[i + 1][0]
            idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
            reflection_in_intermediate_range = A[idx_intermediate]
            transition_reward = transition_reward_weight * np.mean(reflection_in_intermediate_range - high_reflection_value)
            total_penalty -= transition_reward

    return total_penalty


def refine_design_with_pso_bfgs_hybrid(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption, narrowbands):
    # Define the bounds for the thicknesses
    bounds = [(5, 550) for _ in range(len(thicknesses))]

    # Run PSO first with parallelization
    optimized_thicknesses_pso, _ = pso(
        pso_objective_function,
        lb=[b[0] for b in bounds],
        ub=[b[1] for b in bounds],
        args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
        swarmsize=200,
        maxiter=150,
        # processes=-1  # Use all available cores
    )

    # Define the objective function for BFGS with parallelization
    def objective_function_parallel(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands):
        _, _, A = simulator.spectrum(layers, thicknesses.tolist())

        def calculate_penalty(target_range, desired_reflection, narrowband):
            if target_range is not None:
                idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
                target_reflection = A[idx_target]

                if narrowband is not None and narrowband != 0:
                    lower_start, lower_end = target_range
                    upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                    upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                    target_wavelengths = wavelength[idx_target]
                    trapezoid_profile_values = inverted_trapezoid_profile(
                        target_wavelengths,
                        lower_start, lower_end,
                        upper_start, upper_end,
                        desired_reflection, 1.0
                    )

                    trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                    return np.mean(trapezoid_mismatch)
                else:
                    transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                    return np.mean(transmittance_mismatch)
            return 0

        penalties = Parallel(n_jobs=-1)(
            delayed(calculate_penalty)(target_range, desired_reflection, narrowband) for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
        )

        total_penalty = sum(penalties)

        # Introduce a reward for high reflection values in intermediate regions
        transition_reward_weight = 5  # Weight for transition region reward
        high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

        for i in range(len(target_ranges) - 1):
            if target_ranges[i] is not None and target_ranges[i + 1] is not None:
                intermediate_range_start = target_ranges[i][1]
                intermediate_range_end = target_ranges[i + 1][0]
                idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
                reflection_in_intermediate_range = A[idx_intermediate]
                transition_reward = transition_reward_weight * np.mean(reflection_in_intermediate_range - high_reflection_value)
                total_penalty -= transition_reward

        return total_penalty

    # Use BFGS to refine the PSO result
    result = minimize(objective_function_parallel, optimized_thicknesses_pso,
                      args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands),
                      method='L-BFGS-B', bounds=bounds, options={'disp': True})

    optimized_thicknesses = result.x.tolist()
    if len(layers) != len(optimized_thicknesses):
        raise ValueError("Mismatch between optimized layers and thicknesses lengths.")

    return optimized_thicknesses










def pso_objective_function1(thicknesses, layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands, template):
    _, _, A = simulator.spectrum(layers, thicknesses.tolist())

    def calculate_penalty(target_range, desired_reflection, narrowband):
        if target_range is not None:
            idx_target = (wavelength >= target_range[0]) & (wavelength <= target_range[1])
            target_reflection = A[idx_target]

            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, 1.0
                )

                trapezoid_mismatch = np.abs(target_reflection - trapezoid_profile_values)
                return np.mean(trapezoid_mismatch)
            else:
                transmittance_mismatch = np.abs(target_reflection - desired_reflection)
                return np.mean(transmittance_mismatch)
        return 0

    penalties = [
        calculate_penalty(target_range, desired_reflection, narrowband) for
        target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands)
    ]

    total_penalty = sum(penalties)

    # Introduce a reward for high reflection values in intermediate regions
    transition_reward_weight = 8  # Weight for transition region reward
    high_reflection_value = 1.0  # Desired high reflection value in intermediate regions

    for i in range(len(target_ranges) - 1):
        if target_ranges[i] is not None and target_ranges[i + 1] is not None:
            intermediate_range_start = target_ranges[i][1]
            intermediate_range_end = target_ranges[i + 1][0]
            idx_intermediate = (wavelength >= intermediate_range_start) & (wavelength <= intermediate_range_end)
            reflection_in_intermediate_range = A[idx_intermediate]
            transition_reward = transition_reward_weight * np.mean(
                reflection_in_intermediate_range - high_reflection_value)
            total_penalty -= transition_reward

    # Add template deviation penalty
    if template is not None:
        template_mismatch = np.mean((A - template) ** 2)
        total_penalty += template_mismatch

    # Add penalties for deviations from the target wavelength range
    for target_range in target_ranges:
        if target_range is not None:
            lower_start, lower_end = target_range
            idx_outside_target = (wavelength < lower_start) | (wavelength > lower_end)
            reflection_outside_target = A[idx_outside_target]
            outside_target_penalty = np.mean(reflection_outside_target) * 10  # Arbitrary penalty weight
            total_penalty += outside_target_penalty

    return total_penalty

def refine_design_with_pso(layers, thicknesses, simulator, wavelength, target_ranges, desired_absorption, narrowbands, template):
    bounds = [(35, 180) for _ in range(len(thicknesses))]

    optimized_thicknesses, _ = pso(
        pso_objective_function1,
        lb=[b[0] for b in bounds],
        ub=[b[1] for b in bounds],
        args=(layers, simulator, wavelength, target_ranges, desired_absorption, narrowbands, template),
        swarmsize=400,
        maxiter=500
    )

    return optimized_thicknesses.tolist()





def ppo_optimization_phase(ppo_agent, env, shared_memory, isolated_memory, simulator,
                           target_ranges, desired_absorption,
                           narrowbands, template=None, alpha=0.5,
                           episode=None, output_path=None, upper=None, lower=None):

    def design_within_target_range(A, target_ranges, desired_absorption, simulator, narrowbands):
        for target_range, desired_reflection, narrowband in zip(target_ranges, desired_absorption, narrowbands):
            if target_range is None:
                continue
            idx_target = (simulator.wavelength >= target_range[0]) & (simulator.wavelength <= target_range[1])
            if narrowband is not None and narrowband != 0:
                lower_start, lower_end = target_range
                upper_start = lower_start - (narrowband - (lower_end - lower_start)) / 2
                upper_end = lower_end + (narrowband - (lower_end - lower_start)) / 2

                target_wavelengths = simulator.wavelength[idx_target]
                trapezoid_profile_values = inverted_trapezoid_profile(
                    target_wavelengths,
                    lower_start, lower_end,
                    upper_start, upper_end,
                    desired_reflection, 1.0
                )

                if np.all(np.abs(A[idx_target] - trapezoid_profile_values) <= 0.05):  # Tolerance for matching desired reflection
                    return True
            else:
                if np.all(np.abs(A[idx_target] - desired_reflection) <= 0.1):  # Tolerance for matching desired reflection
                    return True
        return False

    if template is not None:
        designs = shared_memory.retrieve()
        for state, layers, thicknesses, reward in designs:
            similarity = calculate_similarity(layers, thicknesses, template, simulator, alpha=alpha)
            if similarity >= 0.75:
                env.layers = layers
                env.thicknesses = thicknesses
                env.current_merit = reward

                done = False
                optimization_step_count = 0
                while not done and optimization_step_count < env.max_layers:
                    actions = ppo_agent.act(state, env.layers, env.min_layers, env.num_materials)
                    next_state, reward, done, _ = env.step(actions[0])
                    ppo_agent.remember(state, actions[0], reward, next_state, done)
                    state = next_state
                    optimization_step_count += 1

                    print(f"PPO Optimization share Step {optimization_step_count}")
                    print(f"Reward: {reward}")

                optimized_thicknesses = refine_design_with_pso(env.layers, env.thicknesses, simulator, simulator.wavelength, target_ranges, desired_absorption, narrowbands,template)
                env.thicknesses = optimized_thicknesses

                _, _, A = simulator.spectrum(env.layers, env.thicknesses)
                if design_within_target_range(A, target_ranges, desired_absorption, simulator, narrowbands):
                    save_ppo_optimized_design(simulator, env.layers, env.thicknesses, episode, optimization_step_count, output_path)
                    shared_memory.store((state, env.layers, env.thicknesses, reward))
                    if np.mean((A - np.array(template)) ** 2) >= 0.85:
                        isolated_memory.store((state, env.layers, env.thicknesses, reward))

    else:
        designs = isolated_memory.retrieve() + shared_memory.retrieve()
        for state, layers, thicknesses, reward in designs:
            env.layers = layers
            env.thicknesses = thicknesses
            env.current_merit = reward

            done = False
            optimization_step_count = 0
            while not done and optimization_step_count < env.max_layers:
                actions = ppo_agent.act(state, env.layers, env.min_layers, env.num_materials)
                next_state, reward, done, _ = env.step(actions[0])
                ppo_agent.remember(state, actions[0], reward, next_state, done)
                state = next_state
                optimization_step_count += 1

                print(f"PPO Optimization Isolated/Shared Step {optimization_step_count}")
                print(f"Reward: {reward}")

                if all(nb == 0 for nb in narrowbands) and all(dt == 0 for dt in desired_absorption):
                    optimized_thicknesses = refine_design_with_trust_constr(env.layers, env.thicknesses, simulator,
                                                                    simulator.wavelength, target_ranges, desired_absorption,
                                                                    narrowbands, upper, lower)


                elif all(nb == 0 for nb in narrowbands) and desired_absorption:
                    optimized_thicknesses = refine_design_with_trust_constr(env.layers, env.thicknesses, simulator,
                                                                    simulator.wavelength, target_ranges, desired_absorption,
                                                                    narrowbands, upper, lower)

                elif narrowbands and all(nb == 0 for nb in desired_absorption):
                    optimized_thicknesses = refine_design_with_bfgs(env.layers, env.thicknesses, simulator,
                                                                    simulator.wavelength, target_ranges, desired_absorption,
                                                                    narrowbands, upper, lower)

                else:
                    optimized_thicknesses = refine_design_with_pso_bfgs_hybrid(env.layers, env.thicknesses, simulator,
                                                                               simulator.wavelength, target_ranges,
                                                                               desired_absorption, narrowbands)

                env.thicknesses = optimized_thicknesses

                _, _, A = simulator.spectrum(env.layers, env.thicknesses)
                if design_within_target_range(A, target_ranges, desired_absorption, simulator, narrowbands):
                    try:
                        save_ppo_optimized_bgs_design(simulator, env.layers, env.thicknesses, episode,
                                                      optimization_step_count, output_path)
                        shared_memory.store((state, env.layers, env.thicknesses, reward))
                    except Exception as e:
                        print(f"Error saving design: {e}")








def calculate_similarity(layers, thicknesses, template, simulator, alpha=0.5):
    _, _, A = simulator.spectrum(layers, thicknesses)
    mse = np.mean((A - np.array(template)) ** 2)
    cosine_similarity = np.dot(A, np.array(template)) / (np.linalg.norm(A) * np.linalg.norm(np.array(template)))

    normalized_cosine_similarity = (cosine_similarity + 1) / 2

    hybrid_similarity = alpha * normalized_cosine_similarity + (1 - alpha) * (1 - mse)
    return hybrid_similarity




def validate_layers_thicknesses(layers, thicknesses):
    if len(layers) != len(thicknesses):
        raise ValueError("Mismatch between layers and thicknesses lengths.")


def final_validation(env, shared_memory, isolated_memory, threshold=0.5):
    designs = shared_memory.retrieve() + isolated_memory.retrieve()

    for state, layers, thicknesses, reward in designs:
        env.layers = layers
        env.thicknesses = thicknesses
        env.current_merit = reward

        # Debugging prints
        print(f"Final Validation")
        print(f"Layers: {layers}")
        print(f"Thicknesses: {thicknesses}")

        validate_layers_thicknesses(layers, thicknesses)  # Validate lengths

        R, T, A = env.simulator.spectrum(layers, thicknesses)
        final_reward = env.calculate_reward(R, T, A)

        # Debugging print
        print(f"Reward: {final_reward}")

        if final_reward > threshold:
            prioritized_experience_replay(ppo_agent, state, layers, thicknesses, final_reward)
def prioritized_experience_replay(agent, state, layers, thicknesses, reward):
    agent.memory.append((state, layers, thicknesses, reward))
    agent.memory = sorted(agent.memory, key=lambda x: x[3], reverse=True)
    if len(agent.memory) > agent.memory_size:
        agent.memory = agent.memory[:agent.memory_size]


