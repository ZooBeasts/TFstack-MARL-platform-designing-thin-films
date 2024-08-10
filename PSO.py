import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os
from tmm_core import coh_tmm, coh_tmm_dispersion, n_sio2, n_tio2
from pyswarm import pso


# class TMM_sim:
#     def __init__(self, available_materials, substrate_materials, wavelengths, substrate_material, substrate_thickness):
#         self.wavelength = np.array(wavelengths)
#         self.substrate = substrate_material
#         self.substrate_thick = substrate_thickness
#         self.layers = []  # No predefined layers, will be generated dynamically
#         self.materials = [material_info['material'] for material_info in available_materials]
#         self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
#         self.nk_dict = self.load_materials(available_materials, substrate_materials)
#
#     def load_materials(self, available_materials, substrate_materials):
#         nk_dict = {}
#
#         for material_info in available_materials + substrate_materials:
#             material = material_info['material']
#             filename = material_info['refractive_index_file']
#             filepath = os.path.join("E:/Reinforcing/TMM_GNN/data", filename)
#             if not os.path.exists(filepath):
#                 raise FileNotFoundError(f"Could not find the file: {filepath}")
#             nk = pd.read_csv(filepath)
#             nk.dropna(inplace=True)
#             wl = nk['wl'].to_numpy()
#             index = (nk['n'] + nk['k'] * 1.j).to_numpy()
#             interp_fn = interp1d(wl, index, kind='quadratic', bounds_error=False, fill_value=0)
#             nk_dict[material] = interp_fn
#         return nk_dict
#
#     def calculate_mismatches(self, layers):
#         mismatches = []
#         for i in range(len(layers) - 1):
#             material1 = layers[i]
#             material2 = layers[i + 1]
#             n1 = self.nk_dict[material1](self.wavelength).real
#             n2 = self.nk_dict[material2](self.wavelength).real
#             mismatch = abs(n1 - n2)
#             mismatches.append({
#                 "interface": f"{material1}/{material2}",
#                 "interface_refractive_index_mismatch": [list(mismatch)]
#             })
#         return mismatches
#
#     def spectrum(self, layers, thicknesses, theta=0):
#         materials = layers
#         # Ensure thicknesses and materials match
#         thicknesses = [np.inf] + list(thicknesses) + [self.substrate_thick, np.inf]
#         R, T, A = [], [], []
#         degree = np.pi / 180
#         for lambda_vac in self.wavelength:
#             if self.substrate == 'Glass':
#                 n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [1.45, 1]
#             else:
#                 n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [
#                     self.nk_dict[self.substrate](lambda_vac), 1]
#
#             res = coh_tmm('s', n_list, thicknesses, theta * degree, lambda_vac)
#             R.append(res['R'])
#             T.append(res['T'])
#
#         R = np.array(R)
#         T = np.array(T)
#         A = 1 - R - T
#
#         if np.any(np.isnan(R)) or np.any(np.isnan(T)) or np.any(np.isnan(A)):
#             print(f"NaN values detected in spectrum calculation. R: {R}, T: {T}, A: {A}")
#
#         return R, T, A
#
#     def append_properties_to_output(self, layers, thicknesses):
#         R, T, A = self.spectrum(layers, thicknesses)
#         output_layers = []
#         mismatches = self.calculate_mismatches(layers)
#
#         for i, layer in enumerate(layers):
#             current_nk = self.nk_dict[layer](self.wavelength)
#             output_layer = {
#                 "material": layer,
#                 "thickness": thicknesses[i],
#             }
#             output_layers.append(output_layer)
#
#         final_output = {
#             "layers": output_layers,
#             "wavelengths": self.wavelength.tolist(),
#             "transmission": T.tolist(),
#             "reflection": R.tolist(),
#             "absorption": A.tolist(),
#         }
#         return final_output

class TMM_sim:
    def __init__(self, available_materials, substrate_materials, wavelengths, substrate_material, substrate_thickness):
        self.wavelength = np.array(wavelengths)
        self.substrate = substrate_material
        self.substrate_thick = substrate_thickness
        self.layers = []  # No predefined layers, will be generated dynamically
        self.materials = [material_info['material'] for material_info in available_materials]
        self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
        self.nk_dict = self.load_materials(available_materials, substrate_materials)
        self.dispersion_dict = self.load_dispersion_relations(available_materials, substrate_materials)

    def load_materials(self, available_materials, substrate_materials):
        nk_dict = {}
        for material_info in available_materials + substrate_materials:
            material = material_info['material']
            if 'refractive_index_file' in material_info:
                filename = material_info['refractive_index_file']
                filepath = os.path.join("E:/Reinforcing/TMM_GNN/data", filename)
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"Could not find the file: {filepath}")
                nk = pd.read_csv(filepath)
                nk.dropna(inplace=True)
                wl = nk['wl'].to_numpy()
                index = (nk['n'] + nk['k'] * 1.j).to_numpy()
                interp_fn = interp1d(wl, index, kind='quadratic', bounds_error=False, fill_value=0)
                nk_dict[material] = interp_fn
        return nk_dict

    def load_dispersion_relations(self, available_materials, substrate_materials):
        dispersion_dict = {}
        for material_info in available_materials + substrate_materials:
            material = material_info['material']
            if 'dispersion_function' in material_info:
                dispersion_dict[material] = material_info['dispersion_function']
        return dispersion_dict

    def get_refractive_index(self, material, wavelength):
        if material in self.dispersion_dict:
            return self.dispersion_dict[material](wavelength)
        elif material in self.nk_dict:
            return self.nk_dict[material](wavelength)
        else:
            raise ValueError(f"No refractive index data for material: {material}")

    def calculate_mismatches(self, layers):
        mismatches = []
        for i in range(len(layers) - 1):
            material1 = layers[i]
            material2 = layers[i + 1]
            n1 = self.get_refractive_index(material1, self.wavelength).real
            n2 = self.get_refractive_index(material2, self.wavelength).real
            mismatch = abs(n1 - n2)
            mismatches.append({
                "interface": f"{material1}/{material2}",
                "interface_refractive_index_mismatch": [list(mismatch)]
            })
        return mismatches

    def spectrum(self, layers, thicknesses, theta=0):
        materials = layers
        thicknesses = [np.inf] + thicknesses + [self.substrate_thick, np.inf]
        R, T, A = [], [], []
        degree = np.pi / 180
        for lambda_vac in self.wavelength:
            if self.substrate == 'Glass':
                n_list = [1] + [self.get_refractive_index(mat, lambda_vac) for mat in materials] + [1.45, 1]
            else:
                n_list = [1] + [self.get_refractive_index(mat, lambda_vac) for mat in materials] + [
                    self.get_refractive_index(self.substrate, lambda_vac), 1]

            res = coh_tmm('s', n_list, thicknesses, theta * degree, lambda_vac)
            R.append(res['R'])
            T.append(res['T'])

        R = np.array(R)
        T = np.array(T)
        A = 1 - R - T

        if np.any(np.isnan(R)) or np.any(np.isnan(T)) or np.any(np.isnan(A)):
            print(f"NaN values detected in spectrum calculation. R: {R}, T: {T}, A: {A}")

        return R, T, A

    def append_properties_to_output(self, layers, thicknesses):
        R, T, A = self.spectrum(layers, thicknesses)
        output_layers = []
        mismatches = self.calculate_mismatches(layers)

        for i, layer in enumerate(layers):
            current_nk = self.get_refractive_index(layer, self.wavelength)
            output_layer = {
                "material": layer,
                "thickness": thicknesses[i],
            }
            output_layers.append(output_layer)

        final_output = {
            "layers": output_layers,
            "wavelengths": self.wavelength.tolist(),
            "transmission": T.tolist(),
            "reflection": R.tolist(),
            "absorption": A.tolist(),
        }
        return final_output




# Define the objective function
def objective_function(thicknesses, simulator, target_wavelength_range, desired_transmission, weight_fluctuation=1.0,
                       weight_transmission=1.0):
    layers = ["TiO2" if i % 2 == 0 else "SiO2" for i in range(len(thicknesses))]
    R, T, A = simulator.spectrum(layers, thicknesses)

    # Calculate the standard deviation of the transmission in the target range (notch area)
    target_indices = np.where(
        (simulator.wavelength >= target_wavelength_range[0]) & (simulator.wavelength <= target_wavelength_range[1]))[0]
    transmission_in_notch = T[target_indices]
    fluctuation = np.std(transmission_in_notch - desired_transmission)

    # Calculate the average transmission outside the notch area
    non_target_indices = \
    np.where((simulator.wavelength < target_wavelength_range[0]) | (simulator.wavelength > target_wavelength_range[1]))[
        0]
    transmission_outside_notch = T[non_target_indices]
    average_transmission = np.mean(transmission_outside_notch)

    # Composite objective: Minimize fluctuation in the notch area and maximize transmission outside the notch area
    objective = weight_fluctuation * fluctuation - weight_transmission * average_transmission
    return objective

if __name__ == '__main__':
    available_materials = [
        {"material": "SiO2", "refractive_index_file": "SiO2.csv", "dispersion_function": n_sio2},
        {"material": "TiO2", "refractive_index_file": "TiO2.csv", "dispersion_function": n_tio2},
    ]
    substrate_materials = [{"material": "Glass", "refractive_index_file": "Glass.csv"}]
    wavelengths = np.linspace(400, 900, 51)  # From 400nm to 900nm, 501 points

    # Initialize the simulator
    simulator = TMM_sim(available_materials, substrate_materials, wavelengths, "Glass", 1e6)  # 500 nm thick Ag substrate

    # Load existing thickness data from the CSV file
    file_path = 'TIO2_SIO2_Template_650_700_2/ppo_optimized_episode_34_step_26.json.csv'
    thickness_data = pd.read_csv(file_path)
    existing_thicknesses = thickness_data['Thickness'].values
    layers = thickness_data['Material'].values

    # Define the PSO parameters
    num_layers = len(existing_thicknesses)
    lower_bounds = [10] * num_layers  # Minimum thickness of 5 nm for each layer
    upper_bounds = [400] * num_layers  # Maximum thickness of 400 nm for each layer
    target_wavelength_range = (650, 720)  # Notch area
    desired_transmission = 0  # Desired transmission in the notch area


    # Run PSO with updated objective function
    optimized_thicknesses, fopt = pso(objective_function, lower_bounds, upper_bounds,
                                      args=(simulator, target_wavelength_range, desired_transmission, 1.0, 1.0),
                                      swarmsize=100, maxiter=400, debug=True)

    # Print optimized thicknesses
    print("Optimized Layer Thicknesses:", optimized_thicknesses)

    # Calculate the final spectrum with optimized thicknesses
    properties = simulator.append_properties_to_output(layers, optimized_thicknesses)
    print("Final Transmission Spectrum:", properties['transmission'])

    # Save the results to CSV files
    # Save transmission, reflection, and absorption spectra
    df_spectrum = pd.DataFrame({
        'Wavelength': properties['wavelengths'],
        'Transmission': properties['transmission'],
        'Reflection': properties['reflection'],
        'Absorption': properties['absorption']
    })
    df_spectrum.to_csv('spectrum_results.csv', index=False)

    # Save layer materials and thicknesses
    df_layers = pd.DataFrame({
        'Material': [layer['material'] for layer in properties['layers']],
        'Thickness': [layer['thickness'] for layer in properties['layers']]
    })
    df_layers.to_csv('layer_results.csv', index=False)