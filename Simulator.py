import os
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tmm_core import coh_tmm
import matplotlib.pyplot as plt


# class TMM_sim:
#     def __init__(self, available_materials, wavelengths, substrate_materials, substrate_thickness):
#         self.wavelength = np.array(wavelengths)
#         self.substrate = substrate_materials
#         self.substrate_thick = substrate_thickness
#         self.layers = []  # No predefined layers, will be generated dynamically
#         self.materials = [material_info['material'] for material_info in available_materials]
#         self.materials_idx = {material: idx for idx, material in enumerate(self.materials)}
#         self.nk_dict = self.load_materials(available_materials, substrate_materials)
#
#     def load_materials(self, available_materials, substrate_materials):
#         nk_dict = {}
#
#         for material_info in available_materials, substrate_materials:
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
#         # thicknesses = list(thicknesses)
#         thicknesses = [np.inf] + thicknesses + [self.substrate_thick, np.inf]
#         R, T, A = [], [], []
#         degree = np.pi / 180
#         for lambda_vac in self.wavelength:
#             if self.substrate == 'Glass':
#                 n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [1.45, 1]
#             else:
#                 n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [self.nk_dict[self.substrate](lambda_vac), 1]
#
#             # n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [self.nk_dict[self.substrate](lambda_vac), 1]
#             # if len(n_list) != len(thicknesses) + 2:
#             #     print(f"Length of n_list: {len(n_list)}, Length of thicknesses: {len(thicknesses)}")
#             #     raise ValueError("Mismatch between n_list and thicknesses lengths.")
#             res = coh_tmm('s', n_list, thicknesses, theta * degree, lambda_vac)
#             R.append(res['R'])
#             T.append(res['T'])
#
#         R = np.array(R)
#         T = np.array(T)
#         A = 1 - R - T
#
#         # Check for NaN values
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
#                 # "n_values": current_nk.real.tolist(),
#                 # "k_values": current_nk.imag.tolist(),
#             }
#             output_layers.append(output_layer)
#
#         final_output = {
#             "layers": output_layers,
#             "wavelengths": self.wavelength.tolist(),
#             "transmission": T.tolist(),
#             "reflection": R.tolist(),
#             "absorption": A.tolist(),
#             # "mismatch": mismatches
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

    def load_materials(self, available_materials, substrate_materials):
        nk_dict = {}

        for material_info in available_materials + substrate_materials:
            material = material_info['material']
            filename = material_info['refractive_index_file']
            filepath = os.path.join("data", filename)
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Could not find the file: {filepath}")
            nk = pd.read_csv(filepath)
            nk.dropna(inplace=True)
            wl = nk['wl'].to_numpy()
            index = (nk['n'] + nk['k'] * 1.j).to_numpy()
            interp_fn = interp1d(wl, index, kind='quadratic', bounds_error=False, fill_value=0)
            nk_dict[material] = interp_fn
        return nk_dict

    # def calculate_mismatches(self, layers):
    #     mismatches = []
    #     for i in range(len(layers) - 1):
    #         material1 = layers[i]
    #         material2 = layers[i + 1]
    #         n1 = self.nk_dict[material1](self.wavelength).real
    #         n2 = self.nk_dict[material2](self.wavelength).real
    #         mismatch = abs(n1 - n2)
    #         mismatches.append({
    #             "interface": f"{material1}/{material2}",
    #             "interface_refractive_index_mismatch": [list(mismatch)]
    #         })
    #     return mismatches

    def spectrum(self, layers, thicknesses, theta=0):
        materials = layers
        thicknesses = [np.inf] + thicknesses + [self.substrate_thick, np.inf]
        R, T, A = [], [], []
        degree = np.pi / 180
        for lambda_vac in self.wavelength:
            if self.substrate == 'Glass':
                n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [1.45, 1]
            else:
                n_list = [1] + [self.nk_dict[mat](lambda_vac) for mat in materials] + [
                    self.nk_dict[self.substrate](lambda_vac), 1]

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
        # mismatches = self.calculate_mismatches(layers)

        for i, layer in enumerate(layers):
            current_nk = self.nk_dict[layer](self.wavelength)
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


if __name__ == '__main__':
    available_materials = [

        {"material": "TiO2", "refractive_index_file": "TiO2.csv"},
        {"material": "SiO2", "refractive_index_file": "SiO2.csv"},
        {"material": "Fe2O3", "refractive_index_file": "Fe2O3.csv"},
        {"material": "Al2O3", "refractive_index_file": "Al2O3.csv"},
        {"material": "Ge", "refractive_index_file": "Ge.csv"},
        {"material": "HfO2", "refractive_index_file": "HfO2.csv"},
        {"material": "MgF2", "refractive_index_file": "MgF2.csv"},
        {"material": "Si", "refractive_index_file": "Si.csv"},
        {"material": "Cr", "refractive_index_file": "Cr.csv"},
        {"material": "Ti", "refractive_index_file": "Ti.csv"},
        {"material": "Ni", "refractive_index_file": "Ni.csv"},
        {"material": "Al", "refractive_index_file": "Al.csv"},
        {"material": "Ag", "refractive_index_file": "Ag.csv"},
        {"material": "Au", "refractive_index_file": "Au.csv"}

    ]

    substrate_materials = [{"material": "Glass", "refractive_index_file": "Glass.csv"}]
    # Wavelengths (in nanometers)
    # wavelengths = np.linspace(400, 710, 30)  # From 400nm to 700nm, 301 points
    wavelengths = np.arange(450, 1550, 1)  # From 400nm to 700nm, 301 points

    # Initialize the simulator
    simulator = TMM_sim(available_materials, substrate_materials, wavelengths, "Glass",
                        500)  # 500 nm thick glass substrate

    # layers = ['MgF2','TiO2','Si','Ge','Cr','Cr','Si','Cr','Cr','Ge','Si','MgF2','Si','Ge']
    # thicknesses = [123,39,20,17,26,37,17,25,43,26,22,46,39,29]

    # layers = ['MgF2', 'TiO2', 'Si', 'Ge', 'Cr', 'Cr', 'Si', 'Cr', 'Cr', 'Ge', 'Si', 'MgF2', 'Si', 'Ge']
    # thicknesses = [79, 33, 18, 16, 41, 52, 15, 29, 47, 26, 22, 46, 39,27]

    # layers = ['MgF2', 'TiO2', 'Si', 'Ge', 'Cr', 'Si', 'Cr', 'MgF2', 'MgF2', 'Ge', 'Ge', 'MgF2', 'Cr']
    # thicknesses = [90, 34, 23, 16, 29, 15, 43, 46, 42, 36, 24,81, 43]
    # data_1 = pd.read_csv('ppod.csv',header=0)
    # layers = data_1['Material'].to_list()
    # thicknesses = data_1['Thickness'].to_list()
    layers = ['SiO2','Ag','Ge','SiO2']
    thicknesses = [30, 7, 0.5, 30]

    # layers = ['Fe2O3',
    #           'Ni',
    #           'Si',
    #           'Au',
    #           'HfO2',
    #           'TiO2',
    #           'Al2O3',
    #           'Cr',
    #           'Ni',
    #           'Al',
    #           'Ge',
    #           'Cr',
    #           'Al2O3',
    #           'HfO2',
    #           'Ni',
    #           ]
    #
    # thicknesses = [
    #     332,
    #     25,
    #     105,
    #     146,
    #     151,
    #     165,
    #     127,
    #     30,
    #     227,
    #     129,
    #     212,
    #     36,
    #     81,
    #     163,
    #     324,
    #
    # ]

    # layers = ['MgF2', 'TiO2', 'MgF2', 'Si', 'TiO2', 'Si', 'Ge', 'Si', 'Cr', 'Ge', 'TiO2', 'Cr','TiO2', 'Cr']
    # thicknesses = [123,32,21,15,15,15,15,15,17,15,33,29,81,116]

    # Calculate the spectrum
    try:
        properties = simulator.append_properties_to_output(layers, thicknesses)
        # print("Calculated Optical Properties:")
        # print("Reflection:", properties['reflection'])
        # print("Transmission:", properties['transmission'])
        # print("Mean Transmission:", np.mean(properties['transmission']))
        # print("Absorption:", properties['absorption'])
        # plt.plot(wavelengths, properties['reflection'], label='Reflection')
        plt.plot(wavelengths, properties['transmission'], label='Transmission')
        # print('avg absorption:', np.mean(properties['absorption']))
        # avg_absorption = np.mean(properties['absorption'])
        # plt.plot(wavelengths, properties['absorption'], 'r', label='Absorption')
        plt.xlabel("Wavelength (nm)")
        # plt.plot(wavelengths, properties['reflection'],'b', label='Reflection')
        # plt.plot(wavelengths, properties['transmission'],'k',label = 'Transmission')
        # plt.ylabel("Reflection")
        # plt.ylabel("Absorption")
        plt.ylabel("Transmission")

        # plt.ylabel("R/T/A")
        plt.grid()
        plt.ylim([0,1])
        plt.legend()
        # plt.title(f"Reflection Spectrum")
        # plt.title("Absorption Spectrum")
        plt.title("Transmission Spectrum")
        # plt.title(f"Absorption Spectrum, avg absorption: {avg_absorption:.5f}")
        # plt.title("Absorption/Reflection/Transmission Spectrum")
        plt.tight_layout()
        # plt.savefig('absorption_spectrum4.png',dpi=600)
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")
