import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
# import time

# plt.switch_backend('Agg')
#
# input_folder = "narrow_100_test_v1"
# output_folder = "narrow_100_test_v1_1"
# os.makedirs(output_folder, exist_ok=True)
#
# class JSONPlotHandler(FileSystemEventHandler):
#     def on_modified(self, event):
#         if event.is_directory or not event.src_path.endswith(".json"):
#             return
#
#         filepath = event.src_path
#         filename = os.path.basename(filepath)  # Extract the filename
#
#         try:
#             with open(filepath, "r") as file:
#                 data = json.load(file)
#
#                 new_csv = []
#                 for layer in data['layers']:
#                     material = layer["material"]
#                     thickness = layer["thickness"]
#                     transmission = data["transmission"]
#                     # reflection = data["reflection"]
#                     # absorption = data["absorption"]
#
#                     new_csv.append([material, thickness, transmission, np.mean(transmission)])
#
#                 df = pd.DataFrame(new_csv, columns=['Material', 'Thickness', 'Transmission', 'Mean Transmission'])
#                 df.to_csv(output_folder + '/{}.csv'.format(filename), index=False)
#
#                 # Create plot
#                 plt.figure()
#                 plt.plot(data["wavelengths"], transmission, 'r')
#                 # plt.plot(data["wavelengths"], reflection, 'b')
#                 # plt.plot(data["wavelengths"], absorption, 'g')
#
#                 plt.xlabel("Wavelength")
#                 plt.ylabel("Transmission")
#                 plt.title(f"Transmission Spectrum from {filename}")
#                 plt.legend(["Transmission", "Reflection", "Absorption"])
#
#                 output_filename = os.path.splitext(filename)[0] + ".png"
#                 output_path = os.path.join(output_folder, output_filename)
#                 plt.savefig(output_path)
#
#                 plt.close()
#
#         except json.JSONDecodeError:
#             print(f"Error: Unable to decode JSON file {filename}")
#
# if __name__ == "__main__":
#     event_handler = JSONPlotHandler()
#     observer = Observer()
#     observer.schedule(event_handler, input_folder, recursive=False)
#     observer.start()
#
#     try:
#         while True:
#             time.sleep(600)  # Sleep for 10 minutes
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()
#
#


input_folder = "abs_test"

output_folder = "abs_test_0702124_1"
os.makedirs(output_folder, exist_ok=True)


# Iterate through JSON files, sorted by episode number
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".json"):
        filepath = os.path.join(input_folder, filename)
        try:
            with open(filepath, "r") as file:
                data = json.load(file)

                new_csv = []
                for layer in data['layers']:  # Iterate through the layers list
                    material = layer["material"]
                    thickness = layer["thickness"]
                    # transmission = data["transmission"]
                    # reflection = data["reflection"]
                    absorption = data["absorption"]


                    new_csv.append([material, thickness, absorption])  # Create the row for csv

                df = pd.DataFrame(new_csv, columns=['Material', 'Thickness', 'R'])
                df.to_csv(output_folder + '/{}.csv'.format(filename), index=False)

                # Create plot
                plt.figure(figsize=(12,5))
                plt.plot(data["wavelengths"], absorption, 'r')
                # plt.xticks(data["wavelengths"])
                plt.xticks(rotation=90)
                plt.xticks(np.arange(400, 1200, 10))

                # Customize plot appearance
                plt.xlabel("Wavelength")
                # plt.ylabel("Transmission")
                # plt.ylabel("Reflection")
                plt.ylabel("Absorption")
                plt.title(f"R from {filename}")
                # plt.legend(["Reflection"])
                plt.legend(["Absorption"])

                # Save plot with same name as JSON file (changing extension to .png)
                output_filename = os.path.splitext(filename)[0] + ".png"  # Remove .json, add .png
                output_path = os.path.join(output_folder, output_filename)
                plt.savefig(output_path)

                plt.close()  # Close the plot to free up memory

        except json.JSONDecodeError:
            print(f"Error: Unable to decode JSON file {filename}")


# def bandpass_reward_function(transmission, wavelengths, previous_bandpasses=None, min_bandwidth=20, high_transmission_threshold=0.7, novelty_weight=0.1):
#     """
#     Reward function to discover new bandpass structures.
#
#     Args:
#         transmission (np.array): Array of transmission values.
#         wavelengths (np.array): Array of corresponding wavelengths.
#         previous_bandpasses (list of tuples): List of previously discovered bandpass regions as tuples (start, end).
#         min_bandwidth (int): Minimum bandwidth for a bandpass region to be considered valid.
#         high_transmission_threshold (float): Transmission threshold to identify high transmission regions.
#         novelty_weight (float): Weight for the novelty component of the reward.
#
#     Returns:
#         float: Calculated reward.
#     """
#
#     # Identify high transmission regions
#     high_transmission_regions = transmission > high_transmission_threshold
#     bandpass_regions = []
#     start_idx = None
#
#     for idx, high_trans in enumerate(high_transmission_regions):
#         if high_trans and start_idx is None:
#             start_idx = idx
#         elif not high_trans and start_idx is not None:
#             end_idx = idx - 1
#             if end_idx - start_idx + 1 >= min_bandwidth:
#                 bandpass_regions.append((wavelengths[start_idx], wavelengths[end_idx]))
#             start_idx = None
#
#     # Handle case where the bandpass region extends to the end of the spectrum
#     if start_idx is not None and len(high_transmission_regions) - start_idx >= min_bandwidth:
#         bandpass_regions.append((wavelengths[start_idx], wavelengths[-1]))
#
#     # Calculate reward based on bandpass width and transmission
#     reward = 0
#     for start, end in bandpass_regions:
#         bandwidth = end - start
#         mean_transmission = np.mean(transmission[(wavelengths >= start) & (wavelengths <= end)])
#         reward += bandwidth * mean_transmission
#
#     # Encourage discovery of new bandpasses
#     if previous_bandpasses is not None:
#         novelty_reward = 0
#         for start, end in bandpass_regions:
#             for prev_start, prev_end in previous_bandpasses:
#                 overlap = max(0, min(end, prev_end) - max(start, prev_start))
#                 bandwidth = end - start
#                 novelty_reward += max(0, (bandwidth - overlap) / bandwidth)
#         reward += novelty_weight * novelty_reward
#
#     return reward
#
# # Example usage:
# transmission = np.array([0.1, 0.3, 0.8, 0.9, 0.85, 0.7, 0.6, 0.4, 0.3, 0.8, 0.9, 0.85, 0.7])
# wavelengths = np.arange(400, 400 + 10 * len(transmission), 10)
# previous_bandpasses = [(450, 500), (550, 600)]
#
# reward = bandpass_reward_function(transmission, wavelengths, previous_bandpasses)
# print(reward)