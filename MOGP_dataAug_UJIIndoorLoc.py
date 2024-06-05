# -*- coding:utf-8 -*-
# @Time : 2021/11/10 12:10 下午
# @Author: tangzhetim
# @File : generate_data_multikernerl.py
# @brief : This file uses hybrid kernels to generate data on a building-wide scale.
# Changing the order in which files are written is not recommended because it is associated
# with other modules.

import os
import GPy
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle as pk
import sklearn.preprocessing
import time

# Start timing the script
time_start = time.time()

def increase_pred_input_nd(num, df_lim: pd.DataFrame, data_set: int):
    """Generate random prediction input locations within the limits of a given dataset."""
    df_lim_dataset = df_lim.loc[df_lim["DATASET"] == data_set]
    x_min = df_lim_dataset["LONGITUDE"]["min"].values[0]
    x_max = df_lim_dataset["LONGITUDE"]["max"].values[0]
    y_min = df_lim_dataset["LATITUDE"]["min"].values[0]
    y_max = df_lim_dataset["LATITUDE"]["max"].values[0]
    return np.array([[random.uniform(x_min, x_max), random.uniform(y_min, y_max), data_set] for _ in range(num)])

def plot_func(xy, z, xy_pred, z_pred, n_dim):
    """Plot real and predicted data in 3D space for a specific dimension."""
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    x_axis = xy[:, 0]
    y_axis = xy[:, 1]
    ax.scatter(x_axis, y_axis, z[:, n_dim], c=z[:, n_dim], cmap='Greys', label=f"WAP{n_dim+1}")
    if xy_pred is not None:
        ax.scatter(xy_pred[:, 0], xy_pred[:, 1], z_pred[:, n_dim], c=z_pred[:, n_dim], cmap='hsv', label="pred")
    plt.legend(loc='upper left')
    plt.show()

############################################################################
# Input parameters
buildingid = 0 #related to the input of the data
ratio = 1

source_data_path = ".csv"
fake_data_path = ".csv"

# Read the data of building 0
with open(source_data_path, "r") as _file:
    df_raw = pd.read_csv(_file).loc[:, "WAP001":"BUILDINGID"]

fake_data_for_whole_building = int(len(df_raw) / ratio)

# Adjust FLOOR values and prepare data
df_raw.loc[:, "FLOOR"] *= 4
df = df_raw
xy = df.loc[:, "LONGITUDE":"FLOOR"].to_numpy()
z_original = df.loc[:, "WAP001": "WAP520"].to_numpy()

# Standardize the RSS data
standarder = sklearn.preprocessing.StandardScaler()
standarder.fit(z_original)
z = standarder.transform(z_original)

# Define the kernel and model
kernel_Matern52 = GPy.util.multioutput.LCM(input_dim=3, num_outputs=520, kernels_list=[GPy.kern.Matern52(3)])
m = GPy.models.GPCoregionalizedRegression([xy], [z], kernel=kernel_Matern52)
m.optimize_restarts(num_restarts=10)

# Generate random prediction inputs
data_set = 0
xy_pred = increase_pred_input_nd(fake_data_for_whole_building, df, data_set)

# Predict at the new locations
Y_metadata = {"output_index": xy_pred[:, -1].astype(int)}
z_pred_raw = m.predict(xy_pred, Y_metadata=Y_metadata)
z_pred = z_pred_raw[0]
z_pred_ori = standarder.inverse_transform(z_pred).round()
z_pred_ori_a = pd.DataFrame(z_pred_ori)

# Create a new DataFrame for the fake data
df_new = pd.DataFrame(z_pred_ori, columns=df_raw.columns[:520]).astype("int64")
df_new["LONGITUDE"] = xy_pred[:, 0]
df_new["LATITUDE"] = xy_pred[:, 1]

# Assign random floor values based on building ID
if buildingid == 2:
    df_floor = np.random.choice(a=[0, 1, 2, 3, 4], size=fake_data_for_whole_building, replace=True, p=[0.204, 0.228, 0.167, 0.285, 0.116])
else:
    df_floor = np.random.choice(a=[0, 1, 2, 3], size=fake_data_for_whole_building, replace=True, p=[0.235, 0.270, 0.270, 0.225])

df_new["FLOOR"] = df_floor
df_new["BUILDINGID"] = np.zeros(xy_pred[:, 0].shape, dtype=int) + buildingid
df_new["SPACEID"] = np.zeros(xy_pred[:, 0].shape, dtype=int)
df_new["RELATIVEPOSITION"] = np.zeros(xy_pred[:, 0].shape, dtype=int)
df_new["USERID"] = np.zeros((fake_data_for_whole_building,1),dtype=int)
df_new["PHONEID"] = np.zeros((fake_data_for_whole_building,1),dtype=int)
df_new["TIMESTAMP"] = np.zeros((fake_data_for_whole_building,1),dtype=int)

with open(fake_data_path, "w") as f:
    f.write(df_new.to_csv(index=False))
time_end = time.time()
print('time cost', time_end - time_start, 's')