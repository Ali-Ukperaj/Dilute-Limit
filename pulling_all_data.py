#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate


# In[6]:


N_frames = 1000

## Simulation/Analysis Settings
start_frame = 10
# Number of frames per block
block_size = 50
block_indices = [[block_size*(i-1), block_size*i] for i in range(1, (N_frames//block_size)+1)]
# print(block_indices)


# In[ ]:


## Simulation settings and paths
Temps = [0.7, 0.8, 0.9, 1.0]
eps12s = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
L1s = 10, 20, 50
N1s = 720, 360, 144
L2s = 10, 20, 50
N2s = {
    10: [1, 2, 5, 10, 20, 40, 80, 160, 320, 640, 1000],
    20: [1, 5, 10, 20, 40, 80, 160, 320, 500],
    50: [1, 2, 4, 8, 16, 32, 64, 128, 200]
}

# e.g., /vol_dignon01/aiu8/limitLJ/SPCP/analysis/partition_coeff_analysis/0.7T/0.5e12/L2_10/L1_10/N2_1000/density/density_phases_dump_{mol_sig}.csv
home = "/vol_dignon01/aiu8"
proj = f"{home}/limitLJ/SPCP"
densbase = f"{proj}/analysis/partition_coeff_analysis"
filesave = f"master_spreadsheet_20261012.csv"


# In[ ]:


## Functions
bead_volume = np.pi/6.0
def rho_to_phi(rho):
    return rho * bead_volume

def calc_block_SEM(data, block_size):
    data = np.asarray(data)
    N_samples = len(data)
    N_blocks = N_samples // block_size

    if N_blocks < 2:
        return np.nan

    block_means = []
    for i in range(1, N_blocks+1):
        start = block_size * (i-1)
        end = block_size * i
        block_data = data[start:end]
        block_mean = np.mean(block_data)
        block_means.append(block_mean)
    SEM = np.std(block_means, ddof=1) / np.sqrt(N_blocks)
    return SEM


def calc_K2_SEM(rho2_dense_data, rho2_dilute_data, block_size):
    dense = np.asarray(rho2_dense_data, dtype=float)
    dilute = np.asarray(rho2_dilute_data, dtype=float)
    if len(dense) != len(dilute):
        raise ValueError("Density arrays must have equal lengths")
    N_blocks = len(dense) // block_size
    if N_blocks < 2:
        return np.nan

    # Paired block means
    dense_blocks = []
    dilute_blocks = []
    for i in range(N_blocks):
        start = i * block_size
        end = start + block_size
        dense_blocks.append(np.mean(dense[start:end]))
        dilute_blocks.append(np.mean(dilute[start:end]))
    dense_blocks = np.asarray(dense_blocks)
    dilute_blocks = np.asarray(dilute_blocks)

    A = np.mean(dense_blocks)
    B = np.mean(dilute_blocks)
    if B == 0:
        return np.nan

    # Variance and covariance of the global mean estimates
    var_A = np.var(dense_blocks, ddof=1) / N_blocks
    var_B = np.var(dilute_blocks, ddof=1) / N_blocks
    cov_AB = np.cov(dense_blocks, dilute_blocks, ddof=1)[0, 1] / N_blocks

    # Delta-method uncertainty propagation
    var_K2 = var_A / B**2 + (A**2 * var_B) / B**4 - 2 * A * cov_AB / B**3
    return np.sqrt(max(var_K2, 0.0))


# In[ ]:


## Loop through files
# data = [temp, eps12, L1, N1, L2, N2, M2, K2, SEM_K2, log(K2),
#         phi_all_dense,     phi_all_dilute,     phi1_dense,     phi1_dilute,     phi2_dense,     phi2_dilute,
#NOTDONE: SEM_phi_all_dense, SEM_phi_all_dilute, SEM_phi1_dense, SEM_phi1_dilute, SEM_phi2_dense, SEM_phi2_dilute
#         rho_all_dense,     rho_all_dilute,     rho1_dense,     rho1_dilute,     rho2_dense,     rho2_dilute,
#NOTDONE: Block_size, start_frame, SEM_rho_all_dense, SEM_rho_all_dilute, SEM_rho1_dense, SEM_rho1_dilute, SEM_rho2_dense, SEM_rho2_dilute]
header = ["temp","eps12","L1","N1","L2","N2","M2",
          "block_size", "start_frame",
          "K2","log_K2", "SEM_K2", "SEM_log_K2",
          "phi_all_dense","phi_all_dilute","phi1_dense","phi1_dilute","phi2_dense","phi2_dilute",
          "SEM_phi_all_dense", "SEM_phi_all_dilute", "SEM_phi1_dense", "SEM_phi1_dilute", "SEM_phi2_dense", "SEM_phi2_dilute",
          "rho_all_dense","rho_all_dilute","rho1_dense","rho1_dilute","rho2_dense","rho2_dilute",
          "SEM_rho_all_dense", "SEM_rho_all_dilute", "SEM_rho1_dense", "SEM_rho1_dilute", "SEM_rho2_dense", "SEM_rho2_dilute"]
data, missing_files, failed_files = [], [], []
for temp in Temps:
    for eps12 in eps12s:
        for L1, N1 in zip(L1s, N1s):
            for L2 in L2s:
                for N2 in N2s[L2]:
                    M2 = L2*N2
                    mol_sig = f"Mol1_L{L1}N{N1}_Mol2_L{L2}N{N2}"
                    dens = f"{densbase}/{temp}T/{eps12}e12/L2_{L2}/L1_{L1}/N2_{N2}/density"
                    density_fit = f"{dens}/density_fit_dump_{mol_sig}.csv"
                    density_frames = f"{dens}/density_frames_dump_{mol_sig}.npz"
                    density_phases = f"{dens}/density_phases_dump_{mol_sig}.csv"
                    density_profile = f"{dens}/density_profile_dump_{mol_sig}.csv"

                     # Incase simulation aspect missing/broken
                    K2, log_K2, SEM_K2, SEM_log_K2 = np.nan, np.nan, np.nan, np.nan
                    rho_all_dense, rho_all_dilute, rho1_dense, rho1_dilute, rho2_dense, rho2_dilute = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    phi_all_dense, phi_all_dilute, phi1_dense, phi1_dilute, phi2_dense, phi2_dilute = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    SEM_rho_all_dense, SEM_rho_all_dilute, SEM_rho1_dense, SEM_rho1_dilute, SEM_rho2_dense, SEM_rho2_dilute = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
                    SEM_phi_all_dense, SEM_phi_all_dilute, SEM_phi1_dense, SEM_phi1_dilute, SEM_phi2_dense, SEM_phi2_dilute = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

                    if not os.path.exists(density_phases):
                        missing_files.append(density_phases)
                    else:
                        try:
                            phases = np.genfromtxt(density_phases, delimiter=",", names=True)

                            density_columns = ["rho_dense_all", "rho_dilute_all", "rho_dense_Mol1", "rho_dilute_Mol1", "rho_dense_Mol2", "rho_dilute_Mol2"]
                            density_means, density_SEMs = [], []
                            phi_means, phi_SEMs = [], []
                            for column in density_columns:
                                density_data = np.asarray(phases[column][start_frame:], dtype=float)
                                if len(density_data) < 2 * block_size:
                                    raise ValueError(f"Insufficient frames for {column}: {len(density_data)}")
                                if not np.all(np.isfinite(density_data)):
                                    raise ValueError(f"Nonfinite values in {column}")
                                density_means.append(np.mean(density_data))
                                density_SEMs.append(calc_block_SEM(density_data, block_size))

                                phi_data = rho_to_phi(density_data)
                                phi_means.append(np.mean(phi_data))
                                phi_SEMs.append(calc_block_SEM(phi_data, block_size))

                            rho_all_dense, rho_all_dilute, rho1_dense, rho1_dilute, rho2_dense, rho2_dilute = density_means
                            SEM_rho_all_dense, SEM_rho_all_dilute, SEM_rho1_dense, SEM_rho1_dilute, SEM_rho2_dense, SEM_rho2_dilute = density_SEMs

                            phi_all_dense, phi_all_dilute, phi1_dense, phi1_dilute, phi2_dense, phi2_dilute = phi_means
                            SEM_phi_all_dense, SEM_phi_all_dilute, SEM_phi1_dense, SEM_phi1_dilute, SEM_phi2_dense, SEM_phi2_dilute = phi_SEMs

                            if (np.isfinite(rho2_dense) and np.isfinite(rho2_dilute) and rho2_dilute > 0):
                                K2 = (rho2_dense / rho2_dilute)
                                if K2 > 0:
                                    log_K2 = np.log(K2)

                            SEM_K2 = calc_K2_SEM(phases["rho_dense_Mol2"][start_frame:], phases["rho_dilute_Mol2"][start_frame:], block_size)
                            SEM_log_K2 = SEM_K2 / K2 if K2 > 0 else np.nan

                            phi_all_dense = rho_to_phi(rho_all_dense)
                            phi_all_dilute = rho_to_phi(rho_all_dilute)
                            phi1_dense = rho_to_phi(rho1_dense)
                            phi1_dilute = rho_to_phi(rho1_dilute)
                            phi2_dense = rho_to_phi(rho2_dense)
                            phi2_dilute = rho_to_phi(rho2_dilute)
                        except Exception as exc:
                            failed_files.append((density_phases, str(exc)))

                    row = [temp, eps12, L1, N1, L2, N2, M2,
                           block_size, start_frame,
                           K2, log_K2, SEM_K2, SEM_log_K2,
                           phi_all_dense, phi_all_dilute, phi1_dense, phi1_dilute, phi2_dense, phi2_dilute,
                           SEM_phi_all_dense, SEM_phi_all_dilute, SEM_phi1_dense, SEM_phi1_dilute, SEM_phi2_dense, SEM_phi2_dilute,
                           rho_all_dense, rho_all_dilute, rho1_dense, rho1_dilute, rho2_dense, rho2_dilute,
                           SEM_rho_all_dense, SEM_rho_all_dilute, SEM_rho1_dense, SEM_rho1_dilute, SEM_rho2_dense, SEM_rho2_dilute]
                    data.append(row)

with open(filesave, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

# Summarize
print()
print("=" * 70)
print("SPREADSHEET COMPLETED")
print("=" * 70)

print(f"Output:             {filesave}")
print(f"Expected rows:      2436")
print(f"Rows written:       {len(data)}")
print(f"Missing files:      {len(missing_files)}")
print(f"Failed files:       {len(failed_files)}")

if missing_files:
    print()
    print("MISSING FILES:")
    for filename in missing_files:
        print(filename)

if failed_files:
    print()
    print("FAILED FILES:")
    for filename, error in failed_files:
        print(filename)
        print(f"    {error}")                    


# In[ ]:




