import pandas as pd
import numpy as np
import os
import math

# --- CONFIGURATION ---
# DATA_PATH = os.path.expanduser("~/xdaadbot_ws/calibration_data.csv")
DATA_PATH = "/home/maryammahmood/xdaadbot_ws/calibration_data_sum_sines.csv"
DELTA = 0.05  # Confidence level 1-delta (e.g., 0.1 means 90% confidence)

def calculate_cp_quantile():
    if not os.path.exists(DATA_PATH):
        print(f"Error: File not found at {DATA_PATH}")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Compute Nonconformity Scores (s_i)
    # The score for a trajectory is the MAXIMUM error observed during that trajectory
    # Group by 'traj_id' and take max of 'error_norm'
    scores = df.groupby('traj_id')['error_norm'].max().values
    
    N = len(scores)
    print(f"Number of Calibration Trajectories (N): {N}")
    
    if N == 0:
        print("Error: No trajectory data found.")
        return

    # 2. Sort Scores
    scores_sorted = np.sort(scores)
    
    # 3. Calculate Quantile Index
    # Formula: (1 - delta) * (1 + 1/N)
    # We find the index corresponding to this percentile
    
    val_percentile = (1.0 - DELTA) * (1.0 + (1.0 / N))
    
    # Clamp to max 1.0 (can happen if N is small)
    if val_percentile > 1.0:
        val_percentile = 1.0
        
    # Get index (0-based)
    # np.quantile or manual indexing.
    # Manual indexing for CP exactness: ceil((N+1)(1-delta)) - 1
    idx = math.ceil((N + 1) * (1.0 - DELTA)) - 1
    idx = min(idx, N - 1) # Clamp index
    idx = max(idx, 0)
    
    q_cp = scores_sorted[idx]
    
    print("-" * 30)
    print(f"Results for Delta = {DELTA} (Confidence {100*(1-DELTA)}%)")
    print("-" * 30)
    print(f"Raw Scores (Max Error per Traj):")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Avg: {scores.mean():.4f}")
    print("-" * 30)
    print(f"Calculated Quantile (q_cp): {q_cp:.5f}")
    print("-" * 30)
    print(f"Use this value in your CR-CLF code:")
    print(f"self.q_cp = {q_cp:.5f}")

if __name__ == "__main__":
    calculate_cp_quantile()