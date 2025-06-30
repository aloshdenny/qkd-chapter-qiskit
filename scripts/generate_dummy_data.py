import pandas as pd
import numpy as np
import argparse

def generate_dummy_data(num_rows: int, output_path: str):
    """
    Generates a detailed, correlated, and concentrated dummy dataset for QKD simulations.

    Args:
        num_rows (int): The number of rows to generate in the dataset.
        output_path (str): The path to save the generated CSV file.
    """
    protocols = ["ARC-QKD", "BB84", "ECDH"]
    device_types = ["Low-end", "Mid-range", "High-end"]
    
    data = {
        "protocol": np.random.choice(protocols, num_rows, p=[0.5, 0.3, 0.2]),
        "device_name": np.random.choice(device_types, num_rows),
    }
    df = pd.DataFrame(data)

    # --- Generate Concentrated Base Metrics using Normal Distributions ---
    
    # Define mean and std dev for device specs to create clusters
    device_specs = {
        "Low-end": {"cpu": (90, 5), "ram": (96, 16), "power": (15, 2)},
        "Mid-range": {"cpu": (120, 10), "ram": (192, 32), "power": (28, 4)},
        "High-end": {"cpu": (160, 8), "ram": (384, 64), "power": (42, 5)},
    }
    df["cpu_mhz"] = df["device_name"].apply(lambda x: np.random.normal(loc=device_specs[x]["cpu"][0], scale=device_specs[x]["cpu"][1]))
    df["ram_kb"] = df["device_name"].apply(lambda x: np.random.normal(loc=device_specs[x]["ram"][0], scale=device_specs[x]["ram"][1]))
    df["power_mw"] = df["device_name"].apply(lambda x: np.random.normal(loc=device_specs[x]["power"][0], scale=device_specs[x]["power"][1]))

    # Channel and block size remain uniformly distributed to cover various scenarios
    df["distance_km"] = np.random.uniform(0.1, 10, num_rows)
    df["channel_loss_db_km"] = np.random.uniform(0.2, 2, num_rows)
    df["block_size"] = np.random.choice([1000, 2000, 3000, 4000, 5000], num_rows)
    df["error_correction"] = np.random.choice(["BCH", "LDPC"], num_rows)

    # --- Model Correlated Performance with Realistic Variance ---

    # QBER clusters around a mean determined by distance and loss
    base_qber = (df["distance_km"] * 0.005) + (df["channel_loss_db_km"] * 0.01)
    df["qber"] = np.random.normal(loc=base_qber, scale=0.005) # scale is std dev
    df["qber"] = df["qber"].clip(0.01, 0.15)

    # Key rate clusters around a mean dependent on QBER, distance, and CPU
    base_key_rate = 1200 * (df["cpu_mhz"] / 120) / (1 + df["qber"] * 30) / (1 + df["distance_km"] * 0.4)
    df["key_rate_bps"] = np.random.normal(loc=base_key_rate, scale=40) # scale is std dev
    df["key_rate_bps"] = df["key_rate_bps"].clip(20)

    # Latency clusters around a mean dependent on block size and RAM
    base_latency = 1.0 + (df["block_size"] / 1500) - (df["ram_kb"] / 512 * 0.8)
    df["latency_s"] = np.random.normal(loc=base_latency, scale=0.15) # scale is std dev
    df["latency_s"] = df["latency_s"].clip(0.4)
    
    # --- Protocol-specific Adjustments ---
    # ARC-QKD is the high-performance baseline
    # BB84 is less efficient
    df.loc[df["protocol"] == "BB84", "key_rate_bps"] *= np.random.normal(loc=0.7, scale=0.05)
    df.loc[df["protocol"] == "BB84", "latency_s"] *= np.random.normal(loc=1.4, scale=0.1)
    
    # ECDH is for initial key agreement only
    ecdh_mask = df["protocol"] == "ECDH"
    df.loc[ecdh_mask, "key_rate_bps"] = 0
    df.loc[ecdh_mask, "qber"] = 0
    df.loc[ecdh_mask, "block_size"] = 0
    # ECDH latency is fast but depends on CPU
    ecdh_latency_base = 0.4 - (df.loc[ecdh_mask, "cpu_mhz"] / 800)
    df.loc[ecdh_mask, "latency_s"] = np.random.normal(loc=ecdh_latency_base, scale=0.05, size=ecdh_mask.sum())

    # --- Protocol Phase Timing ---
    qkd_mask = df["protocol"] != "ECDH"
    total_qkd_latency = df.loc[qkd_mask, "latency_s"]
    # Define phase contributions
    phases_dist = {
        "phase1_init_s": 0.10,
        "phase2_qkd_s": 0.55,
        "phase3_post_processing_s": 0.25,
        "phase4_privacy_amp_s": 0.05,
        "phase5_key_management_s": 0.05,
    }
    for phase, proportion in phases_dist.items():
        # Assign each phase its proportion of the total latency, with some noise
        df.loc[qkd_mask, phase] = total_qkd_latency * np.random.normal(loc=proportion, scale=0.02, size=qkd_mask.sum())
    
    df.fillna(0, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Concentrated and correlated dummy data generated and saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy data for QKD simulations.")
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/comparative_dummy_results.csv",
        help="Path to save the generated CSV file.",
    )
    args = parser.parse_args()

    generate_dummy_data(args.rows, args.output)
