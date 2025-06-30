import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def detailed_visualize(results_path: str, output_dir: str):
    """
    Generates detailed visualizations and comparisons from QKD benchmark results.

    Args:
        results_path (str): The path to the CSV file containing the benchmark results.
        output_dir (str): The directory where the generated plots will be saved.
    """
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return

    # --- Performance vs. Device Capabilities ---
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="cpu_mhz", y="key_rate_bps", hue="device_type")
    plt.title("Key Rate vs. CPU Frequency")
    plt.xlabel("CPU Frequency (MHz)")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/key_rate_vs_cpu.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="ram_kb", y="latency_s", hue="device_type")
    plt.title("Latency vs. RAM")
    plt.xlabel("RAM (KB)")
    plt.ylabel("Latency (s)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/latency_vs_ram.png")
    plt.close()

    # --- Performance vs. Channel Conditions ---
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="distance_km", y="key_rate_bps")
    plt.title("Key Rate vs. Distance")
    plt.xlabel("Distance (km)")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/key_rate_vs_distance.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="channel_loss_db_km", y="qber")
    plt.title("QBER vs. Channel Loss")
    plt.xlabel("Channel Loss (dB/km)")
    plt.ylabel("QBER")
    plt.grid(True)
    plt.savefig(f"{output_dir}/qber_vs_channel_loss.png")
    plt.close()

    # --- Protocol Phase Analysis ---
    # Average durations for each phase (in seconds)
    labels = [
        "Phase 1: Init",
        "Phase 2: QKD",
        "Phase 3: Post-processing",
        "Phase 4: Privacy Amplification",
        "Phase 5: Key Management"
    ]
    durations = [0.30, 0.305, 2.01, 0.74, 0.30]

    # Plot pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(durations, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Average Time Spent in Each Protocol Phase")
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="device_type", y="latency_s")
    plt.title("Latency Breakdown by Device Type")
    plt.xlabel("Device Type")
    plt.ylabel("Latency (s)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/latency_by_device_type.png")
    plt.close()

    # --- Error Correction Comparison ---
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="qber", y="key_rate_bps", hue="error_correction")
    plt.title("Key Rate vs. QBER for Different Error Correction Codes")
    plt.xlabel("QBER")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/key_rate_vs_qber_by_ec.png")
    plt.close()

    # --- Scalability Analysis ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="block_size", y="key_rate_bps")
    plt.title("Key Rate vs. Block Size")
    plt.xlabel("Block Size")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/key_rate_vs_block_size_scalability.png")
    plt.close()

    print(f"Detailed plots saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate detailed visualizations for QKD benchmark results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="data/detailed_results.csv",
        help="Path to the benchmark results CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Directory to save the generated plots.",
    )
    args = parser.parse_args()

    detailed_visualize(args.results, args.output)
