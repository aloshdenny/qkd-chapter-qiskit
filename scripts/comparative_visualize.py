import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

def comparative_visualize(results_path: str, output_dir: str):
    """
    Generates comparative visualizations of QKD protocols.

    Args:
        results_path (str): The path to the CSV file containing the benchmark results.
        output_dir (str): The directory where the generated plots will be saved.
    """
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return

    # --- Performance Comparison ---
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="distance_km", y="key_rate_bps", hue="protocol")
    plt.title("Key Rate vs. Distance")
    plt.xlabel("Distance (km)")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/comp_key_rate_vs_distance.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="device_type", y="latency_s", hue="protocol")
    plt.title("Latency vs. Device Type")
    plt.xlabel("Device Type")
    plt.ylabel("Latency (s)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/comp_latency_vs_device_type.png")
    plt.close()

    # --- Resource Usage Comparison ---
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="protocol", y="cpu_mhz")
    plt.title("CPU Usage vs. Protocol")
    plt.xlabel("Protocol")
    plt.ylabel("CPU Frequency (MHz)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/comp_cpu_usage_vs_protocol.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="protocol", y="ram_kb")
    plt.title("RAM Usage vs. Protocol")
    plt.xlabel("Protocol")
    plt.ylabel("RAM (KB)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/comp_ram_usage_vs_protocol.png")
    plt.close()

    # --- Security vs. Performance ---
    qkd_df = df[df["protocol"] != "ECDH"]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=qkd_df, x="qber", y="key_rate_bps", hue="protocol")
    plt.title("QBER vs. Key Rate for QKD Protocols")
    plt.xlabel("QBER")
    plt.ylabel("Key Rate (bps)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/comp_qber_vs_key_rate.png")
    plt.close()

    # --- Overall Comparison ---
    overall_comparison = df.groupby("protocol")[["key_rate_bps", "latency_s", "power_mw"]].mean().reset_index()
    overall_comparison.plot(x="protocol", y=["key_rate_bps", "latency_s", "power_mw"], kind="bar", figsize=(12, 8), subplots=True, layout=(1, 3), sharey=False)
    plt.suptitle("Overall Comparison of Protocols")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{output_dir}/comp_overall_comparison.png")
    plt.close()


    print(f"Comparative plots saved to '{output_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate comparative visualizations for QKD benchmark results."
    )
    parser.add_argument(
        "--results",
        type=str,
        default="data/comparative_results.csv",
        help="Path to the benchmark results CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Directory to save the generated plots.",
    )
    args = parser.parse_args()

    comparative_visualize(args.results, args.output)
