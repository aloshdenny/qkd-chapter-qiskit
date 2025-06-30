# QKD Simulation for Resource-Constrained IoT Devices

This project simulates a Quantum Key Distribution (QKD) protocol, specifically designed to analyze its feasibility on resource-constrained IoT devices. It includes modules for simulating eavesdropping attacks, various error correction methods, and a benchmarking suite to compare QKD with classical cryptography (ECDH).

## Project Structure

- `src/`: Contains the core Python modules for the simulation.
  - `qkd/`: Modules related to the QKD protocol.
    - `channel.py`: Simulates the quantum channel, including eavesdropping attacks.
    - `device.py`: Simulates a resource-constrained device.
    - `error_correction.py`: Implements different error correction algorithms.
    - `party.py`: Represents a party (e.g., Alice or Bob) in the QKD protocol.
  - `classical/`: Modules related to classical cryptography.
    - `ecdh.py`: Implements the ECDH protocol for baseline comparison.
- `benchmarks/`: Contains the benchmarking suite.
  - `suite.py`: The main benchmarking suite.
  - `runner.py`: A script to run the benchmarks.
- `config/`: Contains configuration files for the simulation.
- `notebooks/`: Contains Jupyter notebooks for analysis and visualization.
- `scripts/`: Contains the original Python scripts.
- `data/`: Stores the results of the simulation.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the simulation:**
   ```bash
   python benchmarks/runner.py --output data/benchmark_results.csv
   ```

   This will run the comprehensive benchmark suite and save the results to `data/benchmark_results.csv`.

3. **Analyze the results:**
   The `runner.py` script will also generate plots summarizing the results. You can also use the Jupyter notebooks in the `notebooks/` directory to perform your own analysis.