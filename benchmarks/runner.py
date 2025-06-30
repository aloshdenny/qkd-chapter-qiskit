import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from suite import QKDBenchmarkSuite

def main():
    parser = argparse.ArgumentParser(description="Run QKD Benchmark Suite")
    parser.add_argument(
        "--output",
        type=str,
        default="qkd_benchmark_results.csv",
        help="Output file for benchmark results",
    )
    args = parser.parse_args()

    suite = QKDBenchmarkSuite()
    results = suite.run_comprehensive_study(args.output)

    print(f"\nBenchmarking Complete.")
    print(f"{len(results)} entries saved to '{args.output}'.")

if __name__ == "__main__":
    main()
