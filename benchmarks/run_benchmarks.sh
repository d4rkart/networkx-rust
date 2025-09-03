#!/bin/bash

# Script to run NetworkX and networkx-rs benchmarks and compare results

set -e  # Exit on error

# Default graph sizes - includes both current sizes and much larger ones
GRAPH_SIZES="100,500 500,2500 1000,5000 5000,25000 10000,50000 20000,100000"
QUICK_SIZES="100,500 1000,5000" # A limited set for quick testing
PYTHON_OUTPUT="results/python_benchmark_results.csv"
RUST_OUTPUT="results/rust_benchmark_results.csv"
CHARTS_DIR="charts"
REPORT="results/benchmark_report.html"
TIMEOUT=7200  # 2 hour timeout for the entire benchmark process
QUICK_MODE=false  # Whether to run in quick mode with limited graph sizes
NUM_RUNS=30  # Number of times to run each test for more accurate results

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --sizes)
      GRAPH_SIZES="$2"
      shift 2
      ;;
    --python-output)
      PYTHON_OUTPUT="$2"
      shift 2
      ;;
    --rust-output)
      RUST_OUTPUT="$2"
      shift 2
      ;;
    --charts-dir)
      CHARTS_DIR="$2"
      shift 2
      ;;
    --report)
      REPORT="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --quick)
      QUICK_MODE=true
      GRAPH_SIZES="$QUICK_SIZES"
      TIMEOUT=900  # 15 minutes timeout for quick mode
      NUM_RUNS=10  # Fewer runs in quick mode
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directories
mkdir -p "$(dirname "$PYTHON_OUTPUT")"
mkdir -p "$(dirname "$RUST_OUTPUT")"
mkdir -p "$CHARTS_DIR"
mkdir -p "$(dirname "$REPORT")"

if [ "$QUICK_MODE" = true ]; then
  echo "=== Running in QUICK MODE with limited graph sizes ==="
fi

echo "=== Running benchmarks with graph sizes: $GRAPH_SIZES ==="
echo "=== Each test will run $NUM_RUNS times for more accurate results ==="
echo "=== Timeout set to: $TIMEOUT seconds ==="
echo

# Make scripts executable
chmod +x src/benchmark_networkx.py
chmod +x src/compare_benchmarks.py

# Run Python NetworkX benchmarks with timeout
echo "=== Running Python NetworkX benchmarks ==="
timeout $TIMEOUT python3 src/benchmark_networkx.py --sizes "$GRAPH_SIZES" --output "$PYTHON_OUTPUT" --runs $NUM_RUNS || {
  echo "Python benchmarks took too long and were terminated."
  echo "You may want to reduce the graph sizes or increase the timeout."
}
echo

# Compile and run Rust networkx-rs benchmarks
echo "=== Compiling Rust networkx-rs benchmark ==="
cd ..
cargo build --release --bin networkx_benchmarks
echo

echo "=== Running Rust networkx-rs benchmarks ==="
timeout $((TIMEOUT / 2)) cargo run --release --bin networkx_benchmarks -- --sizes "$GRAPH_SIZES" --output "benchmarks/$RUST_OUTPUT" --runs $NUM_RUNS || {
  echo "Rust benchmarks took too long and were terminated."
  echo "You may want to reduce the graph sizes or increase the timeout."
}
cd benchmarks
echo

# Compare results and generate report
echo "=== Generating comparison report ==="
python3 src/compare_benchmarks.py \
  --python "$PYTHON_OUTPUT" \
  --rust "$RUST_OUTPUT" \
  --python-detailed "${PYTHON_OUTPUT%.*}_detailed.csv" \
  --rust-detailed "${RUST_OUTPUT%.*}_detailed.csv" \
  --output-dir "$CHARTS_DIR" \
  --report "$REPORT"

echo
echo "=== Benchmark complete! ==="
echo "Python results: $PYTHON_OUTPUT"
echo "Rust results: $RUST_OUTPUT"
echo "HTML report: $REPORT"
echo "Charts directory: $CHARTS_DIR" 