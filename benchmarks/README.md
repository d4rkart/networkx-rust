# NetworkX vs networkx-rs Benchmarking Suite

This benchmarking suite allows you to compare the performance of Python's NetworkX library with the Rust networkx-rs implementation across various graph operations and sizes.

## Prerequisites

### Python Dependencies
- Python 3.6+
- NetworkX
- NumPy
- Matplotlib
- Pandas

You can install the Python dependencies with:
```bash
pip install networkx numpy matplotlib pandas scipy pandas
```

### Rust Dependencies
- Rust (with Cargo)
- networkx-rs crate and its dependencies (automatically handled by Cargo)

## Files in the Suite

- `benchmark_networkx.py` - Python script for benchmarking NetworkX
- `benchmark_networkx_rs.rs` - Rust code for benchmarking networkx-rs
- `compare_benchmarks.py` - Python script to compare and visualize results
- `run_benchmarks.sh` - Shell script to run all benchmarks and generate reports

## Running the Benchmarks

### Option 1: Using the Shell Script (Recommended)

The easiest way to run all benchmarks is to use the provided shell script:

```bash
# Make the script executable
chmod +x run_benchmarks.sh

# Run with default settings
./run_benchmarks.sh

# Run with quick mode (limited graph sizes for faster testing)
./run_benchmarks.sh --quick

# Or customize with options
./run_benchmarks.sh --sizes "100,500 500,2500" --charts-dir "my_charts" --report "my_report.html" --timeout 3600
```

Available options:
- `--sizes` - Space-separated list of node,edge pairs (e.g., "100,500 1000,5000")
- `--python-output` - Path to save Python benchmark results
- `--rust-output` - Path to save Rust benchmark results  
- `--charts-dir` - Directory to save generated charts
- `--report` - Path to save HTML report
- `--timeout` - Maximum execution time in seconds (default: 7200 seconds/2 hours)
- `--quick` - Run in quick mode with limited graph sizes for faster testing

### Option 2: Running Individual Components

You can also run each component separately:

1. Python benchmarks:
```bash
python3 src/benchmark_networkx.py --sizes 100,500 500,2500 1000,5000 --output python_results.csv
```

2. Rust benchmarks:
```bash
cargo run --release --bin networkx_benchmarks -- --sizes 100,500 500,2500 1000,5000 --output rust_results.csv
```

3. Compare results:
```bash
python3 src/compare_benchmarks.py --python python_results.csv --rust rust_results.csv --output-dir charts --report report.html
```

## Graph Sizes

By default, the benchmark tests a wide range of graph sizes:
- Small: 100 nodes with 500 edges
- Medium: 500 nodes with 2,500 edges, 1,000 nodes with 5,000 edges
- Large: 5,000 nodes with 25,000 edges
- Very Large: 10,000 nodes with 50,000 edges, 20,000 nodes with 100,000 edges

For very large graphs, certain expensive operations (like all-pairs shortest path) are automatically skipped to keep benchmark execution time reasonable.

The benchmark has a timeout mechanism to prevent it from running indefinitely. If the benchmark times out, results up to that point will still be saved and compared.

## Operations Benchmarked

The suite benchmarks the following operations:

1. **Graph Creation**
   - Creating undirected graphs
   - Creating directed graphs

2. **Path Finding**
   - Finding shortest paths
   - Computing shortest path lengths

3. **Graph Traversal**
   - Depth-first search (DFS)
   - Breadth-first search (BFS)

4. **Graph Manipulation**
   - Adding nodes
   - Adding edges

5. **Advanced Algorithms** (Python only for some)
   - PageRank
   - Connected Components

For larger graphs, some operations may be skipped to ensure the benchmark completes in a reasonable time.

## Visualization and Reports

After running the benchmarks, the suite generates:

1. **CSV files** with raw timing data
2. **PNG charts** comparing performance across different operations
3. **HTML report** with tables and visualizations for easy analysis

The HTML report includes:
- Performance summary table
- Speedup metrics (Rust vs Python)
- Bar charts comparing execution times
- Detailed breakdown by operation type

## Extending the Benchmarks

### Adding New Operations

To add new operations to benchmark:

1. Add the operation to both the Python and Rust benchmark scripts
2. Update the comparison script if needed

### Testing Different Graph Types

The current benchmarks focus on random graphs. To test other graph types:

1. Modify the graph creation functions in both benchmark scripts
2. Ensure both implementations create equivalent graph structures

## Troubleshooting

- **Missing Python dependencies**: Run `pip install networkx numpy matplotlib pandas`
- **Rust compilation errors**: Ensure your Rust toolchain is up-to-date with `rustup update`
- **Permission denied when running shell script**: Run `chmod +x run_benchmarks.sh`
- **Benchmark takes too long**: Use the `--quick` option or reduce graph sizes with `--sizes`
- **Out of memory**: Reduce the size of graphs being tested or increase system memory

## License

This benchmarking suite is provided under the same license as the networkx-rs project. 