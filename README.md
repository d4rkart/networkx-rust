# Network-rs Library

A high-performance Rust implementation of the popular Python [NetworkX](https://github.com/networkx/networkx) library for graph processing and analysis. This library provides a robust, memory-efficient, and thread-safe implementation of graph data structures and algorithms.

## Key Features

- **Multiple Graph Types**: Support for directed graphs (`DiGraph`), undirected graphs (`Graph`), and multigraphs (`MultiGraph`, `MultiDiGraph`)
- **Generic Data Types**: Store any type of data in nodes and edges using Rust's type system
- **Memory Efficient**: Optimized data structures for storing and accessing graph components
- **Thread Safety**: Built-in support for parallel graph processing using Rayon
- **Rich Algorithm Set**: Comprehensive implementation of graph algorithms including:
  - Graph traversal (DFS, BFS)
  - Shortest path finding (Dijkstra's algorithm)
  - Connected components
  - Subgraph extraction
  - Degree calculations
  - And more...
- **API Compatibility**: Familiar API design matching NetworkX's Python interface
- **Zero Dependencies**: Core functionality requires no external dependencies
- **Extensible**: Easy to extend with custom graph algorithms and data structures

## Performance

The library is designed for performance with:
- Efficient adjacency list representation
- Optimized memory usage
- Parallel processing capabilities
- Zero-copy operations where possible
- Minimal allocations

## Getting Started

Add to your `Cargo.toml`:
```toml
[dependencies]
networkx-rs = "0.1.1"
```

Basic usage:
```rust
use networkx_rs::Graph;

// Create a directed graph
let mut graph = Graph::<String, f64>::new(true);

// Add nodes
let n1 = graph.add_node("Node 1".to_string());
let n2 = graph.add_node("Node 2".to_string());

// Add an edge
graph.add_edge(n1, n2, 1.5);

// Find shortest path
let (path, cost) = graph.shortest_path(n1, n2, |&w| w).unwrap();
```

# Testing

## Running Tests

This library includes comprehensive unit tests and integration tests to ensure correctness and reliability.

### Unit Tests

Run all unit tests:
```bash
cargo test
```

Run tests for a specific module:
```bash
cargo test graph_tests
cargo test layout_tests
cargo test multigraph_tests
```

Run a specific test:
```bash
cargo test test_spring_layout
cargo test test_spring_layout_with_json_weights
```

### Example Files

The repository contains example files demonstrating various features of the NetworkX Rust library:

#### 1. Maze Finder (`maze_finder.rs`)
Demonstrates how to create a grid-based maze and find the shortest path through it.

#### 2. NetworkX Examples (`networkx_examples.rs`)
Shows basic graph operations, graph traversals, and shortest path finding.

#### 3. Multi-Graph Example (`multi_graph_example.rs`)
Demonstrates using MultiGraph and MultiDiGraph for representing transportation networks and network flows.

#### 4. Path Finder (`pathfinder.rs`)
A more complex example showing how to model a subway network and find optimal routes based on different metrics (distance vs. travel time).

#### 5. Parallel Graph Processing (`parallel_graph.rs`)
Benchmarks sequential vs. parallel implementations of various graph algorithms.

These example files demonstrate the library's capabilities and can serve as starting points for your own graph processing applications.

## Features Demonstrated

The examples in this repository demonstrate the following features of the NetworkX Rust library:

### Basic Graph Operations
- Creating directed and undirected graphs
- Adding and removing nodes and edges
- Querying graph information (node count, edge count, degree, etc.)
- Traversing the graph (iterating over nodes, edges, neighbors)

### Graph Traversal Algorithms
- Depth-First Search (DFS)
- Breadth-First Search (BFS)
- Shortest path finding using Dijkstra's algorithm

### Specialized Graphs
- DiGraph (Directed Graph)
- MultiGraph (Undirected graph with multiple edges between nodes)
- MultiDiGraph (Directed graph with multiple edges between nodes)

### Parallel Processing
- Parallel node search
- Parallel BFS traversal
- Parallel connected components calculation
- Parallel subgraph extraction

## Examples

### 1. Maze Finder (`maze_finder.rs`)
Demonstrates how to create a grid-based maze and find the shortest path through it.

Run with:
```bash
cargo run --bin maze_finder
```

### 2. NetworkX Examples (`networkx_examples.rs`)
Shows basic graph operations, graph traversals, and shortest path finding.

Run with:
```bash
cargo run --bin networkx_examples
```

### 3. Multi-Graph Example (`multi_graph_example.rs`)
Demonstrates using MultiGraph and MultiDiGraph for representing transportation networks and network flows.

Run with:
```bash
cargo run --bin multi_graph_example
```

### 4. Path Finder (`pathfinder.rs`)
A more complex example showing how to model a subway network and find optimal routes based on different metrics (distance vs. travel time).

Run with:
```bash
cargo run --bin pathfinder
```

### 5. Parallel Graph Processing (`parallel_graph.rs`)
Benchmarks sequential vs. parallel implementations of various graph algorithms.

Run with:
```bash
cargo run --bin parallel_graph
```

## API Insights

### Graph Types
- `Graph<T, E>`: Basic graph structure supporting both directed and undirected graphs
- `DiGraph<T, E>`: Specialized directed graph
- `MultiGraph<T, E>`: Undirected graph supporting multiple edges between the same nodes
- `MultiDiGraph<T, E>`: Directed graph supporting multiple edges between the same nodes

### Key Methods

#### Graph Creation
```rust
// Create a directed graph
let directed_graph = Graph::<i32, f64>::new(true);

// Create an undirected graph
let undirected_graph = Graph::<String, u32>::new(false);

// Create a directed graph with a name
let named_graph = Graph::<String, f64>::with_name(true, "My Network");
```

#### Node Operations
```rust
// Add a node
let node_id = graph.add_node(data);

// Remove a node
let removed_data = graph.remove_node(node_id);

// Check if a node exists
let exists = graph.has_node(node_id);

// Get node data
let data_ref = graph.get_node_data(node_id);

// Find nodes matching a criterion
let matching_nodes = graph.find_nodes(|&data| data > 50);
```

#### Edge Operations
```rust
// Add an edge
graph.add_edge(from_node, to_node, weight);

// Remove an edge
let weight = graph.remove_edge(from_node, to_node);

// Check if an edge exists
let exists = graph.has_edge(from_node, to_node);

// Get edge weight
let weight = graph.get_edge_data(from_node, to_node);
```

#### Traversal
```rust
// Depth-first search
let dfs_result = graph.dfs(start_node);

// Breadth-first search
let bfs_result = graph.bfs(start_node);

// Shortest path
let (path, cost) = graph.shortest_path(from_node, to_node, |&weight| weight as f64);
```

#### MultiGraph-specific
```rust
// Add an edge (returns a unique edge ID)
let edge_id = multigraph.add_edge(from_node, to_node, weight);

// Remove a specific edge by ID
let removed = multigraph.remove_edge(from_node, to_node, &edge_id);

// Get all edges between two nodes
let edges = multigraph.edges_between(from_node, to_node);
```

#### DiGraph-specific
```rust
// Get successors (outgoing neighbors)
let successors = digraph.successors(node);

// Get predecessors (incoming neighbors)
let predecessors = digraph.predecessors(node);

// Get in-degree
let in_degree = digraph.in_degree(node);

// Get out-degree
let out_degree = digraph.out_degree(node);
```

## Performance Notes

In our testing, the parallel implementations sometimes performed slower than their sequential counterparts for small to medium-sized graphs. This could be due to:

1. The overhead of thread creation and management
2. Running in debug mode without optimizations
3. The specific hardware configuration
4. Implementation details of the parallel algorithms

For large graphs with complex operations, the parallel implementations may show better performance.

## Limitations and Bugs

During testing, we observed the following limitations:

1. The `to_undirected()` method on directed graphs appears to have an issue where edge counts don't match expectations
2. Some layout functions like `circular_layout` and `spring_layout` have different parameter requirements than documented
3. Parallel versions of algorithms may have different behavior than their sequential counterparts
