# NetworkX Rust Library Testing

This repository contains examples showcasing the functionality of the NetworkX Rust library, which is a port of the popular Python NetworkX library for graph processing.

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