use networkx_rs::Graph;
use std::time::Instant;

// Example demonstrating parallel graph processing capabilities
fn main() {
    println!("=== Parallel Graph Processing Example ===");
    
    // Create a large graph for performance testing
    println!("Creating a large graph...");
    let mut graph = create_large_graph(10_000_000, 50_000_000, false);
    
    println!("Graph created with {} nodes and approximately {} edges", 
             graph.node_count(), graph.edge_count());
    
    // Find nodes meeting certain criteria (sequential)
    println!("\nFinding nodes with value > 9000 (sequential)...");
    let start = Instant::now();
    let matching_nodes = graph.find_nodes(|&val| val > 9000);
    let seq_elapsed = start.elapsed();
    println!("Found {} nodes in {:?}", matching_nodes.len(), seq_elapsed);
    
    // Find nodes meeting the same criteria (parallel)
    println!("\nFinding nodes with value > 9000 (parallel)...");
    let start = Instant::now();
    let matching_nodes_par = graph.find_nodes_par(|&val| val > 9000);
    let par_elapsed = start.elapsed();
    println!("Found {} nodes in {:?}", matching_nodes_par.len(), par_elapsed);
    println!("Speedup: {:.2}x", seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64());
    
    // Test BFS traversal (sequential)
    println!("\nPerforming BFS traversal (sequential)...");
    let start_node = 0; // Start from the first node
    let start = Instant::now();
    let bfs_result = graph.bfs(start_node);
    let seq_elapsed = start.elapsed();
    println!("BFS traversed {} nodes in {:?}", bfs_result.len(), seq_elapsed);
    
    // Test parallel BFS
    println!("\nPerforming BFS traversal (parallel)...");
    let start = Instant::now();
    let bfs_result_par = graph.bfs_par(start_node);
    let par_elapsed = start.elapsed();
    println!("BFS traversed {} nodes in {:?}", bfs_result_par.len(), par_elapsed);
    println!("Speedup: {:.2}x", seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64());
    
    // Remove some edges to create multiple components
    println!("\nRemoving some edges to create multiple components...");
    for i in 0..10 {
        let threshold = i * 1000;
        for j in 0..1000 {
            let from = threshold + j;
            let to = from + 1000;
            if graph.has_edge(from, to) {
                graph.remove_edge(from, to);
            }
        }
    }
    
    // Calculate connected components (sequentially)
    println!("\nCalculating connected components (sequential)...");
    // This is a simple implementation for comparison
    let start = Instant::now();
    let components = sequential_connected_components(&graph);
    let seq_elapsed = start.elapsed();
    println!("Found {} components in {:?}", components.len(), seq_elapsed);
    
    // Calculate connected components (parallel)
    println!("\nCalculating connected components (parallel)...");
    let start = Instant::now();
    let components_par = graph.connected_components_par();
    let par_elapsed = start.elapsed();
    println!("Found {} components in {:?}", components_par.len(), par_elapsed);
    println!("Speedup: {:.2}x", seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64());
    
    // Subgraph extraction (sequential vs parallel)
    println!("\nExtracting subgraph of first 5000 nodes (sequential)...");
    let nodes_to_extract: Vec<usize> = (0..5000).collect();
    let start = Instant::now();
    let subgraph = graph.subgraph(nodes_to_extract.clone());
    let seq_elapsed = start.elapsed();
    println!("Created subgraph with {} nodes and {} edges in {:?}", 
             subgraph.node_count(), subgraph.edge_count(), seq_elapsed);
    
    println!("\nExtracting subgraph of first 5000 nodes (parallel)...");
    let start = Instant::now();
    let subgraph_par = graph.subgraph_par(nodes_to_extract);
    let par_elapsed = start.elapsed();
    println!("Created subgraph with {} nodes and {} edges in {:?}", 
             subgraph_par.node_count(), subgraph_par.edge_count(), par_elapsed);
    println!("Speedup: {:.2}x", seq_elapsed.as_secs_f64() / par_elapsed.as_secs_f64());
}

// Create a large graph for performance testing
fn create_large_graph(num_nodes: usize, num_edges: usize, directed: bool) -> Graph<usize, f64> {
    let mut graph = Graph::<usize, f64>::new(directed);
    
    // Add nodes
    for i in 0..num_nodes {
        graph.add_node(i);
    }
    
    // Add random edges
    let mut edge_count = 0;
    while edge_count < num_edges {
        let from = rand::random::<usize>() % num_nodes;
        let to = rand::random::<usize>() % num_nodes;
        
        if from != to && !graph.has_edge(from, to) {
            graph.add_edge(from, to, rand::random::<f64>());
            edge_count += 1;
        }
    }
    
    graph
}

// Simple sequential implementation of connected components for comparison
fn sequential_connected_components<T, E>(graph: &Graph<T, E>) -> Vec<Vec<usize>> 
where 
    T: Clone + std::fmt::Debug,
    E: Clone + std::fmt::Debug,
{
    let mut components = Vec::new();
    let mut visited = std::collections::HashSet::new();
    
    for node in graph.nodes() {
        if !visited.contains(&node) {
            let mut component = Vec::new();
            let mut queue = std::collections::VecDeque::new();
            
            queue.push_back(node);
            visited.insert(node);
            
            while let Some(current) = queue.pop_front() {
                component.push(current);
                
                for neighbor in graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
            
            components.push(component);
        }
    }
    
    components
} 