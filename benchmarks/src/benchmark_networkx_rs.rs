use networkx_rs::{Graph, DiGraph, NodeKey};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::{self, Write, BufReader};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::env;
use std::path::Path;
use serde::{Serialize, Deserialize};

/// Measure the execution time of a function
fn measure_time<F, T>(f: &mut F) -> (T, Duration)
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    (result, elapsed)
}

/// Statistics for benchmark runs
#[derive(Serialize, Deserialize, Debug)]
struct BenchmarkStats {
    mean: f64,
    median: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    runs: usize,
    all_times: Vec<f64>,
}

/// Benchmark results struct
struct BenchmarkResults {
    results: HashMap<String, Vec<f64>>, // Store milliseconds directly as f64
    detailed_results: HashMap<String, BenchmarkStats>, // Detailed statistics
    output_file: String,
    detailed_output_file: String,
    num_runs: usize,
}

impl BenchmarkResults {
    fn new(output_file: &str, num_runs: usize) -> Self {
        let detailed_output_file = output_file.replace(".csv", "_detailed.csv");
        BenchmarkResults {
            results: HashMap::new(),
            detailed_results: HashMap::new(),
            output_file: output_file.to_string(),
            detailed_output_file,
            num_runs,
        }
    }

    fn run_benchmark<F, T>(&mut self, name: &str, f: F) -> T
    where
        F: FnMut() -> T,
    {
        println!("Running benchmark: {}", name);
        
        // Warm-up run
        let mut f = f;  // Make f mutable
        let _ = f();
        
        // Actual benchmark runs
        let mut times = Vec::new();
        
        // First run to get the result
        let (result, elapsed) = measure_time(&mut f);
        // Convert to milliseconds and store
        let ms = elapsed.as_secs_f64() * 1000.0;
        times.push(ms);
        
        // Subsequent runs to measure performance
        for _ in 1..self.num_runs {
            let (_, elapsed) = measure_time(&mut f);
            let ms = elapsed.as_secs_f64() * 1000.0;
            times.push(ms);
        }
        
        // Calculate statistics
        let avg_ms = times.iter().sum::<f64>() / times.len() as f64;
        
        // Sort times for median and min/max
        let mut sorted_times = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let median_ms = if sorted_times.len() % 2 == 0 {
            let mid = sorted_times.len() / 2;
            (sorted_times[mid - 1] + sorted_times[mid]) / 2.0
        } else {
            sorted_times[sorted_times.len() / 2]
        };
        
        let min_ms = *sorted_times.first().unwrap_or(&0.0);
        let max_ms = *sorted_times.last().unwrap_or(&0.0);
        
        // Calculate standard deviation
        let variance = times.iter()
            .map(|x| {
                let diff = avg_ms - *x;
                diff * diff
            })
            .sum::<f64>() / times.len() as f64;
        let std_dev = variance.sqrt();
        
        println!("  Average time: {:.2} ms (median: {:.2}, std dev: {:.2}, min: {:.2}, max: {:.2})", 
                avg_ms, median_ms, std_dev, min_ms, max_ms);
        
        // Store the detailed results
        self.detailed_results.insert(name.to_string(), BenchmarkStats {
            mean: avg_ms,
            median: median_ms,
            std_dev,
            min: min_ms,
            max: max_ms,
            runs: times.len(),
            all_times: times.clone(),
        });
        
        // Store the result
        self.results.entry(name.to_string()).or_default().push(avg_ms);
        
        result
    }

    fn save_results(&self) -> io::Result<()> {
        // Ensure the directory exists
        if let Some(parent) = Path::new(&self.output_file).parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        // Save summary results
        let mut file = File::create(&self.output_file)?;
        
        // Write CSV header
        writeln!(file, "Benchmark,Time (ms)")?;
        
        // Write benchmark results
        for (name, times) in &self.results {
            let avg_time = times.iter().sum::<f64>() / times.len() as f64;
            writeln!(file, "{},{}", name, avg_time)?;
        }
        
        println!("Results saved to {}", self.output_file);
        
        // Save detailed results
        let mut detailed_file = File::create(&self.detailed_output_file)?;
        
        // Write CSV header for detailed results
        writeln!(detailed_file, "Benchmark,Mean (ms),Median (ms),StdDev (ms),Min (ms),Max (ms),Runs")?;
        
        // Write detailed benchmark results
        for (name, stats) in &self.detailed_results {
            writeln!(detailed_file, "{},{},{},{},{},{},{}",
                     name, stats.mean, stats.median, stats.std_dev, stats.min, stats.max, stats.runs)?;
        }
        
        println!("Detailed results saved to {}", self.detailed_output_file);
        
        Ok(())
    }
}

// Load graph seeds from Python
fn load_graph_seeds(seed_file: &str) -> HashMap<String, u64> {
    match File::open(seed_file) {
        Ok(file) => {
            let reader = BufReader::new(file);
            match serde_json::from_reader(reader) {
                Ok(seeds) => seeds,
                Err(e) => {
                    eprintln!("Error parsing graph seeds: {}, using defaults", e);
                    HashMap::new()
                }
            }
        },
        Err(e) => {
            eprintln!("Error opening graph seeds file: {}, using defaults", e);
            HashMap::new()
        }
    }
}

// Load source/target pairs from Python
fn load_source_target_nodes(file_path: &str) -> HashMap<String, (usize, usize)> {
    match File::open(file_path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            match serde_json::from_reader(reader) {
                Ok(nodes) => nodes,
                Err(e) => {
                    eprintln!("Error parsing source/target nodes: {}, using defaults", e);
                    HashMap::new()
                }
            }
        },
        Err(e) => {
            eprintln!("Error opening source/target file: {}, using defaults", e);
            HashMap::new()
        }
    }
}

/// Create a random graph with the given number of nodes and edges
fn create_random_graph<T: Clone + std::fmt::Debug>(
    nodes: usize,
    edges: usize,
    directed: bool,
    node_factory: &dyn Fn(usize) -> T,
    graph_id: &str,
    graph_seeds: &HashMap<String, u64>,
) -> Graph<T, f64> {
    let mut graph = Graph::new(directed);
    
    // Use the same seed as Python if available
    let seed = graph_seeds.get(graph_id).copied().unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Add nodes
    let mut node_keys = Vec::new();
    for i in 0..nodes {
        let node_data = node_factory(i);
        let key = graph.add_node(node_data);
        node_keys.push(key);
    }
    
    // Add random edges
    let mut edge_count = 0;
    let mut max_edges = nodes * (nodes - 1);
    if !directed {
        max_edges /= 2;
    }
    
    let edges = std::cmp::min(edges, max_edges);
    
    while edge_count < edges {
        let u_idx = rng.gen_range(0..nodes);
        let v_idx = rng.gen_range(0..nodes);
        
        if u_idx != v_idx {
            let u = node_keys[u_idx];
            let v = node_keys[v_idx];
            
            if !graph.has_edge(u, v) {
                let weight = rng.gen_range(1.0..10.0);
                graph.add_edge(u, v, weight);
                edge_count += 1;
            }
        }
    }
    
    graph
}

/// Create a random DiGraph with the given number of nodes and edges
fn create_random_digraph<T: Clone + std::fmt::Debug>(
    nodes: usize,
    edges: usize,
    node_factory: &dyn Fn(usize) -> T,
    graph_id: &str,
    graph_seeds: &HashMap<String, u64>,
) -> DiGraph<T, f64> {
    let mut digraph = DiGraph::new();
    
    // Use the same seed as Python if available
    let seed = graph_seeds.get(graph_id).copied().unwrap_or(42);
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Add nodes
    let mut node_keys = Vec::new();
    for i in 0..nodes {
        let node_data = node_factory(i);
        let key = digraph.add_node(node_data);
        node_keys.push(key);
    }
    
    // Add random edges
    let mut edge_count = 0;
    let max_edges = nodes * (nodes - 1);
    let edges = std::cmp::min(edges, max_edges);
    
    while edge_count < edges {
        let u_idx = rng.gen_range(0..nodes);
        let v_idx = rng.gen_range(0..nodes);
        
        if u_idx != v_idx {
            let u = node_keys[u_idx];
            let v = node_keys[v_idx];
            
            if !digraph.has_edge(u, v) {
                let weight = rng.gen_range(1.0..10.0);
                digraph.add_edge(u, v, weight);
                edge_count += 1;
            }
        }
    }
    
    digraph
}

struct Benchmarks {
    results: BenchmarkResults,
    graph_seeds: HashMap<String, u64>,
    source_target_nodes: HashMap<String, (usize, usize)>,
}

impl Benchmarks {
    fn new(output_file: &str, seeds_file: &str, nodes_file: &str, num_runs: usize) -> Self {
        Benchmarks {
            results: BenchmarkResults::new(output_file, num_runs),
            graph_seeds: load_graph_seeds(seeds_file),
            source_target_nodes: load_source_target_nodes(nodes_file),
        }
    }
    
    fn benchmark_graph_creation(&mut self, nodes: usize, edges: usize) {
        let graph_id = format!("undirected_{nodes}_{edges}");
        let node_factory = |i| i;
        
        let create_graph = || {
            create_random_graph(nodes, edges, false, &node_factory, &graph_id, &self.graph_seeds)
        };
        
        self.results.run_benchmark(&format!("graph_creation_{nodes}_{edges}"), create_graph);
    }
    
    fn benchmark_digraph_creation(&mut self, nodes: usize, edges: usize) {
        let graph_id = format!("directed_{nodes}_{edges}");
        let node_factory = |i| i;
        
        let create_digraph = || {
            create_random_digraph(nodes, edges, &node_factory, &graph_id, &self.graph_seeds)
        };
        
        self.results.run_benchmark(&format!("digraph_creation_{nodes}_{edges}"), create_digraph);
    }
    
    fn benchmark_shortest_path<T: Clone + std::fmt::Debug>(
        &mut self, 
        graph: &Graph<T, f64>, 
        source: NodeKey, 
        target: NodeKey
    ) {
        let graph = graph.clone();
        let find_path = || {
            graph.shortest_path(source, target, Some(&|_node1, _node2, weight| *weight))
        };
        
        self.results.run_benchmark("shortest_path", find_path);
    }
    
    fn benchmark_dfs<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>, source: NodeKey) {
        let graph = graph.clone();
        self.results.run_benchmark("dfs", || graph.dfs(source));
    }
    
    fn benchmark_bfs<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>, source: NodeKey) {
        let graph = graph.clone();
        self.results.run_benchmark("bfs", || graph.bfs(source));
    }
    
    fn benchmark_connected_components<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>) {
        let graph = graph.clone();
        
        let benchmark_name = if graph.is_directed() {
            "strongly_connected_components"
        } else {
            "connected_components"
        };
        
        let find_components = || {
            if graph.is_directed() {
                graph.strongly_connected_components()
            } else {
                graph.connected_components()
            }
        };
        
        self.results.run_benchmark(benchmark_name, find_components);
    }
    
    fn benchmark_betweenness_centrality<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>) {
        let graph = graph.clone();
        let run_betweenness = || {
            graph.betweenness_centrality(Some(&|_node1, _node2, weight| *weight))
        };
        
        self.results.run_benchmark("betweenness_centrality", run_betweenness);
    }
    
    fn benchmark_degree_centrality<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>) {
        let graph = graph.clone();
        let run_degree_centrality = || {
            graph.degree_centrality()
        };
        
        self.results.run_benchmark("degree_centrality", run_degree_centrality);
    }
    
    fn benchmark_minimum_spanning_tree<T: Clone + std::fmt::Debug>(&mut self, graph: &Graph<T, f64>) {
        if graph.is_directed() {
            return; // Skip for directed graphs
        }
        
        let graph = graph.clone();
        let run_mst = || {
            graph.minimum_spanning_tree(Some(&|_node1, _node2, weight| *weight))
        };
        
        self.results.run_benchmark("minimum_spanning_tree", run_mst);
    }
    
    fn benchmark_node_addition<T: Clone + std::fmt::Debug>(
        &mut self, 
        graph: &Graph<T, f64>, 
        num_nodes: usize,
        node_factory: &dyn Fn(usize) -> T,
    ) {
        let graph = graph.clone();
        let nodes_in_graph = graph.node_count();
        
        let add_nodes = || {
            let mut graph_copy = graph.clone();
            for i in 0..num_nodes {
                let node_data = node_factory(nodes_in_graph + i);
                graph_copy.add_node(node_data);
            }
            graph_copy
        };
        
        self.results.run_benchmark(&format!("add_{num_nodes}_nodes"), add_nodes);
    }
    
    fn benchmark_edge_addition<T: Clone + std::fmt::Debug>(
        &mut self, 
        graph: &Graph<T, f64>, 
        num_edges: usize,
    ) {
        let graph = graph.clone();
        let mut rng = StdRng::seed_from_u64(42);
        
        let node_keys: Vec<_> = graph.nodes().collect();
        let node_count = node_keys.len();
        
        if node_count < 2 {
            // Can't add edges without at least 2 nodes
            return;
        }
        
        let add_edges = || {
            let mut graph_copy = graph.clone();
            let mut edge_count = 0;
            
            while edge_count < num_edges {
                let u_idx = rng.gen_range(0..node_count);
                let v_idx = rng.gen_range(0..node_count);
                
                if u_idx != v_idx {
                    let u = node_keys[u_idx];
                    let v = node_keys[v_idx];
                    
                    if !graph_copy.has_edge(u, v) {
                        let weight = rng.gen_range(1.0..10.0);
                        graph_copy.add_edge(u, v, weight);
                        edge_count += 1;
                    }
                }
            }
            
            graph_copy
        };
        
        self.results.run_benchmark(&format!("add_{num_edges}_edges"), add_edges);
    }
    
    fn run_all_benchmarks(&mut self, graph_sizes: &[(usize, usize)]) {
        for &(nodes, edges) in graph_sizes {
            let graph_id = format!("{nodes}_{edges}");
            println!("\nBenchmarking with {} nodes and {} edges", nodes, edges);
            
            // Create undirected graph
            println!("\nUndirected Graph:");
            let undirected_id = format!("undirected_{nodes}_{edges}");
            let undirected_graph = create_random_graph(nodes, edges, false, &|i| i, &undirected_id, &self.graph_seeds);
            
            // Create directed graph 
            println!("\nDirected Graph:");
            let directed_id = format!("directed_{nodes}_{edges}");
            let directed_graph = create_random_digraph(nodes, edges, &|i| i, &directed_id, &self.graph_seeds);
            
            // Graph creation benchmarks
            self.benchmark_graph_creation(nodes, edges);
            self.benchmark_digraph_creation(nodes, edges);
            
            // Get source/target nodes for traversal - use same nodes as Python if available
            let (source_undirected, target_undirected) = match self.source_target_nodes.get(&undirected_id) {
                Some(&(s, t)) => {
                    // Convert to NodeKeys
                    let s_key = undirected_graph.get_node_key_by_index(s).unwrap_or_else(|| {
                        undirected_graph.nodes().next().unwrap()
                    });
                    
                    let t_key = undirected_graph.get_node_key_by_index(t).unwrap_or_else(|| {
                        let nodes: Vec<_> = undirected_graph.nodes().collect();
                        if nodes.len() > 1 && nodes[0] == s_key {
                            nodes[1]
                        } else {
                            nodes[0]
                        }
                    });
                    
                    (s_key, t_key)
                },
                None => {
                    // Just use the first two nodes
                    let nodes: Vec<_> = undirected_graph.nodes().collect();
                    if nodes.len() >= 2 {
                        (nodes[0], nodes[1])
                    } else if nodes.len() == 1 {
                        (nodes[0], nodes[0])
                    } else {
                        continue;  // Skip if graph is empty
                    }
                }
            };
            
            let (source_directed, target_directed) = match self.source_target_nodes.get(&directed_id) {
                Some(&(s, t)) => {
                    // Convert to NodeKeys
                    let s_key = directed_graph.get_node_key_by_index(s).unwrap_or_else(|| {
                        directed_graph.nodes().next().unwrap()
                    });
                    
                    let t_key = directed_graph.get_node_key_by_index(t).unwrap_or_else(|| {
                        let nodes: Vec<_> = directed_graph.nodes().collect();
                        if nodes.len() > 1 && nodes[0] == s_key {
                            nodes[1]
                        } else {
                            nodes[0]
                        }
                    });
                    
                    (s_key, t_key)
                },
                None => {
                    // Just use the first two nodes
                    let nodes: Vec<_> = directed_graph.nodes().collect();
                    if nodes.len() >= 2 {
                        (nodes[0], nodes[1])
                    } else if nodes.len() == 1 {
                        (nodes[0], nodes[0])
                    } else {
                        continue;  // Skip if graph is empty
                    }
                }
            };
            
            // Skip expensive operations for very large graphs
            if nodes <= 10000 {
                // Undirected graph operations
                self.benchmark_shortest_path(&undirected_graph, source_undirected, target_undirected);
                self.benchmark_dfs(&undirected_graph, source_undirected);
                self.benchmark_bfs(&undirected_graph, source_undirected);
                self.benchmark_connected_components(&undirected_graph);
                self.benchmark_betweenness_centrality(&undirected_graph);
                self.benchmark_degree_centrality(&undirected_graph);
                self.benchmark_minimum_spanning_tree(&undirected_graph);
                self.benchmark_node_addition(&undirected_graph, 100, &|i| i);
                self.benchmark_edge_addition(&undirected_graph, 100);
                
                // Directed graph operations
                self.benchmark_shortest_path(&directed_graph, source_directed, target_directed);
                self.benchmark_dfs(&directed_graph, source_directed);
                self.benchmark_bfs(&directed_graph, source_directed);
                self.benchmark_connected_components(&directed_graph);
                self.benchmark_betweenness_centrality(&directed_graph);
                self.benchmark_degree_centrality(&directed_graph);
                self.benchmark_node_addition(&directed_graph, 100, &|i| i);
                self.benchmark_edge_addition(&directed_graph, 100);
            } else {
                // For very large graphs, only run a subset of benchmarks
                println!("Skipping some benchmarks for large graph with {} nodes", nodes);
                self.benchmark_shortest_path(&undirected_graph, source_undirected, target_undirected);
                self.benchmark_dfs(&undirected_graph, source_undirected);
                self.benchmark_bfs(&undirected_graph, source_undirected);
                self.benchmark_connected_components(&undirected_graph);
                self.benchmark_node_addition(&undirected_graph, 100, &|i| i);
                self.benchmark_edge_addition(&undirected_graph, 100);
                
                self.benchmark_shortest_path(&directed_graph, source_directed, target_directed);
                self.benchmark_dfs(&directed_graph, source_directed);
                self.benchmark_bfs(&directed_graph, source_directed);
                self.benchmark_connected_components(&directed_graph);
                self.benchmark_node_addition(&directed_graph, 100, &|i| i);
                self.benchmark_edge_addition(&directed_graph, 100);
            }
        }
    }
}

trait GraphExt<T, E> {
    fn get_node_key_by_index(&self, index: usize) -> Option<NodeKey>;
}

impl<T: Clone + std::fmt::Debug, E: Clone + std::fmt::Debug> GraphExt<T, E> for Graph<T, E> {
    fn get_node_key_by_index(&self, index: usize) -> Option<NodeKey> {
        let mut node_iter = self.nodes();
        for _ in 0..index {
            if node_iter.next().is_none() {
                return None;
            }
        }
        node_iter.next()
    }
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    let mut output_file = "benchmarks/results/rust_benchmark_results.csv".to_string();
    let mut graph_sizes = vec![(100, 500), (500, 2500), (1000, 5000)];
    let seeds_file = "results/graph_seeds.json".to_string();
    let nodes_file = "results/source_target.json".to_string();
    let mut num_runs = 30;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                if i + 1 < args.len() {
                    output_file = args[i + 1].clone();
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "--sizes" => {
                if i + 1 < args.len() {
                    // Parse graph sizes in format "100,500 500,2500 1000,5000"
                    graph_sizes = args[i + 1]
                        .split_whitespace()
                        .filter_map(|s| {
                            let parts: Vec<&str> = s.split(',').collect();
                            if parts.len() == 2 {
                                let nodes = parts[0].parse::<usize>().ok()?;
                                let edges = parts[1].parse::<usize>().ok()?;
                                Some((nodes, edges))
                            } else {
                                None
                            }
                        })
                        .collect();
                    i += 2;
                } else {
                    i += 1;
                }
            },
            "--runs" => {
                if i + 1 < args.len() {
                    num_runs = args[i + 1].parse().unwrap_or(30);
                    i += 2;
                } else {
                    i += 1;
                }
            },
            _ => {
                i += 1;
            }
        }
    }
    
    println!("Running benchmarks with graph sizes: {:?}", graph_sizes);
    println!("Results will be saved to: {}", output_file);
    println!("Using graph seeds from: {}", seeds_file);
    println!("Using source/target nodes from: {}", nodes_file);
    println!("Number of runs per benchmark: {}", num_runs);
    
    // Run benchmarks
    let mut benchmarks = Benchmarks::new(&output_file, &seeds_file, &nodes_file, num_runs);
    benchmarks.run_all_benchmarks(&graph_sizes);
    
    // Save results
    if let Err(e) = benchmarks.results.save_results() {
        eprintln!("Error saving results: {}", e);
    }
} 