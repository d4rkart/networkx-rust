#!/usr/bin/env python3
import networkx as nx
import numpy as np
import time
import random
import argparse
import csv
import os
import json
from collections import defaultdict

def measure_time(func, *args, **kwargs):
    """Measure execution time of a function."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, (end - start) * 1000  # Time in ms

class NetworkXBenchmark:
    def __init__(self, output_file="results/python_benchmark_results.csv", seed=42, num_runs=30):
        self.results = defaultdict(list)
        self.detailed_results = defaultdict(dict)
        self.output_file = output_file
        self.detailed_output_file = output_file.replace('.csv', '_detailed.csv')
        # Use fixed seed for reproducibility
        self.seed = seed
        random.seed(self.seed)
        
        # Number of times to run each benchmark for more accurate results
        self.num_runs = num_runs
        
        # Create graph seeds to ensure both Python and Rust use the same graphs
        self.graph_seeds = {}
        
        # Store source/target pairs for traversal algorithms
        self.source_target_nodes = {}
    
    def run_benchmark(self, name, func, *args, **kwargs):
        """Run a benchmark and save results."""
        print(f"Running benchmark: {name}")
        
        # Run once to warm up
        func(*args, **kwargs)
        
        # Run the actual benchmark
        times = []
        for _ in range(self.num_runs):  # Run multiple times for more accurate timing
            _, time_ms = measure_time(func, *args, **kwargs)
            times.append(time_ms)
        
        # Calculate statistics
        times_array = np.array(times)
        avg_time = np.mean(times_array)
        median_time = np.median(times_array)
        std_dev = np.std(times_array)
        min_time = np.min(times_array)
        max_time = np.max(times_array)
        
        # Store detailed results
        self.detailed_results[name] = {
            'mean': avg_time,
            'median': median_time,
            'std_dev': std_dev,
            'min': min_time,
            'max': max_time,
            'runs': self.num_runs,
            'all_times': times
        }
        
        self.results[name].append(avg_time)
        print(f"  Average time: {avg_time:.2f} ms (median: {median_time:.2f}, std dev: {std_dev:.2f}, min: {min_time:.2f}, max: {max_time:.2f})")
        return avg_time
    
    def save_results(self):
        """Save benchmark results to CSV file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save summary results
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Benchmark', 'Time (ms)'])
            for name, times in self.results.items():
                writer.writerow([name, np.mean(times)])
        print(f"Results saved to {self.output_file}")
        
        # Save detailed results
        with open(self.detailed_output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Benchmark', 'Mean (ms)', 'Median (ms)', 'StdDev (ms)', 'Min (ms)', 'Max (ms)', 'Runs'])
            for name, stats in self.detailed_results.items():
                writer.writerow([
                    name, 
                    stats['mean'], 
                    stats['median'], 
                    stats['std_dev'], 
                    stats['min'], 
                    stats['max'], 
                    stats['runs']
                ])
        print(f"Detailed results saved to {self.detailed_output_file}")
        
        # Save graph seeds for Rust to use
        seed_file = os.path.join(os.path.dirname(self.output_file), "graph_seeds.json")
        with open(seed_file, 'w') as f:
            json.dump(self.graph_seeds, f)
        print(f"Graph seeds saved to {seed_file}")
        
        # Save source/target nodes for traversal algorithms
        nodes_file = os.path.join(os.path.dirname(self.output_file), "source_target.json")
        with open(nodes_file, 'w') as f:
            json.dump(self.source_target_nodes, f)
        print(f"Source/target nodes saved to {nodes_file}")

    def create_random_graph(self, nodes, edges, directed=False, graph_id=None):
        """Create a random graph with specified number of nodes and edges."""
        if graph_id is None:
            graph_id = f"{'directed' if directed else 'undirected'}_{nodes}_{edges}"
            
        # If we've already created this graph, use the same seed
        if graph_id not in self.graph_seeds:
            self.graph_seeds[graph_id] = random.randint(0, 100000)
            
        # Use the seed for this specific graph
        local_seed = self.graph_seeds[graph_id]
        random.seed(local_seed)
        
        if directed:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        
        # Add nodes
        for i in range(nodes):
            G.add_node(i)
        
        # Add random edges
        edge_count = 0
        max_edges = nodes * (nodes - 1)
        if not directed:
            max_edges //= 2
        
        edges = min(edges, max_edges)
        
        while edge_count < edges:
            u = random.randint(0, nodes - 1)
            v = random.randint(0, nodes - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, weight=random.uniform(1, 10))
                edge_count += 1
        
        # Store source/target nodes for path algorithms
        if graph_id not in self.source_target_nodes and nodes > 1:
            source = random.randint(0, nodes - 1)
            target = random.randint(0, nodes - 1)
            while target == source:  # Ensure source and target are different
                target = random.randint(0, nodes - 1)
            self.source_target_nodes[graph_id] = (source, target)
        
        # Reset random seed
        random.seed(self.seed)
        return G

    def benchmark_graph_creation(self, nodes, edges):
        """Benchmark graph creation."""
        graph_id = f"undirected_{nodes}_{edges}"
        
        def create_graph():
            return self.create_random_graph(nodes, edges, directed=False, graph_id=graph_id)
        
        self.run_benchmark(f"graph_creation_{nodes}_{edges}", create_graph)
    
    def benchmark_digraph_creation(self, nodes, edges):
        """Benchmark directed graph creation."""
        graph_id = f"directed_{nodes}_{edges}"
        
        def create_digraph():
            return self.create_random_graph(nodes, edges, directed=True, graph_id=graph_id)
        
        self.run_benchmark(f"digraph_creation_{nodes}_{edges}", create_digraph)
    
    def benchmark_shortest_path(self, G, source, target):
        """Benchmark shortest path algorithm."""
        def find_shortest_path():
            return nx.shortest_path(G, source=source, target=target, weight='weight')
        
        return self.run_benchmark(f"shortest_path", find_shortest_path)
    
    def benchmark_shortest_path_length(self, G, source, target):
        """Benchmark shortest path length calculation."""
        def find_shortest_path_length():
            return nx.shortest_path_length(G, source=source, target=target, weight='weight')
        
        return self.run_benchmark(f"shortest_path_length", find_shortest_path_length)
    
    def benchmark_all_pairs_shortest_path(self, G):
        """Benchmark all-pairs shortest path calculation."""
        def find_all_pairs_shortest_path():
            return dict(nx.all_pairs_shortest_path(G))
        
        return self.run_benchmark(f"all_pairs_shortest_path", find_all_pairs_shortest_path)
    
    def benchmark_dfs(self, G, source):
        """Benchmark depth-first search."""
        def run_dfs():
            return list(nx.dfs_preorder_nodes(G, source=source))
        
        return self.run_benchmark(f"dfs", run_dfs)
    
    def benchmark_bfs(self, G, source):
        """Benchmark breadth-first search."""
        def run_bfs():
            return list(nx.bfs_tree(G, source=source).nodes())
        
        return self.run_benchmark(f"bfs", run_bfs)
    
    def benchmark_pagerank(self, G):
        """Benchmark PageRank algorithm."""
        def run_pagerank():
            return nx.pagerank(G, weight='weight')
        
        return self.run_benchmark(f"pagerank", run_pagerank)
    
    def benchmark_connected_components(self, G):
        """Benchmark finding connected components."""
        def find_components():
            if G.is_directed():
                return list(nx.strongly_connected_components(G))
            else:
                return list(nx.connected_components(G))
        
        component_type = "strongly_connected" if G.is_directed() else "connected"
        return self.run_benchmark(f"{component_type}_components", find_components)
    
    def benchmark_betweenness_centrality(self, G):
        """Benchmark betweenness centrality calculation."""
        def run_betweenness():
            return nx.betweenness_centrality(G, weight='weight')
        
        return self.run_benchmark(f"betweenness_centrality", run_betweenness)
    
    def benchmark_closeness_centrality(self, G):
        """Benchmark closeness centrality calculation."""
        def run_closeness():
            return nx.closeness_centrality(G, distance='weight')
        
        return self.run_benchmark(f"closeness_centrality", run_closeness)
    
    def benchmark_clustering(self, G):
        """Benchmark clustering coefficient calculation."""
        def run_clustering():
            return nx.clustering(G, weight='weight')
        
        return self.run_benchmark(f"clustering", run_clustering)
    
    def benchmark_degree_centrality(self, G):
        """Benchmark degree centrality calculation."""
        def run_degree_centrality():
            return nx.degree_centrality(G)
        
        return self.run_benchmark(f"degree_centrality", run_degree_centrality)
    
    def benchmark_minimum_spanning_tree(self, G):
        """Benchmark minimum spanning tree calculation (undirected only)."""
        if G.is_directed():
            return None  # Skip for directed graphs
            
        def run_mst():
            return nx.minimum_spanning_tree(G, weight='weight')
        
        return self.run_benchmark(f"minimum_spanning_tree", run_mst)
    
    def benchmark_node_addition(self, G, num_nodes):
        """Benchmark adding nodes to a graph."""
        def add_nodes():
            G_copy = G.copy()
            for i in range(len(G), len(G) + num_nodes):
                G_copy.add_node(i)
            return G_copy
        
        return self.run_benchmark(f"add_{num_nodes}_nodes", add_nodes)
    
    def benchmark_edge_addition(self, G, num_edges):
        """Benchmark adding edges to a graph."""
        def add_edges():
            G_copy = G.copy()
            nodes = list(G_copy.nodes())
            n = len(nodes)
            
            edge_count = 0
            while edge_count < num_edges:
                u = nodes[random.randint(0, n-1)]
                v = nodes[random.randint(0, n-1)]
                if u != v and not G_copy.has_edge(u, v):
                    G_copy.add_edge(u, v, weight=random.uniform(1, 10))
                    edge_count += 1
            
            return G_copy
        
        return self.run_benchmark(f"add_{num_edges}_edges", add_edges)
    
    def run_all_benchmarks(self, graph_sizes=[(100, 500), (500, 2500), (1000, 5000)]):
        """Run all benchmarks with different graph sizes."""
        # Store source and target nodes for each graph size
        self.source_target_nodes = {}
        
        for nodes, edges in graph_sizes:
            graph_id = f"{nodes}_{edges}"
            print(f"\nBenchmarking with {nodes} nodes and {edges} edges")
            
            # Undirected graph
            print("\nUndirected Graph:")
            G_undirected = self.create_random_graph(nodes, edges, directed=False, graph_id=f"undirected_{nodes}_{edges}")
            
            # Directed graph
            print("\nDirected Graph:")
            G_directed = self.create_random_graph(nodes, edges, directed=True, graph_id=f"directed_{nodes}_{edges}")
            
            # Graph creation benchmarks
            self.benchmark_graph_creation(nodes, edges)
            self.benchmark_digraph_creation(nodes, edges)
            
            # Get source and target nodes for this graph
            undirected_id = f"undirected_{nodes}_{edges}"
            directed_id = f"directed_{nodes}_{edges}"
            
            source_undirected, target_undirected = self.source_target_nodes.get(undirected_id, (0, min(1, nodes-1)))
            source_directed, target_directed = self.source_target_nodes.get(directed_id, (0, min(1, nodes-1)))
            
            # Skip expensive operations for very large graphs
            if nodes <= 10000:
                # Undirected graph operations
                self.benchmark_shortest_path(G_undirected, source_undirected, target_undirected)
                self.benchmark_shortest_path_length(G_undirected, source_undirected, target_undirected)
                if nodes <= 5000:  # Skip for very large graphs
                    self.benchmark_all_pairs_shortest_path(G_undirected)
                self.benchmark_dfs(G_undirected, source_undirected)
                self.benchmark_bfs(G_undirected, source_undirected)
                self.benchmark_pagerank(G_undirected)
                self.benchmark_connected_components(G_undirected)
                self.benchmark_betweenness_centrality(G_undirected) 
                self.benchmark_closeness_centrality(G_undirected)
                self.benchmark_clustering(G_undirected)
                self.benchmark_degree_centrality(G_undirected)
                self.benchmark_minimum_spanning_tree(G_undirected)
                self.benchmark_node_addition(G_undirected, 100)
                self.benchmark_edge_addition(G_undirected, 100)
                
                # Directed graph operations
                self.benchmark_shortest_path(G_directed, source_directed, target_directed)
                self.benchmark_shortest_path_length(G_directed, source_directed, target_directed)
                if nodes <= 5000:  # Skip for very large graphs
                    self.benchmark_all_pairs_shortest_path(G_directed)
                self.benchmark_dfs(G_directed, source_directed)
                self.benchmark_bfs(G_directed, source_directed)
                self.benchmark_pagerank(G_directed)
                self.benchmark_connected_components(G_directed)
                self.benchmark_betweenness_centrality(G_directed)
                self.benchmark_closeness_centrality(G_directed)
                self.benchmark_degree_centrality(G_directed)
                self.benchmark_node_addition(G_directed, 100)
                self.benchmark_edge_addition(G_directed, 100)
            else:
                # For very large graphs, only run a subset of benchmarks
                print(f"Skipping some benchmarks for large graph with {nodes} nodes")
                self.benchmark_shortest_path(G_undirected, source_undirected, target_undirected)
                self.benchmark_dfs(G_undirected, source_undirected)
                self.benchmark_bfs(G_undirected, source_undirected)
                self.benchmark_connected_components(G_undirected)
                self.benchmark_node_addition(G_undirected, 100)
                self.benchmark_edge_addition(G_undirected, 100)
                
                self.benchmark_shortest_path(G_directed, source_directed, target_directed)
                self.benchmark_dfs(G_directed, source_directed)
                self.benchmark_bfs(G_directed, source_directed)
                self.benchmark_connected_components(G_directed)
                self.benchmark_node_addition(G_directed, 100)
                self.benchmark_edge_addition(G_directed, 100)

def main():
    parser = argparse.ArgumentParser(description='Benchmark NetworkX')
    parser.add_argument('--sizes', type=str, default='100,500 500,2500 1000,5000',
                       help='Graph sizes to benchmark in format: "nodes1,edges1 nodes2,edges2 ..."')
    parser.add_argument('--output', type=str, default='results/python_benchmark_results.csv',
                       help='Output CSV file')
    parser.add_argument('--runs', type=int, default=30,
                       help='Number of runs for each benchmark')
    
    args = parser.parse_args()
    
    # Parse graph sizes
    sizes = []
    for size_str in args.sizes.split():
        nodes, edges = map(int, size_str.split(','))
        sizes.append((nodes, edges))
    
    # Run benchmarks
    benchmark = NetworkXBenchmark(output_file=args.output, num_runs=args.runs)
    benchmark.run_all_benchmarks(graph_sizes=sizes)
    benchmark.save_results()

if __name__ == "__main__":
    main() 