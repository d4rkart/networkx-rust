use networkx_rs::{Graph, DiGraph, NodeKey};
use std::collections::HashMap;

fn basic_graph_operations() {
    println!("=== Example 1: Basic Graph Operations ===");
    
    let mut graph = Graph::<String, f64>::with_name(false, "Social Network");
    
    let alice = graph.add_node("Alice".to_string());
    let bob = graph.add_node("Bob".to_string());
    let charlie = graph.add_node("Charlie".to_string());
    let dave = graph.add_node("Dave".to_string());
    
    graph.add_edge(alice, bob, 0.8);
    graph.add_edge(alice, charlie, 0.6);
    graph.add_edge(bob, charlie, 0.9);
    graph.add_edge(bob, dave, 0.4);
    
    println!("Graph name: {}", graph.name());
    println!("Number of nodes: {}", graph.node_count());
    println!("Number of edges: {}", graph.edge_count());
    
    println!("\nNode data:");
    for key in graph.nodes() {
        println!("Node {}: {}", key, graph.get_node_data(key).unwrap());
    }
    
    println!("\nEdge data:");
    for (from, to, weight) in graph.edges() {
        let from_name = graph.get_node_data(from).unwrap();
        let to_name = graph.get_node_data(to).unwrap();
        println!("{} -- {} (weight: {})", from_name, to_name, weight);
    }
    
    println!("\nNode degrees:");
    let degrees = graph.degree(None);
    for (key, degree) in &degrees {
        let name = graph.get_node_data(*key).unwrap();
        println!("{}: {}", name, degree);
    }

    println!("\nNodes with degree > 1:");
    for (key, degree) in degrees {
        if degree > 1 {
            let name = graph.get_node_data(key).unwrap();
            println!("{} (degree: {})", name, degree);
        }
    }
}

fn graph_traversals() {
    println!("\n=== Example 2: Graph Traversals ===");
    
    let mut graph = Graph::<char, ()>::new(true);
    
    let a = graph.add_node('A');
    let b = graph.add_node('B');
    let c = graph.add_node('C');
    let d = graph.add_node('D');
    let e = graph.add_node('E');
    let f = graph.add_node('F');
    
    graph.add_edge(a, b, ());
    graph.add_edge(a, c, ());
    graph.add_edge(b, d, ());
    graph.add_edge(b, e, ());
    graph.add_edge(c, f, ());
    
    println!("\nDFS traversal from A:");
    let dfs_result = graph.dfs(a);
    for key in dfs_result {
        print!("{} ", graph.get_node_data(key).unwrap());
    }
    println!();
    
    println!("\nBFS traversal from A:");
    let bfs_result = graph.bfs(a);
    for key in bfs_result {
        print!("{} ", graph.get_node_data(key).unwrap());
    }
    println!();
    
    let mut digraph = DiGraph::<char, ()>::new();
    
    let a = digraph.add_node('A');
    let b = digraph.add_node('B');
    let c = digraph.add_node('C');
    let d = digraph.add_node('D');
    let e = digraph.add_node('E');
    let f = digraph.add_node('F');
    
    digraph.add_edge(a, b, ());
    digraph.add_edge(a, c, ());
    digraph.add_edge(b, d, ());
    digraph.add_edge(b, e, ());
    digraph.add_edge(c, f, ());
    
    println!("\nSuccessors of A:");
    for node in digraph.successors(a) {
        print!("{} ", digraph.get_node_data(node).unwrap());
    }
    println!();
    
    println!("Predecessors of D:");
    for node in digraph.predecessors(d) {
        print!("{} ", digraph.get_node_data(node).unwrap());
    }
    println!();
}

fn shortest_path_example() {
    println!("\n=== Example 3: Shortest Path ===");
    
    #[derive(Debug, Clone)]
    struct City {
        name: String,
    }
    
    #[derive(Debug, Clone)]
    struct Road {
        distance: u32,
    }
    
    let mut graph = Graph::<City, Road>::new(false);
    
    let cities = vec![
        ("New York", graph.add_node(City { name: "New York".to_string() })),
        ("Boston", graph.add_node(City { name: "Boston".to_string() })),
        ("Philadelphia", graph.add_node(City { name: "Philadelphia".to_string() })),
        ("Washington DC", graph.add_node(City { name: "Washington DC".to_string() })),
        ("Chicago", graph.add_node(City { name: "Chicago".to_string() })),
        ("Detroit", graph.add_node(City { name: "Detroit".to_string() })),
    ];
    
    let city_map: HashMap<&str, NodeKey> = cities.iter().cloned().collect();
    
    graph.add_edge(city_map["New York"], city_map["Boston"], Road { distance: 215 });
    graph.add_edge(city_map["New York"], city_map["Philadelphia"], Road { distance: 95 });
    graph.add_edge(city_map["Boston"], city_map["Detroit"], Road { distance: 613 });
    graph.add_edge(city_map["Philadelphia"], city_map["Washington DC"], Road { distance: 140 });
    graph.add_edge(city_map["Washington DC"], city_map["Chicago"], Road { distance: 701 });
    graph.add_edge(city_map["Detroit"], city_map["Chicago"], Road { distance: 283 });
    
    let (path, total_distance) = graph.shortest_path(
        city_map["New York"], 
        city_map["Chicago"], 
        |road| road.distance as f64
    ).unwrap();
    
    println!("Shortest path from New York to Chicago:");
    println!("Total distance: {} miles", total_distance);
    println!("Route:");
    
    for (i, &key) in path.iter().enumerate() {
        let city = graph.get_node_data(key).unwrap();
        print!("{}", city.name);
        if i < path.len() - 1 {
            let next_key = path[i + 1];
            let road = graph.get_edge_data(key, next_key).unwrap();
            print!(" --({} miles)--> ", road.distance);
        }
    }
    println!();
}

fn main() {
    basic_graph_operations();
    graph_traversals();
    shortest_path_example();
} 