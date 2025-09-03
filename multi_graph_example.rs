use networkx_rs::{MultiGraph, MultiDiGraph};
use std::fmt::Debug;

// Example demonstrating MultiGraph usage (undirected graph with multiple edges between nodes)
fn multi_graph_example() {
    println!("=== MultiGraph Example ===");
    
    // Create a MultiGraph representing a transportation network
    let mut graph = MultiGraph::<String, String>::new();
    
    let london = graph.add_node("London".to_string());
    let paris = graph.add_node("Paris".to_string());
    let new_york = graph.add_node("New York".to_string());
    
    // London to Paris connections
    let eurostar_id = graph.add_edge(london, paris, "Eurostar Train".to_string());
    let air_france_id = graph.add_edge(london, paris, "Air France Flight".to_string());
    let ferry_id = graph.add_edge(london, paris, "Ferry".to_string());
    
    // London to New York connections
    let ba_flight_id = graph.add_edge(london, new_york, "British Airways Flight".to_string());
    let virgin_id = graph.add_edge(london, new_york, "Virgin Atlantic Flight".to_string());
    
    // Paris to New York connection
    let delta_id = graph.add_edge(paris, new_york, "Delta Airlines Flight".to_string());
    
    println!("Number of nodes: {}", graph.node_count());
    println!("Number of edges: {}", graph.edge_count());
    
    println!("\nConnections between London and Paris:");
    let edges_london_paris = graph.edges_between(london, paris);
    for (edge_id, transport) in edges_london_paris {
        println!("  Edge ID: {} - {}", edge_id, transport);
    }
    
    println!("\nAll connections from London:");
    let neighbors = graph.neighbors(london);
    for neighbor in neighbors {
        let neighbor_name = graph.get_node_data(neighbor).unwrap();
        let edges = graph.edges_between(london, neighbor);
        println!("  To {}: {} connections", neighbor_name, edges.len());
        for (edge_id, transport) in edges {
            println!("    - Edge ID: {} - {}", edge_id, transport);
        }
    }
    
    println!("\nRemoving Eurostar Train connection...");
    let removed = graph.remove_edge(london, paris, &eurostar_id);
    println!("Removed: {}", removed.unwrap());
    
    println!("\nUpdated connections between London and Paris:");
    let updated_edges = graph.edges_between(london, paris);
    for (edge_id, transport) in updated_edges {
        println!("  Edge ID: {} - {}", edge_id, transport);
    }
}

// Example demonstrating MultiDiGraph usage (directed graph with multiple edges between nodes)
fn multi_digraph_example() {
    println!("\n=== MultiDiGraph Example ===");
    
    // Create a MultiDiGraph representing a network flow
    #[derive(Debug, Clone, Default)]
    struct Traffic {
        name: String,
        bandwidth: f64,
    }
    
    let mut graph = MultiDiGraph::<String, Traffic>::new();
    
    let server = graph.add_node("Server".to_string());
    let router = graph.add_node("Router".to_string());
    let client = graph.add_node("Client".to_string());
    
    // Server to Router connections (outbound traffic)
    graph.add_edge(server, router, Traffic { name: "HTTP Traffic".to_string(), bandwidth: 10.5 });
    graph.add_edge(server, router, Traffic { name: "Database Traffic".to_string(), bandwidth: 5.2 });
    graph.add_edge(server, router, Traffic { name: "Email Traffic".to_string(), bandwidth: 2.1 });
    
    // Router to Server connections (inbound traffic)
    graph.add_edge(router, server, Traffic { name: "HTTP Requests".to_string(), bandwidth: 4.7 });
    graph.add_edge(router, server, Traffic { name: "API Calls".to_string(), bandwidth: 3.2 });
    
    // Router to Client connections
    graph.add_edge(router, client, Traffic { name: "HTTP Response".to_string(), bandwidth: 8.1 });
    graph.add_edge(router, client, Traffic { name: "Media Streaming".to_string(), bandwidth: 15.3 });
    
    // Client to Router connections
    graph.add_edge(client, router, Traffic { name: "HTTP Requests".to_string(), bandwidth: 3.5 });
    graph.add_edge(client, router, Traffic { name: "Upload Traffic".to_string(), bandwidth: 1.8 });
    
    println!("Number of nodes: {}", graph.node_count());
    println!("Number of edges: {}", graph.edge_count());
    
    // Print outgoing connections from Server
    println!("\nOutgoing connections from Server:");
    let successors = graph.successors(server);
    for succ in successors {
        let neighbor_name = graph.get_node_data(succ).unwrap();
        println!("  To {}:", neighbor_name);
        
        let edges = graph.edges_between(server, succ);
        for (edge_id, traffic) in edges {
            println!("    - Edge ID: {} - {} ({} Mbps)", edge_id, traffic.name, traffic.bandwidth);
        }
    }
    
    // Print incoming connections to Server
    println!("\nIncoming connections to Server:");
    let predecessors = graph.predecessors(server);
    for pred in predecessors {
        let neighbor_name = graph.get_node_data(pred).unwrap();
        println!("  From {}:", neighbor_name);
        
        let edges = graph.edges_between(pred, server);
        for (edge_id, traffic) in edges {
            println!("    - Edge ID: {} - {} ({} Mbps)", edge_id, traffic.name, traffic.bandwidth);
        }
    }
    
    // Calculate total bandwidth for each node
    println!("\nTotal bandwidth analysis:");
    for node in graph.nodes() {
        let node_name = graph.get_node_data(node).unwrap();
        
        // Calculate outgoing bandwidth
        let mut outgoing_bandwidth = 0.0;
        for succ in graph.successors(node) {
            let edges = graph.edges_between(node, succ);
            for (_, traffic) in edges {
                outgoing_bandwidth += traffic.bandwidth;
            }
        }
        
        // Calculate incoming bandwidth
        let mut incoming_bandwidth = 0.0;
        for pred in graph.predecessors(node) {
            let edges = graph.edges_between(pred, node);
            for (_, traffic) in edges {
                incoming_bandwidth += traffic.bandwidth;
            }
        }
        
        println!("  {}: Incoming {:.1} Mbps, Outgoing {:.1} Mbps", 
                 node_name, incoming_bandwidth, outgoing_bandwidth);
    }
}

fn main() {
    multi_graph_example();
    multi_digraph_example();
} 