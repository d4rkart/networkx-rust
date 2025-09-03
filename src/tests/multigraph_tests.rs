use crate::MultiGraph;

#[cfg(test)]
mod multigraph_tests {
    use super::*;
    
    // Basic functionality tests
    
    #[test]
    fn test_new_multigraph() {
        let graph = MultiGraph::<i32, f64>::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_basic_node_operations() {
        let mut graph = MultiGraph::<String, i32>::new();
        
        // Add nodes
        let n1 = graph.add_node("Node 1".to_string());
        let n2 = graph.add_node("Node 2".to_string());
        
        // Check node count
        assert_eq!(graph.node_count(), 2);
        
        // Check node existence
        assert!(graph.has_node(n1));
        assert!(graph.has_node(n2));
        assert!(!graph.has_node(999)); // Non-existent node
        
        // Check node data
        assert_eq!(graph.get_node_data(n1), Some(&"Node 1".to_string()));
        assert_eq!(graph.get_node_data(n2), Some(&"Node 2".to_string()));
        
        // Remove node
        assert_eq!(graph.remove_node(n1), Some("Node 1".to_string()));
        assert_eq!(graph.node_count(), 1);
        assert!(!graph.has_node(n1));
        assert!(graph.has_node(n2));
    }
    
    #[test]
    fn test_basic_edge_operations() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        // Add multiple edges between the same nodes
        let key1 = graph.add_edge(n1, n2, "Edge 1".to_string());
        let key2 = graph.add_edge(n1, n2, "Edge 2".to_string());
        
        // Keys should be different
        assert_ne!(key1, key2);
        
        // Check edge count
        assert_eq!(graph.number_of_edges(n1, n2), 2);
        
        // Check edge existence
        assert!(graph.has_edge(n1, n2, &key1));
        assert!(graph.has_edge(n1, n2, &key2));
        
        // Check edge data
        assert_eq!(graph.get_edge_weight(n1, n2, &key1), Some(&"Edge 1".to_string()));
        assert_eq!(graph.get_edge_weight(n1, n2, &key2), Some(&"Edge 2".to_string()));
        
        // Remove one edge
        assert_eq!(graph.remove_edge(n1, n2, &key1), Some("Edge 1".to_string()));
        assert_eq!(graph.number_of_edges(n1, n2), 1);
        assert!(!graph.has_edge(n1, n2, &key1));
        assert!(graph.has_edge(n1, n2, &key2));
    }
    
    // #[test]
    // fn test_multi_edge_bidirectionality() {
    //     let mut graph = MultiGraph::<char, i32>::new();
        
    //     let a = graph.add_node('A');
    //     let b = graph.add_node('B');
        
    //     // Add multiple edges in both directions
    //     let key1 = graph.add_edge(a, b, 1);
    //     let key2 = graph.add_edge(a, b, 2);
    //     let key3 = graph.add_edge(b, a, 3);
        
    //     // Verify edges from a to b exist with correct data
    //     assert!(graph.has_edge(a, b, &key1));
    //     assert!(graph.has_edge(a, b, &key2));
    //     assert_eq!(graph.get_edge_weight(a, b, &key1), Some(&1));
    //     assert_eq!(graph.get_edge_weight(a, b, &key2), Some(&2));
        
    //     // Verify edges from b to a exist with correct data
    //     assert!(graph.has_edge(b, a, &key3));
    //     assert_eq!(graph.get_edge_weight(b, a, &key3), Some(&3));
        
    //     // Verify the number of edges between nodes
    //     assert_eq!(graph.number_of_edges(a, b), 2);
    //     assert_eq!(graph.number_of_edges(b, a), 1);
        
    //     // Verify the edges between a and b
    //     let all_edges_a_b = graph.edges_between(a, b);
    //     assert_eq!(all_edges_a_b.len(), 2, "Should have exactly 2 edges from a to b");
        
    //     let all_edges_b_a = graph.edges_between(b, a);
    //     assert_eq!(all_edges_b_a.len(), 1, "Should have exactly 1 edge from b to a");
        
    //     // Get total edges in the graph to verify
    //     let total_edges = graph.edges();
    //     assert_eq!(total_edges.len(), 3, "Should have 3 total edges in the graph");
    // }
    
    #[test]
    fn test_node_removal_cascade() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        graph.add_edge(n1, n2, "Edge 1-2".to_string());
        graph.add_edge(n1, n2, "Edge 1-2 alt".to_string());
        graph.add_edge(n2, n3, "Edge 2-3".to_string());
        
        // In undirected graph, each edge is counted in both directions
        assert_eq!(graph.edge_count(), 6);
        
        // Remove node and check that its edges are also removed
        graph.remove_node(n2);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0); // All edges involving n2 are gone
    }
    
    #[test]
    fn test_neighbors() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        
        graph.add_edge(n1, n2, "Edge 1-2".to_string());
        graph.add_edge(n1, n2, "Edge 1-2 alt".to_string());
        graph.add_edge(n1, n3, "Edge 1-3".to_string());
        
        // Check neighbors
        let neighbors = graph.neighbors(n1);
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n2));
        assert!(neighbors.contains(&n3));
        
        let neighbors = graph.neighbors(n2);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&n1));
        
        let neighbors = graph.neighbors(n3);
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&n1));
    }
    
    #[test]
    fn test_clear() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        graph.add_edge(n1, n2, "Edge 1".to_string());
        graph.add_edge(n1, n2, "Edge 2".to_string());
        
        assert_eq!(graph.node_count(), 2);
        // Each edge is counted in both directions in undirected graph
        assert_eq!(graph.edge_count(), 4);
        
        graph.clear();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_edges() {
        let mut graph = MultiGraph::<char, i32>::new();
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        
        let _key1 = graph.add_edge(a, b, 1);
        let _key2 = graph.add_edge(a, b, 2);
        let _key3 = graph.add_edge(b, c, 3);
        
        // Check all edges
        let edges = graph.edges();
        assert_eq!(edges.len(), 3);
        
        // Check edges between specific nodes
        let edges_a_b = graph.edges_between(a, b);
        assert_eq!(edges_a_b.len(), 2);
        
        // Verify edge contents - using integer references in the vector
        let mut weights: Vec<&i32> = edges_a_b.iter()
            .map(|(_, w)| *w)
            .collect();
        weights.sort();
        assert_eq!(weights, vec![&1, &2]);
    }
    
    #[test]
    fn test_subgraph() {
        let mut graph = MultiGraph::<String, i32>::new();
        
        // Add some nodes
        let n1 = graph.add_node("Node 1".to_string());
        let n2 = graph.add_node("Node 2".to_string());
        let n3 = graph.add_node("Node 3".to_string());
        
        // Add some edges
        let edge1 = graph.add_edge(n1, n2, 1);
        graph.add_edge(n2, n3, 2);
        
        // Create a subgraph with just some of the nodes
        let subgraph = graph.subgraph(vec![n1, n2]);
        
        // Test node membership
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph.get_node_data(n1).is_some());
        assert!(subgraph.get_node_data(n2).is_some());
        assert!(subgraph.get_node_data(n3).is_none());
        
        // Test edge preservation
        assert!(subgraph.has_edge(n1, n2, &edge1));
        
        // Count the edges to verify
        let edges = subgraph.edges();
        assert_eq!(edges.len(), 1, "Should have 1 edge in the subgraph");
    }
    
    // Parallel operations tests
    
    #[test]
    fn test_neighbors_par() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        let n3 = graph.add_node(3);
        let n4 = graph.add_node(4);
        let n5 = graph.add_node(5);
        
        graph.add_edge(n1, n2, "1-2".to_string());
        graph.add_edge(n1, n3, "1-3".to_string());
        graph.add_edge(n1, n4, "1-4".to_string());
        graph.add_edge(n1, n5, "1-5".to_string());
        
        // Get neighbors using parallel method
        let neighbors = graph.neighbors_par(n1);
        
        // Check that all neighbors are found
        assert_eq!(neighbors.len(), 4);
        assert!(neighbors.contains(&n2));
        assert!(neighbors.contains(&n3));
        assert!(neighbors.contains(&n4));
        assert!(neighbors.contains(&n5));
        
        // Compare with sequential neighbors
        let seq_neighbors = graph.neighbors(n1);
        assert_eq!(neighbors, seq_neighbors);
    }
    
    #[test]
    fn test_edges_par() {
        let mut graph = MultiGraph::<char, i32>::new();
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        let c = graph.add_node('C');
        
        let _key1 = graph.add_edge(a, b, 1);
        let _key2 = graph.add_edge(a, b, 2);
        let _key3 = graph.add_edge(b, c, 3);
        let _key4 = graph.add_edge(c, a, 4);
        
        // Get edges using parallel method
        let par_edges = graph.edges_par();
        
        // Check that all edges are found
        assert_eq!(par_edges.len(), 4);
        
        // Verify against sequential edges
        let seq_edges = graph.edges();
        
        // Sort both vectors to compare them (since parallel processing might change the order)
        let mut par_sorted = par_edges.clone();
        let mut seq_sorted = seq_edges.clone();
        
        par_sorted.sort_by(|a, b| {
            if a.0 != b.0 {
                a.0.cmp(&b.0)
            } else if a.1 != b.1 {
                a.1.cmp(&b.1)
            } else {
                a.2.cmp(&b.2)
            }
        });
        
        seq_sorted.sort_by(|a, b| {
            if a.0 != b.0 {
                a.0.cmp(&b.0)
            } else if a.1 != b.1 {
                a.1.cmp(&b.1)
            } else {
                a.2.cmp(&b.2)
            }
        });
        
        assert_eq!(par_sorted.len(), seq_sorted.len());
        for i in 0..par_sorted.len() {
            assert_eq!(par_sorted[i].0, seq_sorted[i].0); // from
            assert_eq!(par_sorted[i].1, seq_sorted[i].1); // to
            assert_eq!(par_sorted[i].2, seq_sorted[i].2); // key
            assert_eq!(par_sorted[i].3, seq_sorted[i].3); // weight
        }
    }
    
    #[test]
    fn test_edge_count_par() {
        let mut graph = MultiGraph::<i32, i32>::new();
        
        // Create a graph with many edges
        for i in 0..10 {
            graph.add_node(i);
        }
        
        // Add edges between nodes
        for i in 0..9 {
            for j in 0..3 {
                graph.add_edge(i, i+1, j);
            }
        }
        
        // Count edges using parallel method
        let par_count = graph.edge_count_par();
        
        // Check against sequential count
        let seq_count = graph.edge_count();
        
        assert_eq!(par_count, seq_count);
        // In undirected graph, each edge is counted in both directions
        assert_eq!(par_count, 54); // 9 node pairs with 3 edges each, counted both ways
    }
    
    #[test]
    fn test_clear_par() {
        let mut graph = MultiGraph::<i32, String>::new();
        
        let n1 = graph.add_node(1);
        let n2 = graph.add_node(2);
        
        graph.add_edge(n1, n2, "Edge 1".to_string());
        graph.add_edge(n1, n2, "Edge 2".to_string());
        
        assert_eq!(graph.node_count(), 2);
        // Each edge is counted in both directions in undirected graph
        assert_eq!(graph.edge_count(), 4);
        
        // Use parallel clear
        graph.clear_par();
        
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
    
    #[test]
    fn test_edges_between_par() {
        let mut graph = MultiGraph::<char, i32>::new();
        
        let a = graph.add_node('A');
        let b = graph.add_node('B');
        
        let _key1 = graph.add_edge(a, b, 1);
        let _key2 = graph.add_edge(a, b, 2);
        let _key3 = graph.add_edge(a, b, 3);
        
        // Get edges between nodes using parallel method
        let par_edges = graph.edges_between_par(a, b);
        
        // Check that all edges are found
        assert_eq!(par_edges.len(), 3);
        
        // Sort edges by weight to make comparison easier
        let mut edges_sorted = par_edges.clone();
        edges_sorted.sort_by(|a, b| a.1.cmp(&b.1));
        
        assert_eq!(edges_sorted[0].1, 1);
        assert_eq!(edges_sorted[1].1, 2);
        assert_eq!(edges_sorted[2].1, 3);
    }
    
    #[test]
    fn test_subgraph_par() {
        let mut graph = MultiGraph::<String, i32>::new();
        
        // Add some nodes
        let n1 = graph.add_node("Node 1".to_string());
        let n2 = graph.add_node("Node 2".to_string());
        let n3 = graph.add_node("Node 3".to_string());
        
        // Add some edges
        let edge1 = graph.add_edge(n1, n2, 1);
        graph.add_edge(n2, n3, 2);
        
        // Create a subgraph with just some of the nodes
        let subgraph = graph.subgraph_par(vec![n1, n2]);
        
        // Test node membership
        assert_eq!(subgraph.node_count(), 2);
        assert!(subgraph.get_node_data(n1).is_some());
        assert!(subgraph.get_node_data(n2).is_some());
        assert!(subgraph.get_node_data(n3).is_none());
        
        // Test edge preservation
        assert!(subgraph.has_edge(n1, n2, &edge1));
        
        // Count the edges to verify
        let edges = subgraph.edges();
        assert_eq!(edges.len(), 1, "Should have 1 edge in the subgraph");
    }
    
    #[test]
    fn test_nodes_par() {
        let mut graph = MultiGraph::<i32, i32>::new();
        
        // Add many nodes
        let mut added_nodes = Vec::new();
        for i in 0..100 {
            added_nodes.push(graph.add_node(i));
        }
        
        // Get nodes using parallel method
        let par_nodes = graph.nodes_par();
        
        // Check that all nodes are found
        assert_eq!(par_nodes.len(), 100);
        
        // Check that parallel and sequential results match
        let seq_nodes = graph.nodes();
        
        let mut par_sorted = par_nodes.clone();
        let mut seq_sorted = seq_nodes.clone();
        
        par_sorted.sort();
        seq_sorted.sort();
        
        assert_eq!(par_sorted, seq_sorted);
    }
    
    #[test]
    fn test_get_node_data_par() {
        let mut graph = MultiGraph::<String, i32>::new();
        
        let n1 = graph.add_node("Node 1".to_string());
        let _n2 = graph.add_node("Node 2".to_string());
        
        // Get node data using parallel method
        let par_data = graph.get_node_data_par(n1);
        assert_eq!(par_data, Some("Node 1".to_string()));
        
        // Compare with sequential method
        let seq_data = graph.get_node_data(n1).cloned();
        assert_eq!(par_data, seq_data);
        
        // Test non-existent node
        let no_data = graph.get_node_data_par(999);
        assert_eq!(no_data, None);
    }
    
    #[test]
    fn test_large_parallel_operations() {
        let mut graph = MultiGraph::<i32, i32>::new();
        
        // Create a larger graph
        for i in 0..1000 {
            graph.add_node(i);
        }
        
        // Add edges
        for i in 0..999 {
            for j in 0..5 {  // Add 5 edges between consecutive nodes
                graph.add_edge(i, i+1, j);
            }
        }
        
        // Perform parallel operations
        let start = std::time::Instant::now();
        let par_edges = graph.edges_par();
        let par_duration = start.elapsed();
        
        let start = std::time::Instant::now();
        let seq_edges = graph.edges();
        let seq_duration = start.elapsed();
        
        // Just assert they're the same length
        assert_eq!(par_edges.len(), seq_edges.len());
        assert_eq!(par_edges.len(), 4995); // 999 node pairs with 5 edges each
        
        println!("Parallel edges: {:?}, Sequential edges: {:?}", par_duration, seq_duration);
        
        // Test parallel edge counting
        let start = std::time::Instant::now();
        let par_count = graph.edge_count_par();
        let par_count_duration = start.elapsed();
        
        let start = std::time::Instant::now();
        let seq_count = graph.edge_count();
        let seq_count_duration = start.elapsed();
        
        assert_eq!(par_count, seq_count);
        
        println!("Parallel count: {:?}, Sequential count: {:?}", par_count_duration, seq_count_duration);
    }
} 